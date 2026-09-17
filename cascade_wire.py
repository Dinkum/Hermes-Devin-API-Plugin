"""Protobuf + Connect-streaming primitives for Codeium's Cascade chat API.

The Devin CLI (``chisel``, the ``devin`` binary) does not talk HTTP+JSON to its own backend.
Its inference path is Connect-over-HTTP/1.1 carrying **protobuf**:

    POST {api_server}/exa.api_server_pb.ApiServerService/AssignModel     (unary)
    POST {api_server}/exa.api_server_pb.ApiServerService/GetChatMessage  (server stream)

Unary calls use ``content-type: application/proto`` and a bare protobuf body. The streaming
chat call uses ``content-type: application/connect+proto`` and the Connect envelope framing::

    request  : [flag=0x01 gzip][uint32 BE length][gzip(protobuf)]
    response : repeated [flag][uint32 BE length][payload];  flag&0x01 -> payload is gzipped,
               flag&0x02 -> payload is a JSON end-of-stream trailer (the error channel)

Field numbers below are transcribed from the protos published in can1357/oh-my-pi
(``packages/ai/src/providers/devin/proto/exa/...``) and verified live against
``server.codeium.com`` — see ``references/cascade-api.md`` in the ``devin-cascade-api`` skill.

Only the small field set Hermes needs is encoded here, by hand: no protobuf runtime dependency.
"""

from __future__ import annotations

import gzip
import struct
from typing import Any

# ── Connect framing ───────────────────────────────────────────────────────────────────────────
FLAG_COMPRESSED = 0x01
FLAG_END_STREAM = 0x02
CONNECT_CONTENT_TYPE = "application/connect+proto"
PROTO_CONTENT_TYPE = "application/proto"
CONNECT_PROTOCOL_VERSION = "1"
# The reference client (and the CLI's own Go-flavoured stack) announces itself this way; keep it
# stable so the backend's per-client behaviour stays in the lane we validated.
CONNECT_USER_AGENT = "connect-go/1.18.1 (go1.26.3)"

ASSIGN_MODEL_PATH = "/exa.api_server_pb.ApiServerService/AssignModel"
GET_CHAT_MESSAGE_PATH = "/exa.api_server_pb.ApiServerService/GetChatMessage"

# ── Enums (values are wire numbers) ───────────────────────────────────────────────────────────
SOURCE_USER = 1
SOURCE_SYSTEM = 2          # also how assistant turns are transmitted
SOURCE_TOOL = 4
SOURCE_SYSTEM_PROMPT = 5

REQUEST_TYPE_CASCADE = 5
PLANNER_MODE_DEFAULT = 1

STOP_REASON_MAX_TOKENS = 3
STOP_REASON_FUNCTION_CALL = 10

CACHE_CONTROL_EPHEMERAL = 1

# `<|...|>` sentinels the backend uses per model family; sending the family's set back keeps the
# model from emitting them into the visible answer.
DEFAULT_STOP_PATTERNS = ("<|user|>", "<|bot|>", "<|context_request|>", "<|endoftext|>", "<|end_of_turn|>")


# ── Protobuf writer ──────────────────────────────────────────────────────────────────────────
def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _key(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def pb_str(field: int, value: str | bytes) -> bytes:
    """Length-delimited (wire type 2) field."""
    raw = value.encode() if isinstance(value, str) else value
    return _key(field, 2) + _varint(len(raw)) + raw


def pb_msg(field: int, payload: bytes) -> bytes:
    """Embedded message: same encoding as a string, named for intent."""
    return pb_str(field, payload)


def pb_int(field: int, value: int) -> bytes:
    """Varint (wire type 0) field. Protobuf int32 fields are sign-extended to 64 bits."""
    return _key(field, 0) + _varint(value & 0xFFFFFFFFFFFFFFFF)


def pb_bool(field: int, value: bool) -> bytes:
    return pb_int(field, 1 if value else 0)


def pb_double(field: int, value: float) -> bytes:
    """fixed64 (wire type 1) — ``temperature``/``top_p`` are doubles in the proto."""
    return _key(field, 1) + struct.pack("<d", float(value))


# ── Protobuf reader ──────────────────────────────────────────────────────────────────────────
def pb_parse(buffer: bytes) -> dict[int, list[Any]]:
    """Decode a protobuf message into ``{field_number: [values]}``.

    Wire type 0 → int, 2 → bytes, 1 → double, 5 → float. Unknown groups are an error, which is
    the correct response to a message this decoder does not understand.
    """
    out: dict[int, list[Any]] = {}
    index, size = 0, len(buffer)
    while index < size:
        key, index = _read_varint(buffer, index)
        field, wire_type = key >> 3, key & 0x07
        if wire_type == 0:
            value, index = _read_varint(buffer, index)
        elif wire_type == 2:
            length, index = _read_varint(buffer, index)
            value = buffer[index:index + length]
            index += length
        elif wire_type == 1:
            value = struct.unpack("<d", buffer[index:index + 8])[0]
            index += 8
        elif wire_type == 5:
            value = struct.unpack("<f", buffer[index:index + 4])[0]
            index += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} (field {field})")
        out.setdefault(field, []).append(value)
    return out


def _read_varint(buffer: bytes, index: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        byte = buffer[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7


def pb_text(buffer: bytes, field: int) -> str:
    """First value of *field* as text, or "" — response fields are optional."""
    raw = (pb_parse(buffer).get(field) or [b""])[0]
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def pb_first(buffer: bytes, field: int) -> Any:
    values = pb_parse(buffer).get(field) or []
    return values[0] if values else None


# ── Envelope helpers ─────────────────────────────────────────────────────────────────────────
def encode_envelope(payload: bytes, *, compress: bool = True) -> bytes:
    body = gzip.compress(payload) if compress else payload
    return bytes([FLAG_COMPRESSED if compress else 0]) + struct.pack(">I", len(body)) + body


def iter_envelopes(buffer: bytes):
    """Yield ``(flag, payload)`` per Connect frame, inflating compressed payloads."""
    index, size = 0, len(buffer)
    while index + 5 <= size:
        flag = buffer[index]
        (length,) = struct.unpack(">I", buffer[index + 1:index + 5])
        payload = buffer[index + 5:index + 5 + length]
        index += 5 + length
        if flag & FLAG_COMPRESSED:
            # Connect marks a compressed frame with 0x01 even when it is also the end-of-stream
            # frame (0x03), so inflate before anyone decides what the payload means.
            try:
                payload = gzip.decompress(payload)
            except OSError:
                pass
        yield flag, payload


# ── Request builders ─────────────────────────────────────────────────────────────────────────
def metadata_message(
    api_key: str, *, ide_name: str, ide_version: str, extension_name: str,
    os_name: str, ide_type: str = "chisel", locale: str = "en", session_id: str = "",
) -> bytes:
    """``exa.codeium_common_pb.Metadata`` — the identity tuple the backend gates behaviour on.

    ``ide_name='devin-cli'`` / ``ide_type='chisel'`` is what unlocks the CLI model surface and
    router assignment; an older Windsurf identity reaches a different (and here, broken) lane.
    A stale ``ide_version`` is answered with ``failed_precondition: ... please update your editor``.
    """
    parts = [
        pb_str(1, ide_name),          # ide_name
        pb_str(2, extension_version := ide_version),  # extension_version
        pb_str(3, api_key),           # api_key (already carries its scheme prefix)
        pb_str(4, locale),            # locale
        pb_str(5, os_name),           # os
        pb_bool(6, False),            # disable_telemetry
        pb_str(7, extension_version), # ide_version
    ]
    if session_id:
        parts.append(pb_str(10, session_id))
    parts += [
        pb_str(12, extension_name),   # extension_name
        pb_str(28, ide_type),         # ide_type
    ]
    return b"".join(parts)


def chat_prompt_message(
    *, message_id: str, source: int, prompt: str = "", tool_call_id: str = "",
    tool_calls: list[tuple[str, str, str]] | None = None, thinking: str = "",
    tool_result_is_error: bool = False,
) -> bytes:
    """``exa.chat_pb.ChatMessagePrompt``: one conversation turn.

    ``tool_calls`` is ``[(id, name, arguments_json), ...]`` on an assistant turn; ``tool_call_id``
    pairs a tool *result* with the call it answers. Assistant turns travel with
    ``source=SOURCE_SYSTEM`` and user turns with ``SOURCE_USER`` — that is how the reference
    client encodes roles, since the source enum has no "assistant" member.
    """
    parts = [pb_str(1, message_id), pb_int(2, source), pb_str(3, prompt)]
    for call_id, name, arguments in tool_calls or []:
        parts.append(pb_msg(6, tool_call_message(call_id=call_id, name=name, arguments_json=arguments)))
    if tool_call_id:
        parts.append(pb_str(7, tool_call_id))
    if thinking:
        parts.append(pb_str(11, thinking))
    if tool_result_is_error:
        parts.append(pb_bool(9, True))
    return b"".join(parts)


def tool_call_message(*, call_id: str, name: str, arguments_json: str) -> bytes:
    """``exa.codeium_common_pb.ChatToolCall``."""
    return b"".join([pb_str(1, call_id), pb_str(2, name), pb_str(3, arguments_json)])


def tool_definition_message(*, name: str, description: str, parameters_schema: str, strict: bool = False) -> bytes:
    """``exa.chat_pb.ChatToolDefinition`` — an OpenAI function schema as a JSON string."""
    parts = [pb_str(1, name), pb_str(2, description), pb_str(3, parameters_schema)]
    if strict:
        parts.append(pb_bool(12, True))
    return b"".join(parts)


def tool_choice_message(choice: Any) -> bytes:
    """``exa.chat_pb.ChatToolChoice`` — ``option_name`` ("auto"/"none"/"required") or a tool name."""
    if isinstance(choice, dict):
        function = choice.get("function") if isinstance(choice.get("function"), dict) else {}
        if isinstance(function.get("name"), str) and function["name"].strip():
            return pb_str(2, function["name"].strip())
        choice = choice.get("type") or "auto"
    text = str(choice or "auto").strip().lower()
    if text in ("required", "any"):
        text = "required"
    elif text in ("none", "auto"):
        text = text
    else:
        return pb_str(2, text)
    return pb_str(1, text)


def completion_configuration_message(
    *, max_tokens: int, temperature: float, top_p: float = 1.0, stop_patterns: tuple[str, ...] = DEFAULT_STOP_PATTERNS,
) -> bytes:
    """``exa.codeium_common_pb.CompletionConfiguration`` — sampling knobs, doubles on the wire."""
    parts = [
        pb_int(1, 1),                # num_completions
        pb_int(2, max_tokens),       # max_tokens
        pb_int(3, 200),              # max_newlines
        pb_double(5, temperature),   # temperature
        pb_double(6, temperature),   # first_temperature
        pb_int(7, 50),               # top_k
        pb_double(8, top_p),         # top_p
    ]
    parts += [pb_str(9, pattern) for pattern in stop_patterns]
    parts.append(pb_double(11, 1.0))  # fim_eot_prob_threshold
    return b"".join(parts)


def assign_model_request(
    metadata: bytes, *, model_router_uid: str, cascade_id: str, prompt: bytes | None = None,
) -> bytes:
    """``GetChatMessageRequest``-sibling ``AssignModelRequest``: router uid in, JWT + uid out."""
    parts = [pb_msg(1, metadata), pb_str(2, model_router_uid), pb_str(3, cascade_id)]
    if prompt is not None:
        parts.append(pb_msg(5, prompt))
    return b"".join(parts)


def parse_assignment(payload: bytes) -> tuple[str, str, list[str]]:
    """``AssignModelResponse`` → ``(assignment_jwt, model_uid, harness_uids)``."""
    assignment = pb_first(payload, 1)
    if not isinstance(assignment, bytes):
        return "", "", []
    jwt = pb_text(assignment, 1)
    model_uid = pb_text(assignment, 2)
    harness = [value.decode("utf-8", "replace") for value in (pb_parse(assignment).get(3) or [])]
    return jwt, model_uid, harness


def get_chat_message_request(
    *,
    metadata: bytes,
    system_prompt: str,
    prompts: list[bytes],
    chat_model_uid: str,
    cascade_id: str,
    execution_id: str,
    tools: list[bytes] | None = None,
    tool_choice: bytes | None = None,
    configuration: bytes | None = None,
    model_assignment_jwt: str = "",
    disable_parallel_tool_calls: bool = False,
    system_prompt_cache_options: bytes | None = None,
) -> bytes:
    """Assemble ``GetChatMessageRequest`` in the field order the reference client uses."""
    parts = [pb_msg(1, metadata)]
    if system_prompt:
        parts.append(pb_str(2, system_prompt))
    parts += [pb_msg(3, prompt) for prompt in prompts]
    if configuration is not None:
        parts.append(pb_msg(8, configuration))
    parts += [pb_msg(10, tool) for tool in tools or []]
    if disable_parallel_tool_calls:
        parts.append(pb_bool(11, True))
    if tool_choice is not None:
        parts.append(pb_msg(12, tool_choice))
    parts.append(pb_msg(13, system_prompt_cache_options or pb_int(1, CACHE_CONTROL_EPHEMERAL)))
    parts += [
        pb_int(7, REQUEST_TYPE_CASCADE),
        pb_str(16, cascade_id),
        pb_int(20, PLANNER_MODE_DEFAULT),
        pb_str(21, chat_model_uid),
        pb_str(22, execution_id),
    ]
    if model_assignment_jwt:
        parts.append(pb_str(26, model_assignment_jwt))
    return b"".join(parts)


# ── Response decoding ────────────────────────────────────────────────────────────────────────
class ChatStream:
    """Decoded ``GetChatMessageResponse`` stream: text, reasoning, tool calls, usage, stop reason."""

    def __init__(self) -> None:
        self.text = ""
        self.thinking = ""
        self.tool_calls: list[dict[str, str]] = []
        self.stop_reason: int | None = None
        self.actual_model_uid = ""
        self.usage: dict[str, int] = {}
        self.messages = 0
        self.error = ""

    @property
    def finished_with_tool_calls(self) -> bool:
        return bool(self.tool_calls) or self.stop_reason == STOP_REASON_FUNCTION_CALL


def decode_chat_stream(payload: bytes) -> ChatStream:
    """Walk every Connect frame of a GetChatMessage body into a :class:`ChatStream`.

    Tool calls arrive the way OpenAI streams them: one frame opens the call (``id`` + ``name``)
    and later frames append ``arguments_json`` fragments::

        {1: "get_weather_0#…", 2: "get_weather"}      <- opens the call
        {3: "{"} {3: '"city": "'} {3: "T"} …          <- argument fragments

    Concatenating frames naively invents one call per fragment (and drops the argument-less
    fragments for having no name), so fragments are accumulated per call id here.
    """
    stream = ChatStream()
    order: list[str] = []
    partials: dict[str, dict[str, str]] = {}

    def _open(call_id: str) -> dict[str, str]:
        entry = partials.get(call_id)
        if entry is None:
            entry = partials[call_id] = {"id": call_id, "name": "", "arguments": "", "last": ""}
            order.append(call_id)
        return entry

    for flag, frame in iter_envelopes(payload):
        if flag & FLAG_END_STREAM:
            trailer = frame.decode("utf-8", "replace").strip()
            if trailer and trailer != "{}":
                stream.error = _trailer_error(trailer)
            continue
        if not frame:
            continue
        message = pb_parse(frame)
        stream.messages += 1
        for value in message.get(3) or []:
            stream.text += value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        for value in message.get(9) or []:
            stream.thinking += value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        for value in message.get(6) or []:
            if not isinstance(value, bytes):
                continue
            call = pb_parse(value)
            call_id = _as_text((call.get(1) or [b""])[0])
            name = _as_text((call.get(2) or [b""])[0])
            arguments = _as_text((call.get(3) or [b""])[0])
            if call_id:
                entry = _open(call_id)
            elif order:
                entry = partials[order[-1]]      # a continuation of the call already open
            else:
                continue
            if name:
                if not entry["name"]:
                    entry["name"] = name
                elif name != entry["name"] and not entry["name"].endswith(name):
                    entry["name"] += name
            if arguments and arguments != entry["last"]:
                entry["arguments"] += arguments
                entry["last"] = arguments
        if (reason := message.get(5)) is not None:
            stream.stop_reason = reason[0]
        if (uid := message.get(23)) is not None:
            stream.actual_model_uid = uid[0].decode("utf-8", "replace") if isinstance(uid[0], bytes) else str(uid[0])
        for value in message.get(7) or []:
            if isinstance(value, bytes):
                stats = pb_parse(value)
                stream.usage = {
                    "input_tokens": int((stats.get(2) or [0])[0]),
                    "output_tokens": int((stats.get(3) or [0])[0]),
                    "cache_write_tokens": int((stats.get(4) or [0])[0]),
                    "cache_read_tokens": int((stats.get(5) or [0])[0]),
                    "model_uid": pb_text(value, 9),
                }
    stream.tool_calls = [
        {"id": entry["id"], "name": entry["name"], "arguments": entry["arguments"] or "{}"}
        for entry in (partials[key] for key in order)
    ]
    return stream


def _as_text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else ("" if value is None else str(value))


def _trailer_error(trailer: str) -> str:
    """Connect end-of-stream trailers carry ``{"error": {...}}``; unwrap it, else return "".

    A trailer without an ``error`` member is a normal end of stream (``{}``), and a payload that
    is not JSON at all is not a trailer the client should interpret — both mean "no error".
    """
    import json

    try:
        parsed = json.loads(trailer)
    except ValueError:
        return ""
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        code = str(error.get("code") or "").strip()
        message = str(error.get("message") or "").strip() or "unspecified error"
        return f"{code}: {message}" if code else message
    return ""
