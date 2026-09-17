"""OpenAI-client-shaped facade over Codeium's Cascade chat API (the Devin CLI's own backend).

Hermes treats a model provider as an OpenAI client: it calls
``client.chat.completions.create(model=…, messages=[…], tools=[…], stream=…)`` and reads back
``choices[0].message`` (``content`` + OpenAI-shaped ``tool_calls``) or stream chunks. This class
implements exactly that surface, so Hermes' own agent loop — its tools, its approvals, its
transcript — drives the model. Nothing here is ACP: it is HTTPS + protobuf straight to
``{api_server}/exa.api_server_pb.ApiServerService/GetChatMessage``, the same endpoint the
``devin`` CLI uses.

Two calls per turn, both verified live against ``server.codeium.com``:

1. ``AssignModel`` (unary, ``application/proto``) — router models are not valid chat uids, so the
   server resolves them: ``model_router_uid="adaptive"`` → ``{assignment_jwt, model_uid}``.
   Cached per (router, cascade) because the JWT is bound to the cascade id.
2. ``GetChatMessage`` (Connect stream, ``application/connect+proto``) — the conversation plus
   ``tools`` goes up as protobuf; text, reasoning, tool calls and usage stream back.

Authentication is the credential the CLI already stores (``~/.local/share/devin/credentials.toml``),
which is a ``devin-session-token$…`` value: no API key is invented here and nothing is written
back. Override for experiments with ``HERMES_DEVIN_API_KEY`` / ``HERMES_DEVIN_API_BASE``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.acp_openai_bridge import build_openai_tool_call, completion_to_stream_chunks

from . import cascade_wire as wire

logger = logging.getLogger(__name__)

MARKER_BASE_URL = "acp://devin-api"   # provider-owned transport, like the ACP shims
SESSION_TOKEN_PREFIX = "devin-session-token$"
DEFAULT_API_BASE = "https://server.codeium.com"
DEFAULT_ROUTER_UID = "adaptive"        # the router that picks a concrete model per request
DEFAULT_MODEL = "adaptive"
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_TOKENS = 32000
DEFAULT_TEMPERATURE = 0.4
ASSIGNMENT_TTL_SECONDS = 1800.0
_IDE_NAME, _IDE_TYPE, _EXTENSION_NAME = "devin-cli", "chisel", "chisel"
_FALLBACK_IDE_VERSION = "3000.6.2"     # last version known to satisfy the backend's gate
_SYSTEM_FIELD_MAX = 60000

_version_memo: tuple[float, str] | None = None
_models_memo: tuple[float, float, list[str] | None] | None = None


def _env(*names: str) -> str:
    return next((value.strip() for name in names if (value := os.getenv(name, "")).strip()), "")


def credentials_path() -> Path:
    """The CLI's credential store (honours XDG, matching ``devin auth status``)."""
    data_home = _env("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(data_home) / "devin" / "credentials.toml"


def _read_credentials() -> dict[str, str]:
    """Parse the CLI's ``credentials.toml`` (tomllib when available, tolerant line scan otherwise)."""
    path = credentials_path()
    if not path.is_file():
        return {}
    try:
        import tomllib

        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int))}
    except Exception:  # pragma: no cover — tomllib absent or an unexpected shape
        values: dict[str, str] = {}
        for line in path.read_text(errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() and not key.strip().startswith("#"):
                values[key.strip()] = value.strip().strip('"')
        return values


def resolve_api_key() -> str:
    """The session token, scheme-prefixed as the wire format requires."""
    raw = _env("HERMES_DEVIN_API_KEY", "DEVIN_API_KEY") or _read_credentials().get("windsurf_api_key", "")
    if not raw:
        raise RuntimeError(
            "No Devin credential found. Run `devin auth login`, or set HERMES_DEVIN_API_KEY "
            f"(the CLI stores its token in {credentials_path()})."
        )
    return raw if raw.startswith(SESSION_TOKEN_PREFIX) else SESSION_TOKEN_PREFIX + raw


def resolve_api_base() -> str:
    base = _env("HERMES_DEVIN_API_BASE", "DEVIN_API_SERVER_URL") or _read_credentials().get("api_server_url", "")
    return (base or DEFAULT_API_BASE).rstrip("/")


def cli_identity_version() -> str:
    """The installed CLI's version, so the backend's client-identity gate sees a current client.

    A stale version is answered with ``failed_precondition: … please update your editor``, so this
    is read from the binary itself rather than hardcoded; memoized for 10 minutes.
    """
    global _version_memo
    now = time.monotonic()
    if _version_memo and now - _version_memo[0] < 600:
        return _version_memo[1]
    version = _env("HERMES_DEVIN_API_IDE_VERSION")
    if not version:
        command = _env("HERMES_DEVIN_API_COMMAND", "DEVIN_CLI_PATH") or "devin"
        if shutil.which(command):
            try:
                result = subprocess.run([command, "--version"], capture_output=True, text=True, timeout=10)
                tokens = (result.stdout or "").split()
                version = tokens[1] if len(tokens) > 1 and tokens[1][:1].isdigit() else ""
            except Exception:
                version = ""
    version = version or _FALLBACK_IDE_VERSION
    _version_memo = (now, version)
    return version


def _os_name() -> str:
    return {"Darwin": "darwin", "Windows": "windows"}.get(platform.system(), "linux")


def _flatten_content(content: Any) -> str:
    """One message's content as text (OpenAI content may be a str, a parts list, or a dict)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return content.get("content") if isinstance(content.get("content"), str) else json.dumps(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("type") == "image_url":
                parts.append("[image omitted: the Cascade wire takes images as ImageData, not URLs]")
        return "\n".join(parts)
    return str(content)


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Pull system/developer turns out of the transcript: the wire has a single system prompt slot."""
    system: list[str] = []
    rest: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() in ("system", "developer"):
            text = _flatten_content(message.get("content")).strip()
            if text:
                system.append(text)
        else:
            rest.append(message)
    joined = "\n\n".join(system)
    return joined[:_SYSTEM_FIELD_MAX], rest


def _prompt_message(message: dict[str, Any]) -> bytes | None:
    """One Hermes message → ``ChatMessagePrompt``; None for turns with nothing to send."""
    role = str(message.get("role") or "").strip().lower()
    text = _flatten_content(message.get("content"))
    if role == "assistant":
        calls = [
            (
                str(call.get("id") or ""),
                str((call.get("function") or {}).get("name") or ""),
                (call["function"].get("arguments") if isinstance(call["function"].get("arguments"), str)
                 else json.dumps((call.get("function") or {}).get("arguments") or {})),
            )
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        ]
        thinking = str(message.get("reasoning_content") or message.get("reasoning") or "")
        if not text and not calls and not thinking:
            return None
        return wire.chat_prompt_message(
            message_id=str(message.get("id") or uuid.uuid4()), source=wire.SOURCE_SYSTEM,
            prompt=text, tool_calls=calls, thinking=thinking)
    if role == "tool":
        call_id = str(message.get("tool_call_id") or message.get("call_id") or "")
        content_text = text or "(empty tool result)"
        return wire.chat_prompt_message(
            message_id=str(message.get("id") or uuid.uuid4()), source=wire.SOURCE_TOOL,
            prompt=content_text, tool_call_id=call_id,
            tool_result_is_error=_looks_like_error(content_text))
    if not text:
        return None
    return wire.chat_prompt_message(
        message_id=str(message.get("id") or uuid.uuid4()), source=wire.SOURCE_USER, prompt=text)


def _looks_like_error(text: str) -> bool:
    lowered = text[:400].lower()
    return lowered.startswith("error") or "\nerror" in lowered or "traceback (most recent call last)" in lowered


def _tool_definitions(tools: list[dict[str, Any]] | None) -> list[bytes]:
    """OpenAI ``tools`` → ``ChatToolDefinition`` protos, skipping malformed entries."""
    definitions: list[bytes] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = (function or {}).get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        definitions.append(wire.tool_definition_message(
            name=name.strip(),
            description=str(function.get("description") or ""),
            parameters_schema=json.dumps(function.get("parameters") or {"type": "object", "properties": {}}),
            strict=bool(function.get("strict")),
        ))
    return definitions


_CONNECT_CODE_STATUS = {
    "invalid_argument": 400, "out_of_range": 400, "failed_precondition": 412,
    "unauthenticated": 401, "permission_denied": 403, "not_found": 404,
    "already_exists": 409, "aborted": 409, "resource_exhausted": 429,
    "unimplemented": 501, "internal": 500, "unavailable": 503, "deadline_exceeded": 504,
}


def _as_httpx_timeout(value: Any, fallback: Any) -> "httpx.Timeout":
    """Coerce a call timeout into ``httpx.Timeout``.

    The OpenAI ``create()`` contract accepts float seconds OR a ready-made ``httpx.Timeout``
    (Hermes' streaming loop passes the latter, MoA forwards it to this client). Pass Timeout
    objects through untouched; wrap anything else as a total timeout, falling back when unset.
    """
    import httpx

    if isinstance(value, httpx.Timeout):
        return value
    if not value:
        value = fallback
    if isinstance(value, httpx.Timeout):
        return value
    return httpx.Timeout(float(value))


def _status_for_connect_error(text: str) -> int | None:
    """HTTP status for a Connect error code, so Hermes can judge retryability.

    Trailers arrive as ``"<code>: <message>"``; the classifier keys retryability off a status
    code and walks causes, so mapping the code here is what stops a deterministic refusal
    (stale client, bad uid, rejected key) from being retried three times.
    """
    code = (text or "").split(":", 1)[0].strip().lower()
    return _CONNECT_CODE_STATUS.get(code)


class CascadeError(RuntimeError):
    """Devin Cascade failure, carrying an HTTP status whenever one applies.

    A bare ``RuntimeError`` reaches Hermes with no status, so it is classified as a generic
    gateway error and marked ``retryable`` even when no retry can help.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DevinCascadeClient:
    """``openai.OpenAI``-compatible facade for the Cascade chat API."""

    # Already a complete client: Hermes must not re-dispatch it through a wire adapter, and it is
    # safe to call from the async paths as-is (same contract as the ACP shims).
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None,
        default_headers: dict[str, str] | None = None, timeout: float | None = None, **_: Any,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or MARKER_BASE_URL).rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._timeout = _as_httpx_timeout(timeout, _env("HERMES_DEVIN_API_TIMEOUT") or DEFAULT_TIMEOUT_SECONDS)
        self._cascade_id = _env("HERMES_DEVIN_API_CASCADE") or str(uuid.uuid4())
        self._session_id = str(uuid.uuid4())
        self._assignments: dict[str, tuple[float, str, str]] = {}
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))

    # ── OpenAI surface ────────────────────────────────────────────────────────────────────────
    def _create_chat_completion(
        self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
        stream: bool = False, max_tokens: int | None = None, temperature: float | None = None,
        top_p: float | None = None, timeout: float | None = None, **_: Any,
    ) -> Any:
        model_id = str(model or "").strip()
        if not model_id or model_id in ("devin-api", "cascade", "cascade-router"):
            model_id = _env("HERMES_DEVIN_API_MODEL") or DEFAULT_MODEL

        system_prompt, transcript = _split_system(messages or [])
        prompts = [prompt for prompt in (_prompt_message(m) for m in transcript) if prompt]
        if not prompts:
            prompts = [wire.chat_prompt_message(message_id=str(uuid.uuid4()), source=wire.SOURCE_USER, prompt="Continue.")]

        chat_model_uid, assignment_jwt = self._resolve_model(model_id)
        request = wire.get_chat_message_request(
            metadata=self._metadata(),
            system_prompt=system_prompt,
            prompts=prompts,
            chat_model_uid=chat_model_uid,
            cascade_id=self._cascade_id,
            execution_id=str(uuid.uuid4()),
            tools=_tool_definitions(tools),
            tool_choice=wire.tool_choice_message(tool_choice) if tool_choice is not None else None,
            configuration=wire.completion_configuration_message(
                max_tokens=int(max_tokens or DEFAULT_MAX_TOKENS),
                temperature=float(temperature if temperature is not None else DEFAULT_TEMPERATURE),
                top_p=float(top_p or 1.0)),
            model_assignment_jwt=assignment_jwt,
        )
        stream_result = self._chat(request, timeout=timeout)
        completion = self._to_completion(stream_result, requested_model=model_id)
        return completion_to_stream_chunks(completion) if stream else completion

    def list_models(self, *, timeout_seconds: float = 20.0) -> list[str]:
        """Model ids the account can reach, from ``devin models list`` (memoized; [] when absent)."""
        global _models_memo
        now = time.monotonic()
        if _models_memo and now - _models_memo[0] < (_models_memo[1]):
            return list(_models_memo[2] or [])
        ids: list[str] | None = None
        command = _env("HERMES_DEVIN_API_COMMAND", "DEVIN_CLI_PATH") or "devin"
        if shutil.which(command):
            try:
                result = subprocess.run([command, "models", "list"], capture_output=True, text=True,
                                        timeout=timeout_seconds)
                ids = _parse_models_output(result.stdout or "")
            except Exception:
                ids = None
        _models_memo = (now, 300.0 if ids else 30.0, ids)
        return list(ids or [])

    def close(self) -> None:
        self.is_closed = True

    # ── Cascade transport ─────────────────────────────────────────────────────────────────────
    def _metadata(self) -> bytes:
        version = cli_identity_version()
        return wire.metadata_message(
            resolve_api_key(), ide_name=_IDE_NAME, ide_version=version, extension_name=_EXTENSION_NAME,
            os_name=_os_name(), ide_type=_IDE_TYPE, session_id=self._session_id)

    def _resolve_model(self, model_id: str) -> tuple[str, str]:
        """``(chat_model_uid, assignment_jwt)`` — router uids are resolved through AssignModel."""
        if model_id.startswith("MODEL_") or _env("HERMES_DEVIN_API_SKIP_ASSIGN") == "1":
            return model_id, ""
        cached = self._assignments.get(model_id)
        if cached and time.monotonic() - cached[0] < ASSIGNMENT_TTL_SECONDS:
            return cached[1], cached[2]
        try:
            body = wire.assign_model_request(
                self._metadata(), model_router_uid=model_id, cascade_id=self._cascade_id,
                prompt=wire.chat_prompt_message(
                    message_id=str(uuid.uuid4()), source=wire.SOURCE_USER, prompt="Assign a model for this session."))
            payload = self._post(wire.ASSIGN_MODEL_PATH, body, wire.PROTO_CONTENT_TYPE, timeout=60.0)
            assignment_jwt, model_uid, harness = wire.parse_assignment(payload)
        except RuntimeError as exc:
            # A concrete uid the router does not know is still usable directly; only routers must
            # be assigned. Log at debug so a wrong uid shows up as a server-side error instead.
            logger.debug("Devin AssignModel failed for %r (%s); sending it as a chat uid.", model_id, exc)
            return model_id, ""
        if not assignment_jwt or not model_uid:
            logger.debug("Devin AssignModel returned no assignment for %r; sending it as a chat uid.", model_id)
            return model_id, ""
        logger.debug("Devin router %r assigned %s (harness=%s)", model_id, model_uid, harness)
        self._assignments[model_id] = (time.monotonic(), model_uid, assignment_jwt)
        return model_uid, assignment_jwt

    def _chat(self, request: bytes, *, timeout: float | None = None) -> wire.ChatStream:
        payload = self._post(wire.GET_CHAT_MESSAGE_PATH, request, wire.CONNECT_CONTENT_TYPE,
                             timeout=timeout, compress=True)
        stream = wire.decode_chat_stream(payload)
        if stream.error:
            raise CascadeError(f"Devin Cascade error: {stream.error}",
                               status_code=_status_for_connect_error(stream.error))
        if not stream.text and not stream.tool_calls and stream.messages == 0:
            raise RuntimeError("Devin Cascade returned an empty stream.")
        return stream

    def _post(self, path: str, body: bytes, content_type: str, *, timeout: float | None,
              compress: bool = False) -> bytes:
        import httpx

        headers = {
            "content-type": content_type,
            "connect-protocol-version": wire.CONNECT_PROTOCOL_VERSION,
            "user-agent": wire.CONNECT_USER_AGENT,
            "accept-encoding": "identity",
            **self._default_headers,
        }
        payload = body
        if compress:
            headers["connect-content-encoding"] = "gzip"
            headers["connect-accept-encoding"] = "gzip"
            payload = wire.encode_envelope(body)
        url = f"{resolve_api_base()}{path}"
        try:
            with httpx.Client(timeout=_as_httpx_timeout(timeout, self._timeout)) as client:
                response = client.post(url, content=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _short(exc.response.text)
            raise CascadeError(f"Devin Cascade HTTP {exc.response.status_code} on {path}: {detail}",
                               status_code=exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise CascadeError(f"Devin Cascade request to {url} failed: {exc}") from exc
        return response.content

    def _to_completion(self, result: wire.ChatStream, *, requested_model: str) -> SimpleNamespace:
        """Decoded stream → the OpenAI completion shape Hermes' loop consumes."""
        tool_calls = [
            build_openai_tool_call(call_id=call["id"] or f"cascade_call_{index}", name=call["name"],
                                   arguments=call["arguments"] or "{}")
            for index, call in enumerate(result.tool_calls, start=1)
            if call.get("name")
        ]
        if result.finished_with_tool_calls and tool_calls:
            finish_reason = "tool_calls"
        elif result.stop_reason == wire.STOP_REASON_MAX_TOKENS:
            finish_reason = "length"
        else:
            finish_reason = "stop"
        usage = result.usage
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        message = SimpleNamespace(
            content=result.text or None, tool_calls=tool_calls or None,
            reasoning=result.thinking or None, reasoning_content=result.thinking or None,
            reasoning_details=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=int(usage.get("cache_read_tokens") or 0)),
            ),
            model=result.actual_model_uid or usage.get("model_uid") or requested_model,
        )


def _short(text: str, limit: int = 400) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def _parse_models_output(output: str) -> list[str]:
    """``devin models list`` prints indented ``<uid>   <label>`` rows; take the uids."""
    ids: list[str] = []
    for line in output.splitlines():
        if not line.startswith("  ") or line.startswith("   "):
            continue
        token = line.strip().split()[0] if line.strip() else ""
        if token and token != "aliases:" and token[0].isalnum() and token not in ids:
            ids.append(token)
    return ids
