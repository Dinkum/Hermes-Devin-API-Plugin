"""Devin (Cognition) provider profile over the Codeium Cascade chat API.

Unlike the sibling ``devin`` plugin — which drives ``devin acp`` over stdio — this profile speaks
the CLI's *own* inference protocol directly: Connect+protobuf to
``{api_server}/exa.api_server_pb.ApiServerService/GetChatMessage``. That endpoint takes
OpenAI-style function definitions (``tools``) and streams real tool calls back, so Hermes' normal
agent loop owns the tools: no ACP, no prompt-side ``<tool_call>`` text bridge, and every Hermes
tool is available to the model as a first-class function.

Auth is the credential the CLI already stores (``~/.local/share/devin/credentials.toml``); the
profile declares no env vars and no key of its own, so the subscription — not a borrowed API key —
stays the billing path. ``process_command="devin"`` is declared so Hermes' external-process
credential resolution has a real precondition to check (the binary whose store holds the token).
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class DevinCascadeProfile(ProviderProfile):
    """Devin via Cascade — HTTP+protobuf, native tool calling, provider-owned transport."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the Cascade client instead of an ``openai.OpenAI`` HTTP client."""
        from .cascade_client import DevinCascadeClient

        return DevinCascadeClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 20.0
    ) -> list[str] | None:
        """Model uids the signed-in account can reach, from ``devin models list``.

        The catalog is gated on plan and team policy, so it is read from the CLI rather than
        hardcoded. ``api_key``/``base_url`` are ignored: the CLI owns auth. None when the CLI is
        missing or the probe fails, per the base contract, so callers fall back to
        ``fallback_models``.

        Also retries the picker-catalogue registration: that table lives in
        ``hermes_cli.models``, which is often still mid-import when plugin discovery runs (this
        module is imported *from* the discovery scan), so the first attempt can silently miss and
        leave the picker row empty. A profile probe always happens after the module is importable,
        which makes this the reliable second chance.
        """
        _install_catalog_fetcher()
        from .cascade_client import DevinCascadeClient

        try:
            models = DevinCascadeClient().list_models(timeout_seconds=timeout)
        except Exception:
            return None
        return models or None


devin_api = DevinCascadeProfile(
    name="devin-api",
    aliases=("cascade", "devin-cascade", "devin-http", "cognition-cascade"),
    display_name="Devin (Cascade API)",
    description="Devin/Cognition Cascade chat API — native tool calling over HTTP, Hermes runs the tools",
    signup_url="https://app.devin.ai/",
    api_mode="chat_completions",  # the client is chat-shaped and skips transport wrapping
    env_vars=(),  # token comes from the CLI's own credential store
    base_url="acp://devin-api",  # provider-owned transport marker (keeps Hermes' HTTP paths off it)
    auth_type="external_process",
    supports_health_check=False,  # there is no /models endpoint to probe
    # The binary is not launched, but its presence is the real precondition: it holds the token.
    process_command="devin",
    process_args=(),
    process_command_env_vars=("HERMES_DEVIN_API_COMMAND", "DEVIN_CLI_PATH"),
    process_args_env_var="HERMES_DEVIN_API_ARGS",
    # Shown in the /model picker when the live probe fails. "adaptive" first: it is the router the
    # CLI itself asks for, and it resolves to whatever model the account is entitled to.
    fallback_models=(
        "adaptive", "swe-1-7-medium", "swe-2-high", "swe-2-medium", "swe-2-max",
        "claude-opus-5-high", "claude-sonnet-5-high", "gpt-5-6-sol-high",
    ),
    default_max_tokens=32000,
)

register_provider(devin_api)


# ── Picker catalogue ─────────────────────────────────────────────────────────────────────────
# Mirrors the sibling ACP plugin: `hermes_cli.models` only live-fetches for `api_key` profiles, so
# an out-of-tree external-process provider has to add its own fetcher row, canonical entry and
# overlay or it never appears in any picker (even though `--provider devin-api` works). Everything
# is mutated in place from the plugin, so nothing in the hermes-agent checkout changes.

_MEMO_TTL, _MEMO_FAIL_TTL = 300.0, 30.0
_catalog_memo: tuple[float, float, list[str] | None] | None = None
_SLUG = "devin-api"
_TUI_DESC = "Devin (Cascade API) (Native tool calling over HTTP; Hermes runs the tools)"


def _devin_api_catalog(normalized: str, force_refresh: bool) -> list[str] | None:
    """Model ids for the ``/model`` picker and for model-switch validation (memoized)."""
    global _catalog_memo
    import time

    now = time.monotonic()
    if not force_refresh and _catalog_memo is not None and now - _catalog_memo[0] < _catalog_memo[1]:
        live = _catalog_memo[2]
    else:
        live = devin_api.fetch_models() or None
        _catalog_memo = (now, _MEMO_TTL if live else _MEMO_FAIL_TTL, live)
    return live or list(devin_api.fallback_models)


def _install_catalog_fetcher() -> None:
    """Best-effort: a core refactor of the table must degrade the picker, never break registration."""
    try:
        from hermes_cli import models as _models

        _models._PROVIDER_CATALOG_FETCHERS.setdefault(_SLUG, _devin_api_catalog)
    except Exception:  # pragma: no cover — import cycle during discovery, or the table moved
        import logging

        logging.getLogger(__name__).debug("devin-api: could not register a picker catalog fetcher", exc_info=True)


def _install_canonical_provider() -> None:
    """List Devin API as a selectable provider in the CLI/setup/desktop pickers."""
    try:
        from hermes_cli import models_catalog_static as _static

        if any(entry.slug == _SLUG for entry in _static.CANONICAL_PROVIDERS):
            return
        entry = _static.ProviderEntry(_SLUG, "Devin (Cascade API)", _TUI_DESC)
        _static.CANONICAL_PROVIDERS.append(entry)
        _static._PROVIDER_LABELS.setdefault(entry.slug, entry.label)
    except Exception:  # pragma: no cover — the table moved or is not importable yet
        import logging

        logging.getLogger(__name__).debug("devin-api: could not register the provider row", exc_info=True)


def _install_overlay() -> None:
    """Give the provider a ``HERMES_OVERLAYS`` row so the pickers build a row for it."""
    try:
        from hermes_cli.providers import _LABEL_OVERRIDES, HERMES_OVERLAYS, HermesOverlay

        _LABEL_OVERRIDES.setdefault(_SLUG, "Devin (Cascade API)")
        HERMES_OVERLAYS.setdefault(_SLUG, HermesOverlay(
            transport="openai_chat",  # the client is chat-shaped and skips transport wrapping
            auth_type="external_process",
            base_url_override="acp://devin-api",
            base_url_env_var="DEVIN_API_BASE_URL",
        ))
    except Exception:  # pragma: no cover — the overlay table moved or is mid-import
        import logging

        logging.getLogger(__name__).debug("devin-api: could not register the provider overlay", exc_info=True)


def _install_signed_in_probe() -> None:
    """Let the pickers see that the Devin credential store exists (the CLI's own login)."""
    try:
        from hermes_cli import inventory as _inventory

        original = _inventory._external_process_signed_in
        if getattr(original, "_devin_api_wrapped", False):
            return

        def _with_devin_api(slug: str) -> bool:
            if slug != _SLUG:
                return original(slug)
            try:
                from .cascade_client import credentials_path

                path = credentials_path()
                return path.is_file() and path.stat().st_size > 2
            except Exception:
                return False

        _with_devin_api._devin_api_wrapped = True
        _inventory._external_process_signed_in = _with_devin_api
    except Exception:  # pragma: no cover — inventory moved or is mid-import
        import logging

        logging.getLogger(__name__).debug("devin-api: could not install the signed-in probe", exc_info=True)


_install_catalog_fetcher()
_install_canonical_provider()
_install_overlay()
_install_signed_in_probe()
