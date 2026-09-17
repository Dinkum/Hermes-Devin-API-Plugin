# Hermes Devin API Plugin

Devin (Cognition) model ids inside Hermes as a first-class provider — Hermes runs the tools, the plugin talks HTTPS+protobuf straight to the Codeium Cascade chat API.

## Prereqs

- Hermes Agent **v0.21.3+** (the plugin imports `agent.acp_openai_bridge` at load time)
- Devin CLI installed and signed in:
  ```bash
  devin auth login
  ```
  This writes the session token to `~/.local/share/devin/credentials.toml`. The plugin reads it from there — there is no separate API key — and it needs the `devin` binary on `PATH` for its version string and model list. A Devin subscription is required; usage bills to your own account.
- `httpx` (already shipped with Hermes). No protobuf runtime needed.

## Usage

```bash
mkdir -p ~/.hermes/plugins/model-providers/devin-api
cp plugin.yaml __init__.py cascade_client.py cascade_wire.py \
   ~/.hermes/plugins/model-providers/devin-api/
```

Then:

```bash
hermes --provider devin-api -m adaptive -z "hello"   # one-shot
hermes                                                # then /model → "Devin (Cascade API)"
```

`adaptive` is the router and the safe default. Aliases: `cascade`, `devin-cascade`, `devin-http`,
`cognition-cascade`.

Optionally pin the context window (the transport has no `/models` probe, so Hermes otherwise falls
back to 256K and logs `Could not determine context length` every turn):

```bash
hermes config set model_overrides.devin-api.swe-2-high.context_window 262144
```

Not to be confused with the sibling ACP plugin (`Devin (ACP agent)`), which spawns `devin acp` and
lets Devin run its own tools.
