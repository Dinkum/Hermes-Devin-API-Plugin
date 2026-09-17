# Hermes Devin API Plugin

Run Devin (Cognition) models in Hermes over the Codeium Cascade chat API — Hermes runs the tools.

## Prereqs

- Hermes Agent (needs `agent/acp_openai_bridge.py`, added 2026-08-17 — newer than tag `v2026.8.13`)
- Devin CLI installed and logged in: `devin auth login`

## Usage

```bash
mkdir -p ~/.hermes/plugins/model-providers/devin-api
cp plugin.yaml __init__.py cascade_client.py cascade_wire.py ~/.hermes/plugins/model-providers/devin-api/

hermes --provider devin-api -m adaptive     # or /model → "Devin (Cascade API)"
```

Aliases: `cascade`, `devin-cascade`, `devin-http`. This is not the ACP plugin
(`Devin (ACP agent)`), which spawns `devin acp` and lets Devin run its own tools.
