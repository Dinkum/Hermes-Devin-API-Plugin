# Hermes Devin API Plugin

Run Devin (Cognition) models in Hermes over the Cognition subscription API. Native Hermes tool calling.

## Prereqs

- Hermes Agent (needs `agent/acp_openai_bridge.py`, added 2026-08-17 — newer than tag `v2026.8.13`)
- Devin CLI installed and logged in: `devin auth login`

## Usage

1. Clone this repo

   ```bash
   git clone https://github.com/Dinkum/Hermes-Devin-API-Plugin
   ```

2. Copy the plugin into Hermes

   ```bash
   mkdir -p ~/.hermes/plugins/model-providers/devin-api
   cp Hermes-Devin-API-Plugin/{plugin.yaml,__init__.py,cascade_client.py,cascade_wire.py} \
      ~/.hermes/plugins/model-providers/devin-api/
   ```

3. Restart Hermes — plugins are discovered once at startup, so a running session won't see it
   (if you use the messaging gateway: `hermes gateway restart`)

4. Use it

   ```bash
   hermes --provider devin-api -m adaptive     # or /model → "Devin (Cascade API)"
   ```
