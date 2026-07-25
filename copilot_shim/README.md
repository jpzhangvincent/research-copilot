# Copilot Shim — subscription-backed OpenAI provider

`openai_shim.py` is a local `POST /v1/chat/completions` server (non-streaming + SSE,
with OpenAI tool-calling) that drives **Claude Code (`claude -p`)** or **Codex
(`codex exec`)** as a *single-turn responder*. It lets DeepCode's reproduction
pipeline and the Librarian wiki-compile run on your **subscription with zero API
billing**.

## How it works

- OpenAI `tools` + the app system prompt are passed on the CLI's **system channel**
  (`--append-system-prompt-file`) so Claude doesn't flag them as prompt injection.
- The conversation transcript is passed on stdin as trusted developer data.
- The CLI is constrained to a single turn with its own tools disabled
  (`--allowedTools "" --permission-mode plan --strict-mcp-config --max-turns 1`),
  running in a clean sandbox cwd so no project `.mcp.json` is auto-loaded.
- A tool call comes back as `{"tool_calls":[{"name":...,"arguments":{...}}]}`, which
  the shim maps to OpenAI `choices[0].message.tool_calls` with
  `finish_reason: "tool_calls"`.

## Run

```bash
python3 copilot_shim/openai_shim.py                 # claude backend, port 8899
COPILOT_SHIM_BACKEND=codex python3 copilot_shim/openai_shim.py
COPILOT_SHIM_DEBUG=1 python3 copilot_shim/openai_shim.py   # verbose
```

Env: `COPILOT_SHIM_PORT` (8899), `COPILOT_SHIM_BACKEND` (claude|codex),
`COPILOT_SHIM_TIMEOUT` (240s), `COPILOT_SHIM_DEBUG`.

## DeepCode config (deepcode_config.json — gitignored by DeepCode)

Point the `vllm` provider at the shim and force it:

```json
"agents": {
  "defaults":       { "provider": "vllm", "model": "vllm/claude", ... },
  "planning":       { "model": "vllm/claude" },
  "implementation": { "model": "vllm/claude" }
},
"providers": {
  "vllm": { "apiBase": "http://localhost:8899/v1", "apiKey": "shim" }
}
```

The Librarian defaults to the same shim (`COPILOT_WIKI_BASE_URL`,
`COPILOT_WIKI_MODEL=vllm/claude`); override those env vars to use a real provider.

## Caveats

- Latency is a few seconds to ~15s per call — fine for building and **pre-baking**
  the demo reproduction, slow for very long unattended tool loops.
- Claude Code is a safety-trained assistant, not a raw model; occasional refusals or
  off-protocol replies are possible. The shim tolerates malformed JSON best-effort.
