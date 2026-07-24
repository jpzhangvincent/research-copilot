# Extend DeepCode's web app as the architecture spine

The Research Copilot is built by extending DeepCode's existing React+Vite frontend and FastAPI backend rather than building a fresh app. DeepCode already ships the hero surface — a paper2code pipeline that streams progress to the UI over WebSockets — so reusing it is the fastest path to a polished Reproduction demo. Discovery (You.com MCP) and the Wiki (karpathy-llm-wiki) are added as new routes/pages inside the same app.

## Considered Options

- **Fresh thin UI (Next.js) + DeepCode backend as engine**: more UX control, but re-wiring reproduction streaming and running two stacks costs time we don't have for a hackathon.
- **Fresh full-stack app, DeepCode as subprocess/library**: maximum control, maximum work; throws away the already-built streaming reproduction UI.

## Consequences

- We inherit DeepCode's frontend conventions (React+Vite, lucide-react) and backend patterns (FastAPI routes + WebSockets). New features must fit these.
- Discovery and Wiki UIs must be built to match DeepCode's existing look/feel for a coherent demo.
- Long-term "vision" features (teach/visualize, enterprise assessment) will also live in this app.
