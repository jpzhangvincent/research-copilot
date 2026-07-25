# AI Research Copilot

A personal AI research partner that discovers relevant papers, maps a field as a knowledge graph, summarizes papers into a personal wiki, and reproduces chosen papers as running code. Built for the You.com Agentic Hackathon; the demo hero is turning a discovered paper into a working reproduction.

## Language

**AI Research Copilot** (a.k.a. Research Copilot):
The whole system — the personal AI research partner. The product name for everything in this repo.
_Avoid_: "the app", "the assistant", "DeepCode" (that names the vendored engine, not our product)

**Knowledge Graph**:
An interactive, typed map of a research area — nodes (papers, methods, datasets, metrics, concepts) and labeled relation edges — produced by the Cartographer from one You.com research call and rendered with reactflow.
_Avoid_: "mind map", "diagram", "network" (ambiguous)

**Landscape Brief**:
The cited prose synthesis returned alongside the Knowledge Graph, summarizing the field. Shown next to the graph.
_Avoid_: "summary" (ambiguous with Wiki article)

**output_schema**:
The JSON-Schema (subset) we hand to the You.com Research API so it returns the Knowledge Graph as structured JSON instead of prose. Supported on standard/deep/exhaustive, not lite.
_Avoid_: "response format", "function schema"

**Reproduction**:
The act of turning a chosen paper into a running codebase on a real dataset. This is the demo climax. Powered by DeepCode's paper2code pipeline.
_Avoid_: "paper2code" as a noun for the output (that names the engine, not the result), "implementation"

**paper2code**:
The DeepCode engine/pipeline (`workflows/agent_orchestration_engine.py`) that performs a Reproduction. Refers to the tool, not the output.
_Avoid_: using it to mean the generated code

**Discovery**:
Finding relevant, recent papers via You.com search, based on the user's stated interests. The opening beat of the demo arc.
_Avoid_: "subscribe", "feed" (those imply background scheduling we are not building for the hackathon)

**Wiki**:
The user's personal knowledge base of compiled article markdown, maintained by the karpathy-llm-wiki skill (`raw/` + `wiki/`). Lives in a folder that can double as an Obsidian vault.
_Avoid_: "notes", "Obsidian" (Obsidian is just a viewer over the markdown)

**Clarification**:
The guided back-and-forth where the Copilot pins down scope with the user before a Reproduction (which parts of the paper, which dataset, success criteria).
_Avoid_: "requirements gathering", "intake"

**Agents API**:
OUR internal multi-agent orchestration layer (DeepCode agent kernel + You.com orchestration recipes). NOT an external You.com product — no such endpoint exists.
_Avoid_: implying a third-party "Agents API" service

**Search tools**:
The You.com `you-search` and `you-research` capabilities (web/news search and cited synthesis), used for Discovery and paper understanding.
_Avoid_: "the search API" (ambiguous across you-search vs you-research)

**Compute tools**:
The code-execution + filesystem sandbox that runs generated code during a Reproduction. Provided by DeepCode's native/MCP tool servers.
_Avoid_: "the runtime", "the executor"

## Agent Team

The Research Copilot is presented to the user as a team of named agents. These are thin roles over existing pieces, not new orchestration frameworks.

**Scout**:
The Discovery agent. Uses You.com Search tools to surface relevant recent papers from the user's interests.
_Avoid_: "searcher", "crawler"

**Cartographer**:
The mapping agent. Runs one You.com research call with an output_schema to build the Knowledge Graph and Landscape Brief for an area. Lives in `graph_service.py`.
_Avoid_: "grapher", "mapper"

**Librarian**:
The wiki agent. Compiles a discovered paper into a Wiki article (llm-wiki format), rendered in-app.
_Avoid_: "summarizer", "note-taker"

**Architect**:
The planning half of Reproduction. Owns Clarification and the paper2code plan (DeepCode's planning agent + plan-review gate).
_Avoid_: "planner"

**Engineer**:
The implementation half of Reproduction. Generates and runs the code (DeepCode's implementation workflow + Compute tools).
_Avoid_: "coder", "builder"

**Reviewer**:
The audit agent. Runs an Opsera DevSecOps scan + architecture analysis over the reproduced repo and reports findings. Demo-scoped: one on-demand run, shown as a step.
_Avoid_: "linter", "scanner"

## Autonomy

**Goal Loop**:
The Engineer re-running paper2code until the Reproduction's NDCG@5 lands within tolerance of the paper's reported number, capped at a max iteration count. Built on DeepCode's `core/loop`.
_Avoid_: "retry loop", "auto-fix"

**Reproduction Goal**:
The success criterion the Goal Loop targets — NDCG@5 within tolerance of the paper's figure. The measurable definition of "done" for a Reproduction.
_Avoid_: "target", "acceptance"

**Scheduled Scout**:
Scout running on an interval to surface newly published papers in the user's interests. Demo-scoped: a visible interval run with pre-baked results; built on DeepCode's `core/schedule`.
_Avoid_: "cron", "subscription", "feed"
