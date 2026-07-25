# Knowledge graph via You.com Research with a structured output schema

The Cartographer builds the research knowledge graph by calling the **You.com
Research API with a structured `output_schema`** (JSON-Schema subset) in a single
agentic, multi-step web-research call. The API returns a typed graph
(`summary`, `nodes`, `edges`) *and* cited `sources`; we normalize it and cache it
to `wiki/graph/<slug>.json`. The graph renders in the UI with `reactflow` and a
lightweight force-directed layout.

## Considered options

- **Search + our own LLM extraction**: run `you-search`, then prompt the shim LLM
  to extract entities/relations. More moving parts, weaker grounding, and we own
  the extraction quality. Rejected as the primary path.
- **Client-side graphing of raw search hits**: no relations, no synthesis — just
  a list with edges we invent. Not a real knowledge graph.
- **Research (plain text) + regex/LLM parse**: loses the structure guarantee the
  `output_schema` gives us for free.

## Why this way

- **One grounded call** produces both the structure and the citations, so the
  graph is defensible (every paper node can point at a source).
- `output_schema` is supported on `standard`/`deep`/`exhaustive` (not `lite`), so
  we default to `standard` (~30–90s) and allow deeper efforts on demand.
- Caching gives an **instant demo payoff** with a live-refresh option, matching
  the pre-baked philosophy of [ADR-0002](0002-prebaked-reproduction-of-lcr-paper.md).

## Consequences

- We depend on the model filling the schema well; we normalize defensively
  (coerce node types, dedupe edges, synthesize placeholder nodes for dangling
  endpoints, fuzzy-match source URLs onto paper nodes).
- Research latency is user-visible; the UI shows a clear progress state and a
  cached fallback.
- `api.you.com` requires a browser `User-Agent` (Cloudflare error 1010 otherwise)
  — handled in `youcom_service._request`.
- A live **You.com spend meter** is derived from published per-call pricing, not
  billed usage, so it is approximate.
