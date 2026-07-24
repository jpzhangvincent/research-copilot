# Librarian uses a lightweight compile endpoint, not the full llm-wiki skill

The Librarian compiles a paper into a Wiki article via a backend endpoint (you-contents fetches the arXiv page, OpenAI writes an article in llm-wiki format, saved to `wiki/<topic>/<slug>.md` with index/log updates) rather than spawning an agent that runs the full karpathy-llm-wiki SKILL.md workflow (triage, cascade updates, evidence grounding).

## Why

- The llm-wiki skill is interactive and agent-driven; its variability (triage decisions, cascade edits) is a liability in a scripted live demo.
- The demo compiles a single new article, so the skill's compounding/cascade machinery adds risk without demo value.
- The output files remain llm-wiki- and Obsidian-compatible, so we keep the artifact story without the runtime unpredictability.

## Consequences

- We are not exercising the skill's grounding invariant, triage, or cascade logic — the wiki won't be "correct by the skill's rules", just structurally compatible.
- Post-hackathon, swapping in the real skill agent is possible without changing the file layout.
