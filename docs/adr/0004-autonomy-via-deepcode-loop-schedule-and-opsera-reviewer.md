# Autonomy via DeepCode loop/schedule + an Opsera Reviewer

The Copilot's autonomy features reuse DeepCode's existing `core/loop` (autonomous loop-until-goal) and `core/schedule` (scheduler) rather than building new orchestration, and add a Reviewer step backed by the Opsera DevSecOps agent. All are demo-scoped.

- **Goal Loop**: the Engineer re-runs paper2code until NDCG@5 is within tolerance of the paper's reported figure (the Reproduction Goal), capped at N iterations.
- **Scheduled Scout**: Scout on an interval surfaces new papers; for the demo it's a visible interval run with pre-baked results.
- **Reviewer**: one on-demand Opsera scan + architecture analysis over the reproduced repo, shown as a step.
- **Git**: the repo is version-controlled; the Engineer's generated code and our own changes are committed as we go.

## Why

- DeepCode already implements loop and schedule primitives; reusing them is faster and less risky than a bespoke scheduler/loop.
- A metric-defined Reproduction Goal makes "implement until it works" objective and demoable.
- Opsera provides security + architecture analysis without us hand-rolling scanners.

## Consequences

- The Goal Loop needs a hard iteration cap and a cached known-good result (per ADR-0002) so a live loop can't hang the demo.
- Scheduled Scout is not a real background daemon for the demo; the long-term vision may promote it to one.
- Reviewer output depends on the Opsera agent's availability at demo time; have a cached report as fallback.
