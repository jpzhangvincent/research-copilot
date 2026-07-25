# Reproductions

Paper reproductions produced by the AI Research Copilot's paper2code pipeline
(DeepCode engine + the OpenAI-compatible shim), then completed and verified into
self-contained, runnable projects.

| Folder | Paper | Metric | Run |
|--------|-------|--------|-----|
| [`lcr/`](lcr/) | LLM-Confidence Reranker (training-free reranking for RAG) | NDCG@5 | `cd lcr && pip install numpy rank_bm25 && python run.py --smoke` |

Each reproduction is generated into DeepCode's runtime lab
(`DeepCode/deepcode_lab/tasks/<id>/…`, gitignored) and the finished project is
curated here as tracked source. See each folder's own `README.md` for details.
