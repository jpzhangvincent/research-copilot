"""LCR reproduction entry point (paper §4).

Two modes:

  python run.py --smoke
      Runs the FULL pipeline (BM25 retrieve -> reranker -> LCR_SORT -> NDCG@5)
      end-to-end on a tiny built-in toy corpus with a deterministic MockLLM, so
      it needs no GPU, no model download, and (with rank_bm25) no dataset. Proves
      the wiring is correct.

  python run.py --config configs/default.yaml
      Runs the real experiment: loads a BEIR dataset (e.g. NFCorpus) and a real
      confidence LLM backend (vLLM / HF transformers). Requires those deps + a GPU
      for the paper's numbers.
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, List, Optional

from confidence.sampling import LLMClient, SamplingConfig
from confidence.nli_cluster import NLIConfig
from lcr.binning import LCRThresholds
from retrievers.bm25_retriever import CorpusDoc, simple_tokenize
from eval.run_experiment import ExperimentConfig, run_experiment


# --------------------------------------------------------------------------- #
# Deterministic mock LLM (smoke mode only)
# --------------------------------------------------------------------------- #
class MockLLMClient:
    """A dependency-free LLMClient that fakes MSCP behaviour deterministically.

    - Sampling (query-only): returns a spread of answers => low C(q) => LCR engages.
    - Sampling (query+doc):  answer diversity is inversely tied to query/doc lexical
      overlap, so relevant docs get high MSCP (=> binned high) and irrelevant docs
      low MSCP (=> binned low). This makes LCR lift relevant docs without a real LLM.
    - Entailment: 'entailment' iff the two answers are identical strings.
    NOTE: this is a stand-in for a real 7-9B confidence model; it demonstrates the
    mechanism, not the paper's absolute NDCG gains.
    """

    _Q_ONLY = re.compile(r"Question:\s*(.*)")
    _CTX = re.compile(r"Context:\s*(.*)")
    _A1 = re.compile(r"Possible Answer 1:\s*(.*)")
    _A2 = re.compile(r"Possible Answer 2:\s*(.*)")

    def generate(self, prompts, *, n=1, temperature=1.0, max_tokens=64, stop=None):
        return [self._one(p, n) for p in prompts]

    def _overlap(self, query: str, doc: str) -> float:
        q = set(simple_tokenize(query))
        d = set(simple_tokenize(doc))
        if not q:
            return 0.0
        return len(q & d) / len(q)

    def _one(self, prompt: str, n: int) -> List[str]:
        if "semantically entail" in prompt:
            a1 = self._A1.search(prompt)
            a2 = self._A2.search(prompt)
            same = a1 and a2 and a1.group(1).strip() == a2.group(1).strip()
            return ["entailment" if same else "neutral"] * n
        if "Question:" in prompt and "Answer:" in prompt:
            ctx = self._CTX.search(prompt)
            q = self._Q_ONLY.search(prompt)
            query = q.group(1).strip() if q else ""
            if ctx is not None:  # query + document sampling => confidence C(q,d)
                overlap = self._overlap(query, ctx.group(1))
                distinct = max(1, round((1 - overlap) * (n - 1))) if n > 1 else 1
            else:  # query-only sampling => C(q); keep it below t_query so LCR engages
                distinct = max(2, round(n * 0.4))
            return [f"answer_{i % distinct}" for i in range(n)]
        return ["yes"] * n  # fallback (e.g. yesno/rankgpt rerankers)


# --------------------------------------------------------------------------- #
# Toy dataset (smoke mode only) — relevant docs cover all query terms,
# distractors share only some, fillers share none.
# --------------------------------------------------------------------------- #
def _toy_data():
    corpus = [
        # q1 relevant (full query-term coverage)
        CorpusDoc("d1", "Insulin resistance and diabetes: what causes it explained."),
        CorpusDoc("d7", "What causes diabetes to worsen: diet and insulin management."),
        # q1 distractor: very high term-frequency on 'what/causes' but NO 'diabetes'
        # -> BM25 over-ranks it; LCR (missing a query term => lower confidence) fixes it.
        CorpusDoc("d2", "What causes what? Causes, causes, causes of headaches and what causes what."),
        # q2 relevant
        CorpusDoc("d3", "Green tea benefits: antioxidants and the health benefits of green tea."),
        # q3 relevant
        CorpusDoc("d5", "How do vaccines work to train the immune system against viruses."),
        # fillers
        CorpusDoc("d4", "The benefits of a morning walk and a light exercise routine."),
        CorpusDoc("d6", "A short history of tea cultivation across different regions."),
        CorpusDoc("d8", "Sunny weather forecast for the weekend with clear skies."),
    ]
    queries = {
        "q1": "what causes diabetes",
        "q2": "benefits of green tea",
        "q3": "how do vaccines work",
    }
    qrels = {
        "q1": {"d1": 2, "d7": 2},
        "q2": {"d3": 2},
        "q3": {"d5": 2},
    }
    return corpus, queries, qrels


def run_smoke() -> Dict[str, object]:
    corpus, queries, qrels = _toy_data()
    config = ExperimentConfig(
        dataset="toy", reranker="retriever_only", top_k=6, ndcg_k=5,
        thresholds=LCRThresholds(t_query=0.5, t_upper=0.7, t_lower=0.3),
        sampling=SamplingConfig(K=10),
        nli=NLIConfig(),
    )
    try:
        from retrievers.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(corpus)
    except ImportError:
        retriever = _OverlapRetriever(corpus)  # last-resort, zero-dep fallback
    return run_experiment(config, MockLLMClient(), corpus, queries, qrels, retriever=retriever)


class _OverlapRetriever:
    """Trivial lexical-overlap retriever used only if rank_bm25 is unavailable."""

    def __init__(self, corpus: List[CorpusDoc]):
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 10):
        from retrievers.bm25_retriever import RetrievedDoc
        q = set(simple_tokenize(query))
        scored = []
        for doc in self.corpus:
            d = set(simple_tokenize(doc.full_text))
            scored.append((len(q & d), doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedDoc(doc_id=doc.doc_id, text=doc.text, score=float(s), rank=i + 1)
            for i, (s, doc) in enumerate(scored[:top_k])
        ]


# --------------------------------------------------------------------------- #
# Real run (config-driven)
# --------------------------------------------------------------------------- #
def _load_config(path: str) -> ExperimentConfig:
    import yaml  # optional dep; only needed for the real path

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    thr = raw.get("thresholds", {})
    smp = raw.get("sampling", {})
    return ExperimentConfig(
        dataset=raw.get("dataset", "nfcorpus"),
        data_dir=raw.get("data_dir", "./datasets"),
        split=raw.get("split", "test"),
        retriever=raw.get("retriever", "bm25"),
        reranker=raw.get("reranker", "retriever_only"),
        top_k=int(raw.get("top_k", 10)),
        ndcg_k=int(raw.get("ndcg_k", 5)),
        max_queries=raw.get("max_queries"),
        thresholds=LCRThresholds(
            t_query=float(thr.get("t_query", 0.5)),
            t_upper=float(thr.get("t_upper", 0.7)),
            t_lower=float(thr.get("t_lower", 0.3)),
        ),
        sampling=SamplingConfig(K=int(smp.get("K", 10)),
                                temperature=float(smp.get("temperature", 1.0))),
    )


def _build_llm(model_cfg: dict) -> LLMClient:
    from confidence.sampling import HFTransformersClient, VLLMClient
    backend = (model_cfg.get("backend") or "vllm").lower()
    name = model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct")
    if backend == "vllm":
        return VLLMClient(model_name=name)
    return HFTransformersClient(model_name=name,
                                device=model_cfg.get("device", "cuda"))


def run_real(config_path: str) -> Dict[str, object]:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = _load_config(config_path)
    from data.beir_loader import load_beir
    corpus, queries, qrels = load_beir(config.dataset, config.data_dir, config.split)
    llm = _build_llm(raw.get("model", {}))
    return run_experiment(config, llm, corpus, queries, qrels)


def _print_results(res: Dict[str, object]) -> None:
    print("\n" + "=" * 56)
    print(f"  LCR reproduction — dataset={res['dataset']} reranker={res['reranker']}")
    print("=" * 56)
    print(f"  queries evaluated : {res['num_queries']}")
    print(f"  NDCG@{res['ndcg_k']} (base)  : {res['base_ndcg']:.4f}")
    print(f"  NDCG@{res['ndcg_k']} (+LCR)  : {res['lcr_ndcg']:.4f}")
    print(f"  absolute gain     : {res['abs_gain']:+.4f}")
    print(f"  relative gain     : {res['rel_gain_pct']:+.2f}%")
    print("-" * 56)
    if res["abs_gain"] > 0:
        print("  LCR re-sorted low-confidence queries and improved NDCG@5.")
    elif res["abs_gain"] == 0:
        print("  Safe lower bound held: LCR did not degrade the base ranking.")
    else:
        print("  NDCG@5 decreased (check thresholds / confidence model).")
    if res.get("dataset") == "toy":
        print("  (smoke: mock confidence model — paper reports up to +20.6% with a")
        print("   real 7-9B LLM; run --config for the full BEIR reproduction.)")
    print("=" * 56 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LCR reproduction runner")
    parser.add_argument("--smoke", action="store_true",
                        help="Run end-to-end on a toy corpus with a mock LLM ($0, no GPU).")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to experiment config (real run).")
    args = parser.parse_args(argv)

    if args.smoke:
        print("[smoke] running LCR pipeline on toy data with MockLLM…")
        res = run_smoke()
    else:
        res = run_real(args.config)
    _print_results(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
