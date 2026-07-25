"""BEIR dataset loader (NFCorpus by default; paper §4.1).

Loads a BEIR-format dataset into the (corpus, queries, qrels) triple the rest of
the pipeline expects. Prefers the official `beir` package (which downloads and
caches datasets); falls back to reading the standard BEIR on-disk layout
(`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) so no network is required if
the data is already present.

Returned types:
  corpus:  List[CorpusDoc]                     (from retrievers.bm25_retriever)
  queries: Dict[query_id, query_text]
  qrels:   Dict[query_id, Dict[doc_id, rel]]
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

from retrievers.bm25_retriever import CorpusDoc

QRelDict = Dict[str, Dict[str, int]]


def _load_from_disk(data_dir: str, split: str) -> Tuple[List[CorpusDoc], Dict[str, str], QRelDict]:
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries.jsonl")
    qrels_path = os.path.join(data_dir, "qrels", f"{split}.tsv")

    corpus: List[CorpusDoc] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus.append(
                CorpusDoc(
                    doc_id=str(row.get("_id") or row.get("id")),
                    text=row.get("text", ""),
                    title=row.get("title", ""),
                )
            )

    all_queries: Dict[str, str] = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_queries[str(row.get("_id") or row.get("id"))] = row.get("text", "")

    qrels: QRelDict = {}
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)  # query-id  corpus-id  score
        for parts in reader:
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            qrels.setdefault(qid, {})[did] = int(float(score))

    # Keep only queries that have judgments (BEIR test split convention).
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return corpus, queries, qrels


def load_beir(
    dataset: str = "nfcorpus",
    data_dir: str = "./datasets",
    split: str = "test",
) -> Tuple[List[CorpusDoc], Dict[str, str], QRelDict]:
    """Load a BEIR dataset. Downloads via `beir` if available, else reads disk."""
    dataset_path = os.path.join(data_dir, dataset)

    if not os.path.isdir(dataset_path):
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader

            url = (
                "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
                f"datasets/{dataset}.zip"
            )
            dataset_path = util.download_and_unzip(url, data_dir)
            corpus_raw, queries_raw, qrels = GenericDataLoader(dataset_path).load(split=split)
            corpus = [
                CorpusDoc(doc_id=did, text=doc.get("text", ""), title=doc.get("title", ""))
                for did, doc in corpus_raw.items()
            ]
            qrels_int = {q: {d: int(r) for d, r in rels.items()} for q, rels in qrels.items()}
            return corpus, dict(queries_raw), qrels_int
        except Exception as exc:  # pragma: no cover - network/optional dep
            raise FileNotFoundError(
                f"Dataset '{dataset}' not found at {dataset_path} and could not be "
                f"downloaded via the `beir` package ({exc}). Install beir or place "
                f"corpus.jsonl/queries.jsonl/qrels/{split}.tsv under {dataset_path}."
            ) from exc

    return _load_from_disk(dataset_path, split)
