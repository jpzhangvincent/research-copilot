"""Minimal TREC-format helpers (paper also reports on TREC-DL, §4.1).

Reads TREC qrels (`qid 0 docid rel`) and TREC runs (`qid Q0 docid rank score tag`).
Kept dependency-free; useful for scoring against pre-computed run files.
"""
from __future__ import annotations

from typing import Dict, List

QRelDict = Dict[str, Dict[str, int]]


def load_trec_qrels(path: str) -> QRelDict:
    qrels: QRelDict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, did, rel = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(qid, {})[did] = int(float(rel))
    return qrels


def load_trec_run(path: str) -> Dict[str, List[str]]:
    """Return {qid: [doc_ids ranked best-first]} from a TREC run file."""
    scored: Dict[str, List[tuple]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, did, rank = parts[0], parts[2], parts[3]
            scored.setdefault(qid, []).append((int(rank), did))
    return {qid: [d for _, d in sorted(items)] for qid, items in scored.items()}
