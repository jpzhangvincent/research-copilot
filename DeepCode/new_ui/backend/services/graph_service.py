"""Knowledge-graph builder for the AI Research Copilot (the "Cartographer").

Uses the You.com Research API with a structured ``output_schema`` to perform
multi-step web research on a topic and return a typed knowledge graph
(papers, methods, datasets, metrics, concepts) plus cited sources — in a
single grounded call. Graphs are cached to disk (``wiki/graph/<slug>.json``)
so the demo can load an instant, guaranteed result and optionally refresh live.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from services import youcom_service

_BACKEND_SERVICES_DIR = Path(__file__).resolve().parent
_YOUHACK_ROOT = _BACKEND_SERVICES_DIR.parents[3]  # …/youhack
_GRAPH_DIR = _YOUHACK_ROOT / "wiki" / "graph"

_NODE_TYPES = {"paper", "method", "dataset", "metric", "concept", "task"}

# JSON-Schema (subset) the Research API fills in. Rules: object root,
# additionalProperties=false, every property required, <=5 depth.
GRAPH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "nodes", "edges"],
    "properties": {
        "summary": {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "type", "detail"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(_NODE_TYPES)},
                    "detail": {"type": "string"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source", "target", "relation"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                },
            },
        },
    },
}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "graph")[:60]


def _build_prompt(topic: str, seed: Optional[str]) -> str:
    seed_line = (
        f" Anchor the map on the paper '{seed}' and its neighbourhood."
        if seed else ""
    )
    return (
        f"Build a knowledge graph of the research landscape around '{topic}'.{seed_line} "
        "Identify the key recent papers, the methods/algorithms they propose, the "
        "datasets and benchmarks used, the evaluation metrics, and the core concepts. "
        "Return nodes with stable snake-case ids prefixed by type "
        "(e.g. 'paper:lcr', 'method:mscp', 'dataset:beir', 'metric:ndcg5') and the "
        "relations between them using verbs like proposes, builds_on, evaluates_on, "
        "uses_concept, uses_retriever, improves_metric, compared_to. Keep it focused: "
        "at most ~40 nodes centred on the most important, recent work."
    )


def _normalize(content: Dict[str, Any],
               sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce raw model output into a clean, render-safe graph."""
    raw_nodes = content.get("nodes") or []
    raw_edges = content.get("edges") or []

    nodes: Dict[str, Dict[str, Any]] = {}
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "").strip()
        if not nid:
            continue
        ntype = str(n.get("type") or "concept").strip().lower()
        if ntype not in _NODE_TYPES:
            ntype = "concept"
        nodes[nid] = {
            "id": nid,
            "label": str(n.get("label") or nid).strip()[:80],
            "type": ntype,
            "detail": str(n.get("detail") or "").strip()[:400],
            "url": "",
        }

    edges: List[Dict[str, str]] = []
    seen_edges = set()
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        rel = str(e.get("relation") or "related_to").strip()[:40]
        if not src or not tgt or src == tgt:
            continue
        # Create placeholder nodes for dangling endpoints so nothing is lost.
        for endpoint in (src, tgt):
            if endpoint not in nodes:
                nodes[endpoint] = {
                    "id": endpoint,
                    "label": endpoint.split(":")[-1].replace("_", " ")[:80],
                    "type": endpoint.split(":")[0]
                    if endpoint.split(":")[0] in _NODE_TYPES else "concept",
                    "detail": "",
                    "url": "",
                }
        key = (src, tgt, rel)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({"source": src, "target": tgt, "relation": rel})

    # Attach source URLs to paper nodes by fuzzy label match against citations.
    clean_sources = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        clean_sources.append({
            "title": (s.get("title") or "").strip()[:160],
            "url": s.get("url") or "",
            "snippet": (s.get("snippet") or s.get("description") or "").strip()[:240],
        })
    for node in nodes.values():
        if node["type"] != "paper" or node["url"]:
            continue
        label_words = set(re.findall(r"[a-z0-9]+", node["label"].lower()))
        best, best_score = None, 0
        for s in clean_sources:
            title_words = set(re.findall(r"[a-z0-9]+", s["title"].lower()))
            score = len(label_words & title_words)
            if score > best_score:
                best, best_score = s, score
        if best and best_score >= 2:
            node["url"] = best["url"]

    return {
        "summary": str(content.get("summary") or "").strip(),
        "nodes": list(nodes.values()),
        "edges": edges,
        "sources": clean_sources,
    }


def build_graph(topic: str, *, seed: Optional[str] = None,
                effort: str = "standard", cache: bool = True) -> Dict[str, Any]:
    """Run You.com research with the graph schema and return a normalized graph."""
    t0 = time.time()
    out = youcom_service.research(
        _build_prompt(topic, seed),
        effort=effort,
        output_schema=GRAPH_SCHEMA,
        source_control={"freshness": "year"},
    )
    content = out.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = {"summary": content, "nodes": [], "edges": []}
    if not isinstance(content, dict):
        content = {"summary": "", "nodes": [], "edges": []}

    graph = _normalize(content, out.get("sources") or [])
    graph.update({
        "topic": topic,
        "seed": seed or "",
        "effort": effort,
        "elapsed": round(time.time() - t0, 1),
        "cached": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spend": youcom_service.get_spend(),
    })

    if cache:
        _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        slug = _slug(topic)
        (_GRAPH_DIR / f"{slug}.json").write_text(
            json.dumps(graph, indent=2), encoding="utf-8"
        )
        (_GRAPH_DIR / "_latest.json").write_text(
            json.dumps(graph, indent=2), encoding="utf-8"
        )
    return graph


def load_cached(topic: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a cached graph by topic slug, or the most recent one."""
    if not _GRAPH_DIR.is_dir():
        return None
    path = _GRAPH_DIR / (f"{_slug(topic)}.json" if topic else "_latest.json")
    if not path.exists():
        path = _GRAPH_DIR / "_latest.json"
    if not path.exists():
        return None
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
        graph["cached"] = True
        return graph
    except json.JSONDecodeError:
        return None
