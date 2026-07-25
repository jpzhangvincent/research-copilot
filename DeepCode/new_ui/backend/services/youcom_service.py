"""You.com API client for the Research Copilot (Scout + Librarian).

Zero-dependency (stdlib only) client over the You.com REST endpoints, mirroring
the youcom-agent-starter-kit's `youcom.py`. Two hostnames:
  - search / contents -> https://ydc-index.io/v1/*
  - research          -> https://api.you.com/v1/*

The API key is resolved from (in order):
  1. env var YDC_API_KEY
  2. the repo-root .mcp.json bearer token (gitignored; where it currently lives)
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

SEARCH_URL = "https://ydc-index.io/v1/search"
CONTENTS_URL = "https://ydc-index.io/v1/contents"
RESEARCH_URL = "https://api.you.com/v1/research"

# Research pricing (per call, USD) — used for the live "You.com spend" meter.
RESEARCH_COST = {"lite": 0.012, "standard": 0.05, "deep": 0.10, "exhaustive": 0.45}
SEARCH_COST = 0.005

# Session spend meter (approximate, from published per-call pricing).
_SPEND: Dict[str, Any] = {"total": 0.0, "search": 0, "research": 0}


def _track(kind: str, cost: float) -> None:
    _SPEND["total"] = round(_SPEND["total"] + cost, 4)
    _SPEND[kind] = _SPEND.get(kind, 0) + 1


def get_spend() -> Dict[str, Any]:
    """Approximate cumulative You.com spend this session (USD)."""
    return dict(_SPEND)

# new_ui/backend/services -> repo root is 4 parents up (…/youhack/DeepCode) then one more to youhack
_BACKEND_SERVICES_DIR = Path(__file__).resolve().parent
_YOUHACK_ROOT = _BACKEND_SERVICES_DIR.parents[3]  # …/youhack


class YouComError(RuntimeError):
    """Raised when a You.com API call fails."""


@lru_cache(maxsize=1)
def _api_key() -> str:
    key = os.environ.get("YDC_API_KEY") or os.environ.get("YOU_API_KEY")
    if key:
        return key.strip()
    mcp_path = _YOUHACK_ROOT / ".mcp.json"
    if mcp_path.exists():
        try:
            cfg = json.loads(mcp_path.read_text())
            auth = cfg["mcpServers"]["you-com"]["headers"]["Authorization"]
            return auth.replace("Bearer", "").strip()
        except (KeyError, json.JSONDecodeError):
            pass
    raise YouComError(
        "No You.com API key found. Set YDC_API_KEY or provide .mcp.json."
    )


def _request(url: str, *, method: str, params: Optional[Dict[str, Any]] = None,
             body: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-API-Key", _api_key())
    req.add_header("Content-Type", "application/json")
    # api.you.com sits behind Cloudflare, which 403s (error 1010) the default
    # urllib User-Agent. A browser-like UA is required for the research endpoint.
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0 Safari/537.36")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        detail = exc.read().decode(errors="replace")[:500]
        raise YouComError(f"You.com {method} {url} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network
        raise YouComError(f"You.com request failed: {exc}") from exc


def search(query: str, *, count: int = 10, freshness: Optional[str] = None,
           include_domains: Optional[List[str]] = None) -> Dict[str, Any]:
    """Web/news search. Returns the raw You.com response."""
    params: Dict[str, Any] = {"query": query, "count": count}
    if freshness:
        params["freshness"] = freshness
    if include_domains:
        params["include_domains"] = include_domains
    result = _request(SEARCH_URL, method="GET", params=params)
    _track("search", SEARCH_COST)
    return result


def contents(urls: List[str], *, fmt: str = "markdown") -> Dict[str, Any]:
    """Extract page content for the given URLs."""
    return _request(CONTENTS_URL, method="POST",
                    body={"urls": urls, "formats": [fmt]})


def _unwrap_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Research responses nest the payload under "output":
    {"output": {content, content_type, sources}, "warnings"}.
    Return the inner dict so callers read .content/.sources directly.
    """
    if isinstance(data, dict) and isinstance(data.get("output"), dict):
        out = data["output"]
        if data.get("warnings"):
            out.setdefault("warnings", data["warnings"])
        return out
    return data


def research(input_text: str, *, effort: str = "standard",
             output_schema: Optional[Dict[str, Any]] = None,
             source_control: Optional[Dict[str, Any]] = None,
             timeout: int = 300) -> Dict[str, Any]:
    """Agentic multi-step research with cited sources.

    effort: lite | standard | deep | exhaustive (output_schema unsupported on lite).
    Returns the unwrapped output dict: {content, content_type, sources, ...}.
    """
    body: Dict[str, Any] = {"input": input_text, "research_effort": effort}
    if output_schema:
        if effort == "lite":
            raise ValueError("output_schema is not supported with effort='lite'")
        body["output_schema"] = output_schema
    if source_control:
        body["source_control"] = source_control
    out = _unwrap_output(
        _request(RESEARCH_URL, method="POST", body=body, timeout=timeout)
    )
    _track("research", RESEARCH_COST.get(effort, RESEARCH_COST["standard"]))
    return out


_CITATION_RE = re.compile(r"\[\[(\d+(?:\s*,\s*\d+)*)\]\]")


def render_citations(content: str, sources: List[Dict[str, Any]]) -> str:
    """Replace inline [[1, 2]] markers with markdown links into `sources` (1-indexed)."""
    def repl(match: "re.Match[str]") -> str:
        links = []
        for tok in match.group(1).split(","):
            i = int(tok)
            if 1 <= i <= len(sources):
                links.append(f"[{i}]({sources[i - 1].get('url', '')})")
            else:
                links.append(str(i))
        return "[" + ", ".join(links) + "]"

    return _CITATION_RE.sub(repl, content or "")


_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")


def extract_paper_hits(raw: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flatten a search response into paper-shaped hits (title/url/snippet/date)."""
    hits: List[Dict[str, str]] = []
    results = raw.get("results") or raw.get("hits") or []
    # You.com nests results under "results" -> list of {url,title,description,...}
    if isinstance(results, dict):
        results = results.get("web") or results.get("results") or []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = r.get("url") or r.get("link") or ""
        title = (r.get("title") or "").strip()
        snippet = (r.get("description") or r.get("snippet") or "").strip()
        published = r.get("published") or r.get("page_age") or ""
        m = _ARXIV_ID_RE.search(url)
        arxiv_id = m.group(1) if m else ""
        hits.append({
            "title": title,
            "url": url,
            "snippet": snippet[:400],
            "published": str(published),
            "arxiv_id": arxiv_id,
        })
    return hits
