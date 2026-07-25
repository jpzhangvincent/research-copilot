"""Librarian service — compile a discovered paper into an llm-wiki article.

Flow (see ADR-0003): You.com contents fetches the paper page -> OpenAI writes an
article in karpathy-llm-wiki format -> saved to <youhack>/wiki/<topic>/<slug>.md,
with index.md and log.md updated. Files are Obsidian-compatible markdown.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from services import youcom_service

_BACKEND_SERVICES_DIR = Path(__file__).resolve().parent
_YOUHACK_ROOT = _BACKEND_SERVICES_DIR.parents[3]  # …/youhack
_WIKI_ROOT = _YOUHACK_ROOT / "wiki"

# By default the Librarian uses the subscription-backed shim (zero API cost).
# Override COPILOT_WIKI_BASE_URL / COPILOT_WIKI_MODEL to use a real provider.
_WIKI_BASE_URL = os.environ.get("COPILOT_WIKI_BASE_URL", "http://127.0.0.1:8899/v1")
_WIKI_MODEL = os.environ.get("COPILOT_WIKI_MODEL", "vllm/claude")
_WIKI_API_KEY = os.environ.get("COPILOT_WIKI_API_KEY", "shim")

_ARTICLE_SYSTEM = """You are the Librarian of a personal research wiki.
Write ONE concise knowledge article in Markdown about the given research paper.
Rules:
- Start with a single H1 title (the concept/method name, not the paper filename).
- Sections: ## Summary (2-3 sentences), ## Key Ideas (3-6 bullets),
  ## Method (how it works), ## Results (concrete numbers/metrics if present),
  ## Why It Matters.
- Only state facts supported by the provided text. Preserve exact numbers/metrics.
- End with a metadata block:
  ---
  Sources: <authors/venue + date>
  Raw: <arxiv url>
  Updated: <today's date>
No preamble, no code fences around the whole thing. Output only the article Markdown."""


def _slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len].strip("-") or "untitled"


def _extract_page_text(raw: Any) -> str:
    """Pull markdown text out of a You.com contents response (defensive).

    The contents endpoint returns a top-level list of {url, markdown} dicts;
    older/other shapes nest under "results"/"contents".
    """
    if isinstance(raw, list):
        results = raw
    elif isinstance(raw, dict):
        results = raw.get("results") or raw.get("contents") or []
    else:
        results = []
    if isinstance(results, dict):
        results = [results]
    chunks = []
    for r in results:
        if not isinstance(r, dict):
            continue
        for key in ("markdown", "content", "html", "text"):
            val = r.get(key)
            if isinstance(val, str) and val.strip():
                chunks.append(val)
                break
    if not chunks:
        # last resort: stringify
        chunks.append(str(raw))
    return "\n\n".join(chunks)[:20000]


def _openai_article(paper_url: str, page_text: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=_WIKI_BASE_URL, api_key=_WIKI_API_KEY)
    resp = client.chat.completions.create(
        model=_WIKI_MODEL,
        messages=[
            {"role": "system", "content": _ARTICLE_SYSTEM},
            {"role": "user", "content": f"Paper URL: {paper_url}\n\nPage content:\n{page_text}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _title_from_article(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _ensure_wiki_skeleton() -> None:
    _WIKI_ROOT.mkdir(parents=True, exist_ok=True)
    index = _WIKI_ROOT / "index.md"
    if not index.exists():
        index.write_text("# Knowledge Base Index\n\n")
    log = _WIKI_ROOT / "log.md"
    if not log.exists():
        log.write_text("# Wiki Log\n\n")


def _append_index(topic: str, title: str, rel_path: str, summary: str, today: str) -> None:
    index = _WIKI_ROOT / "index.md"
    body = index.read_text()
    entry = f"- [{title}]({rel_path}) — {summary} _(Updated: {today})_\n"
    section = f"## {topic}\n"
    if section in body:
        body = body.replace(section, section + entry, 1)
    else:
        body = body.rstrip() + f"\n\n{section}{entry}"
    index.write_text(body)


def _append_log(title: str, topic: str, rel_path: str, today: str) -> None:
    log = _WIKI_ROOT / "log.md"
    log.write_text(
        log.read_text().rstrip()
        + f"\n\n## [{today}] ingest | {title}\n- Disposition: New\n- Topic: {topic}\n- Article: {rel_path}\n"
    )


def compile_paper(paper_url: str, topic: str = "llm-agents",
                  interest: Optional[str] = None) -> Dict[str, Any]:
    """Compile a paper URL into a wiki article. Returns metadata + markdown."""
    raw = youcom_service.contents([paper_url], fmt="markdown")
    page_text = _extract_page_text(raw)
    article_md = _openai_article(paper_url, page_text)

    title = _title_from_article(article_md, fallback="Untitled Paper")
    slug = _slugify(title)
    today = _dt.date.today().isoformat()

    _ensure_wiki_skeleton()
    topic_dir = _WIKI_ROOT / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    article_path = topic_dir / f"{slug}.md"
    article_path.write_text(article_md + "\n")

    # first non-heading paragraph as summary for the index
    summary = ""
    for line in article_md.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("##"):
            summary = s[:160]
            break

    rel_path = f"{topic}/{slug}.md"
    _append_index(topic, title, rel_path, summary or title, today)
    _append_log(title, topic, rel_path, today)

    return {
        "title": title,
        "topic": topic,
        "path": str(article_path),
        "rel_path": rel_path,
        "markdown": article_md,
        "wiki_root": str(_WIKI_ROOT),
    }
