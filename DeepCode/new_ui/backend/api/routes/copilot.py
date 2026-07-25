"""Research Copilot routes — Scout (discovery) and Librarian (wiki compile).

Scout uses You.com search to surface recent papers from an interest.
Librarian compiles a chosen paper into an llm-wiki article (see ADR-0003).
Reproduction reuses the existing /api/v1/workflows/paper-to-code flow.
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import graph_service, librarian_service, youcom_service

router = APIRouter()

# new_ui/backend/api/routes/copilot.py -> DeepCode root is parents[4]
_DEEPCODE_ROOT = Path(__file__).resolve().parents[4]
_TASKS_DIR = _DEEPCODE_ROOT / "deepcode_lab" / "tasks"


def _find_prebaked_repo():
    """Locate a completed reproduction's generated repo dir.

    Uses COPILOT_PREBAKED_TASK (task dir name) if set, else the most recently
    modified paper_* task that has a generate_code/<repo>/ directory.
    """
    override = os.environ.get("COPILOT_PREBAKED_TASK")
    if override:
        candidates = [_TASKS_DIR / override]
    else:
        candidates = sorted(
            _TASKS_DIR.glob("paper_*"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
    for task in candidates:
        gc = task / "generate_code"
        if gc.is_dir():
            repos = [d for d in gc.iterdir() if d.is_dir()]
            if repos:
                return task, repos[0]
    return None, None


class DiscoverRequest(BaseModel):
    interest: str
    count: int = 8
    freshness: Optional[str] = "year"
    arxiv_only: bool = True


class PaperHit(BaseModel):
    title: str
    url: str
    snippet: str
    published: str
    arxiv_id: str


class DiscoverResponse(BaseModel):
    interest: str
    query: str
    papers: List[PaperHit]


class CompileRequest(BaseModel):
    paper_url: str
    topic: str = "llm-agents"
    interest: Optional[str] = None
    refresh: bool = False  # force re-compile, bypassing the cache


class CompileResponse(BaseModel):
    title: str
    topic: str
    rel_path: str
    path: str
    markdown: str
    wiki_root: str
    cached: bool = False
    cached_at: Optional[str] = None


@router.post("/discover", response_model=DiscoverResponse)
async def discover(request: DiscoverRequest):
    """Scout: find recent papers for an interest via You.com search."""
    query = f"{request.interest} arxiv paper"
    if request.arxiv_only:
        query = f"{request.interest} recent method"
    include = ["arxiv.org"] if request.arxiv_only else None
    try:
        raw = youcom_service.search(
            query, count=request.count, freshness=request.freshness,
            include_domains=include,
        )
    except youcom_service.YouComError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    hits = youcom_service.extract_paper_hits(raw)
    hits = [h for h in hits if h["url"]]
    return DiscoverResponse(
        interest=request.interest, query=query,
        papers=[PaperHit(**h) for h in hits],
    )


@router.post("/wiki/compile", response_model=CompileResponse)
async def compile_wiki(request: CompileRequest):
    """Librarian: compile a paper into an llm-wiki article."""
    import asyncio

    try:
        result = await asyncio.to_thread(
            librarian_service.compile_paper,
            request.paper_url, request.topic, request.interest,
            not request.refresh,
        )
    except youcom_service.YouComError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Compile failed: {exc}") from exc
    return CompileResponse(**result)


# ---------------------------------------------------------------------------
# Pre-baked reproduction (demo payoff, per ADR-0002)
# ---------------------------------------------------------------------------

_DOC_LIMIT = 60_000
_FILE_LIMIT = 200_000


class PrebakedFile(BaseModel):
    path: str
    size: int


class PrebakedResponse(BaseModel):
    task_id: str
    repo_name: str
    file_count: int
    files: List[PrebakedFile]
    readme: str
    summary: str
    plan: str


class PrebakedFileResponse(BaseModel):
    path: str
    content: str
    truncated: bool


def _read_text(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[:limit]


@router.get("/prebaked", response_model=PrebakedResponse)
async def prebaked():
    """Return the cached reproduction repo (file tree + key docs) for the demo."""
    task, repo = _find_prebaked_repo()
    if repo is None:
        raise HTTPException(status_code=404, detail="No pre-baked reproduction found")

    files: List[PrebakedFile] = []
    for p in sorted(repo.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files.append(
                PrebakedFile(path=str(p.relative_to(repo)), size=p.stat().st_size)
            )

    summary_path = task / "implement_code_summary.md"
    plan_path = task / "initial_plan.txt"
    return PrebakedResponse(
        task_id=task.name,
        repo_name=repo.name,
        file_count=len(files),
        files=files,
        readme=_read_text(repo / "README.md", _DOC_LIMIT),
        summary=_read_text(summary_path, _DOC_LIMIT),
        plan=_read_text(plan_path, _DOC_LIMIT),
    )


@router.get("/prebaked/file", response_model=PrebakedFileResponse)
async def prebaked_file(path: str):
    """Return the contents of one file inside the pre-baked repo."""
    task, repo = _find_prebaked_repo()
    if repo is None:
        raise HTTPException(status_code=404, detail="No pre-baked reproduction found")

    repo_root = repo.resolve()
    target = (repo / path).resolve()
    if not str(target).startswith(str(repo_root)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    full = target.read_text(encoding="utf-8", errors="replace")
    return PrebakedFileResponse(
        path=path, content=full[:_FILE_LIMIT], truncated=len(full) > _FILE_LIMIT
    )


# ---------------------------------------------------------------------------
# Knowledge graph (the Cartographer) — You.com Research w/ structured output
# ---------------------------------------------------------------------------


class GraphRequest(BaseModel):
    topic: str
    seed: Optional[str] = None
    effort: str = "standard"  # lite | standard | deep | exhaustive


@router.post("/graph")
async def build_graph(request: GraphRequest):
    """Cartographer: research a topic and return a typed knowledge graph."""
    import asyncio

    effort = request.effort if request.effort in graph_service.youcom_service.RESEARCH_COST else "standard"
    if effort == "lite":  # output_schema unsupported on lite
        effort = "standard"
    try:
        graph = await asyncio.to_thread(
            graph_service.build_graph, request.topic, seed=request.seed, effort=effort,
        )
    except youcom_service.YouComError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Graph build failed: {exc}") from exc
    return graph


@router.get("/graph")
async def cached_graph(topic: Optional[str] = None):
    """Return a cached knowledge graph (instant demo payoff)."""
    graph = graph_service.load_cached(topic)
    if graph is None:
        raise HTTPException(status_code=404, detail="No cached graph found")
    return graph


@router.get("/spend")
async def spend():
    """Approximate cumulative You.com spend this session (USD)."""
    return youcom_service.get_spend()
