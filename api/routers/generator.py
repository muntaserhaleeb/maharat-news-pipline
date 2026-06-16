import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.db import get_conn

router = APIRouter()

# Lazily initialised — loads embedding models + Qdrant + Claude client on first draft
_pipeline_cache: Dict[str, Any] = {}


def _get_pipeline():
    if "main" not in _pipeline_cache:
        from pipelines.drafting_pipeline import DraftingPipeline
        _pipeline_cache["main"] = DraftingPipeline.from_config()
    return _pipeline_cache["main"]


# ── Pydantic models ───────────────────────────────────────────────────────────

class DraftRequest(BaseModel):
    topic: str
    mode: str = "website_news"
    article_type: Optional[str] = None
    use_knowledge: bool = False
    dry_run: bool = False
    year: Optional[int] = None
    limit: Optional[int] = None
    score_threshold: Optional[float] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/modes")
def get_modes() -> Dict[str, Any]:
    """Return available generation modes and article types from generation.yaml."""
    from services.config_service import load_generation_config
    from services.style_service import VALID_ARTICLE_TYPES
    cfg   = load_generation_config().get("generation", {})
    modes = cfg.get("generation_modes", {})
    return {
        "model": cfg.get("model", ""),
        "modes": [
            {
                "key":         k,
                "description": v.get("description", k),
                "word_range":  v.get("length", {}),
            }
            for k, v in modes.items()
        ],
        "article_types": list(VALID_ARTICLE_TYPES),
    }


@router.post("/draft")
def create_draft(body: DraftRequest) -> Dict[str, Any]:
    """Run the full RAG drafting pipeline (non-streaming)."""
    if not body.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            400,
            "ANTHROPIC_API_KEY is not set. "
            "Export it in your shell before starting the backend: "
            "export ANTHROPIC_API_KEY=sk-ant-..."
        )
    if not body.topic.strip():
        raise HTTPException(400, "topic must not be empty")

    job_id     = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    conn = get_conn()
    with conn:
        conn.execute(
            """INSERT INTO generation_jobs
               (job_id, topic, mode, article_type, status, created_at)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (job_id, body.topic, body.mode, body.article_type, created_at),
        )
    conn.close()

    t0 = time.time()
    try:
        pipeline = _get_pipeline()
        result = pipeline.draft(
            topic          = body.topic,
            generation_mode = body.mode or None,
            article_type   = body.article_type,
            year           = body.year,
            limit          = body.limit,
            score_threshold = body.score_threshold,
            dry_run        = body.dry_run,
            stream         = False,
            use_knowledge  = body.use_knowledge,
        )
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000)
        _mark_job(job_id, "error", error=str(exc))
        raise HTTPException(500, str(exc))

    elapsed_ms = round((time.time() - t0) * 1000)

    if body.dry_run or result is None:
        _mark_job(job_id, "done")
        return {
            "job_id":     job_id,
            "dry_run":    True,
            "elapsed_ms": elapsed_ms,
            "result":     None,
        }

    result_dict = result.to_dict()
    _mark_job(job_id, "done", result_json=json.dumps(result_dict))

    return {
        "job_id":     job_id,
        "dry_run":    False,
        "elapsed_ms": elapsed_ms,
        "result": {
            "headline":          result.headline,
            "summary":           result.summary,
            "body":              result.body,
            "suggested_slug":    result.suggested_slug,
            "seo_summary":       result.seo_summary,
            "hashtags":          result.hashtags,
            "qa_warnings":       result.qa_warnings,
            "entities_detected": result.entities_detected,
            "sources_used":      result.sources_used,
            "model":             result.model,
            "input_tokens":      result.input_tokens,
            "output_tokens":     result.output_tokens,
            "generated_at":      result.generated_at,
        },
    }


@router.get("/jobs")
def list_jobs() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT job_id, topic, mode, article_type, status, created_at, finished_at, error
           FROM generation_jobs ORDER BY id DESC LIMIT 20"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM generation_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Job not found")
    d = dict(row)
    if d.get("result_json"):
        d["result"] = json.loads(d.pop("result_json"))
    else:
        d.pop("result_json", None)
        d["result"] = None
    return d


# ── helpers ───────────────────────────────────────────────────────────────────

def _mark_job(
    job_id: str,
    status: str,
    result_json: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            """UPDATE generation_jobs
               SET status=?, finished_at=?, result_json=?, error=?
               WHERE job_id=?""",
            (status, datetime.utcnow().isoformat(), result_json, error, job_id),
        )
    conn.close()
