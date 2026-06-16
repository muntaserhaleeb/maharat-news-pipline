import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from api.db import get_conn

router = APIRouter()

REVIEW_STATUSES = ("pending_review", "approved", "rejected")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _parse_job(row: Any) -> Dict[str, Any]:
    d = dict(row)
    result: Optional[Dict] = None
    draft:  Optional[Dict] = None
    if d.get("result_json"):
        try:
            result = json.loads(d["result_json"])
        except Exception:
            pass
    if d.get("draft_json"):
        try:
            draft = json.loads(d["draft_json"])
        except Exception:
            pass
    d["result"] = result
    d["draft"]  = draft
    d.pop("result_json", None)
    d.pop("draft_json",  None)
    if not d.get("review_status"):
        d["review_status"] = "pending_review"
    return d


def _effective(job: Dict[str, Any]) -> Dict[str, Any]:
    """Merge original result with any saved draft overrides."""
    r = job.get("result") or {}
    ov = job.get("draft") or {}
    return {
        "headline":         ov.get("headline")        or r.get("headline", ""),
        "summary":          ov.get("summary")         or r.get("summary", ""),
        "body":             ov.get("body")             or r.get("body", ""),
        "suggested_slug":   ov.get("suggested_slug")  or r.get("suggested_slug", ""),
        "seo_summary":      r.get("seo_summary", ""),
        "hashtags":         r.get("hashtags") or [],
        "qa_warnings":      r.get("qa_warnings") or [],
        "entities_detected":r.get("entities_detected") or {},
        "sources_used":     r.get("sources_used") or [],
        "model":            r.get("model", ""),
        "input_tokens":     r.get("input_tokens"),
        "output_tokens":    r.get("output_tokens"),
        "generated_at":     r.get("generated_at", ""),
    }


# ── Pydantic models ───────────────────────────────────────────────────────────

class DraftUpdate(BaseModel):
    headline:       Optional[str] = None
    summary:        Optional[str] = None
    body:           Optional[str] = None
    suggested_slug: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def list_jobs(review_status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_conn()
    base = """
        SELECT job_id, topic, mode, article_type, status,
               COALESCE(review_status, 'pending_review') AS review_status,
               created_at, finished_at
        FROM generation_jobs
        WHERE status = 'done' AND result_json IS NOT NULL
    """
    if review_status and review_status in REVIEW_STATUSES:
        rows = conn.execute(
            base + " AND COALESCE(review_status, 'pending_review') = ? ORDER BY id DESC LIMIT 50",
            (review_status,),
        ).fetchall()
    else:
        rows = conn.execute(base + " ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    conn = get_conn()
    row  = conn.execute(
        "SELECT * FROM generation_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Job not found")
    job = _parse_job(row)
    job["effective"] = _effective(job)
    return job


@router.put("/{job_id}/draft")
def update_draft(job_id: str, body: DraftUpdate) -> Dict[str, Any]:
    conn  = get_conn()
    row   = conn.execute(
        "SELECT draft_json FROM generation_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Job not found")
    existing: Dict[str, Any] = {}
    if row["draft_json"]:
        try:
            existing = json.loads(row["draft_json"])
        except Exception:
            pass
    updates = {k: v for k, v in body.dict().items() if v is not None}
    existing.update(updates)
    with conn:
        conn.execute(
            "UPDATE generation_jobs SET draft_json = ? WHERE job_id = ?",
            (json.dumps(existing), job_id),
        )
    conn.close()
    return {"ok": True, "job_id": job_id}


@router.post("/{job_id}/approve")
def approve(job_id: str) -> Dict[str, Any]:
    _set_review_status(job_id, "approved")
    return {"ok": True, "job_id": job_id, "review_status": "approved"}


@router.post("/{job_id}/reject")
def reject(job_id: str) -> Dict[str, Any]:
    _set_review_status(job_id, "rejected")
    return {"ok": True, "job_id": job_id, "review_status": "rejected"}


@router.post("/{job_id}/reset")
def reset_review(job_id: str) -> Dict[str, Any]:
    _set_review_status(job_id, "pending_review")
    return {"ok": True, "job_id": job_id, "review_status": "pending_review"}


def _set_review_status(job_id: str, status: str) -> None:
    conn = get_conn()
    row  = conn.execute(
        "SELECT job_id FROM generation_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Job not found")
    with conn:
        conn.execute(
            "UPDATE generation_jobs SET review_status = ? WHERE job_id = ?",
            (status, job_id),
        )
    conn.close()


# ── Export ────────────────────────────────────────────────────────────────────

def _safe_slug(slug: str, topic: str) -> str:
    if slug:
        return re.sub(r"[^\w-]", "-", slug.strip().lower())
    return re.sub(r"[^\w-]", "-", topic.strip().lower())[:60]


def _export_markdown(eff: Dict[str, Any], job: Dict[str, Any]) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    slug  = _safe_slug(eff.get("suggested_slug", ""), job.get("topic", "draft"))
    tags  = ", ".join(f'"{t.lstrip("#")}"' for t in (eff.get("hashtags") or [])[:7])
    return "\n".join([
        "---",
        f'title: "{eff.get("headline", "")}"',
        f'slug: "{slug}"',
        f'date: "{today}"',
        f'summary: "{eff.get("summary", "")}"',
        f'seo_description: "{eff.get("seo_summary", "") or eff.get("summary", "")}"',
        f"tags: [{tags}]",
        'category: ""',
        'status: "draft"',
        'language: "en"',
        "---",
        "",
        f'# {eff.get("headline", "")}',
        "",
        eff.get("summary", ""),
        "",
        eff.get("body", ""),
    ])


def _md_to_html_body(text: str) -> str:
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$",  r"<h2>\1</h2>",  text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$",   r"<h1>\1</h1>",   text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    out = []
    for block in text.split("\n\n"):
        b = block.strip()
        if not b:
            continue
        if b.startswith("<h") or b.startswith("<ul") or b.startswith("<ol"):
            out.append(b)
        else:
            out.append(f"<p>{b}</p>")
    return "\n".join(out)


def _export_html(eff: Dict[str, Any]) -> str:
    headline = eff.get("headline", "")
    summary  = eff.get("summary", "")
    body_html = _md_to_html_body(eff.get("body", ""))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{headline}</title>
  <style>
    body {{ font-family: sans-serif; max-width: 800px; margin: 2rem auto; line-height: 1.6; color: #222; }}
    h1 {{ font-size: 1.8rem; margin-bottom: .5rem; }}
    .summary {{ color: #555; font-size: 1.05rem; margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <article>
    <h1>{headline}</h1>
    <p class="summary"><em>{summary}</em></p>
    {body_html}
  </article>
</body>
</html>"""


def _export_payload(eff: Dict[str, Any], job: Dict[str, Any]) -> str:
    today = datetime.utcnow().isoformat() + "Z"
    slug  = _safe_slug(eff.get("suggested_slug", ""), job.get("topic", "draft"))
    tags  = [t.lstrip("#") for t in (eff.get("hashtags") or [])]
    payload = {
        "_note": "Verify field names match your Payload CMS collection schema before importing.",
        "title":          eff.get("headline", ""),
        "slug":           slug,
        "status":         "draft",
        "publishedAt":    None,
        "excerpt":        eff.get("summary", ""),
        "content":        eff.get("body", ""),
        "seoTitle":       eff.get("headline", ""),
        "seoDescription": eff.get("seo_summary", "") or eff.get("summary", ""),
        "tags":           tags,
        "language":       "en",
        "category":       job.get("mode", ""),
        "entities":       eff.get("entities_detected") or {},
        "generatedBy":    eff.get("model", ""),
        "sourceJobId":    job.get("job_id", ""),
        "createdAt":      today,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


@router.get("/{job_id}/export")
def export_job(
    job_id: str,
    format: str = Query("markdown"),
) -> Response:
    if format not in ("markdown", "html", "payload"):
        raise HTTPException(400, "format must be markdown, html, or payload")
    conn = get_conn()
    row  = conn.execute(
        "SELECT * FROM generation_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Job not found")

    job = _parse_job(row)
    eff = _effective(job)
    slug = _safe_slug(eff.get("suggested_slug", ""), job.get("topic", "draft"))

    if format == "markdown":
        content    = _export_markdown(eff, job)
        filename   = f"{slug}.md"
        media_type = "text/markdown; charset=utf-8"
    elif format == "html":
        content    = _export_html(eff)
        filename   = f"{slug}.html"
        media_type = "text/html; charset=utf-8"
    else:
        content    = _export_payload(eff, job)
        filename   = f"{slug}-payload.json"
        media_type = "application/json; charset=utf-8"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
