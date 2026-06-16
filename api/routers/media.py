"""
Media Library router.

Provides event-folder scanning, image quality scoring, duplicate detection,
AI vision analysis, hero/gallery selection, metadata editing, and export.
"""

import base64
import hashlib
import io
import json
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from api.db import get_conn

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_MEDIA = ROOT / "output" / "media"

router = APIRouter()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}

# In-memory thumbnail cache (path:w → bytes)
_thumb_cache: Dict[str, bytes] = {}
MAX_THUMB_CACHE = 200

# Per-event processing lock (prevents concurrent scans on same event)
_scan_locks: Dict[str, threading.Lock] = {}


# ── Settings helpers ──────────────────────────────────────────────────────────

def _get_setting(key: str, default: str = "") -> str:
    conn = get_conn()
    row  = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def _set_setting(key: str, value: str) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
    conn.close()


# ── Event ID ──────────────────────────────────────────────────────────────────

def _event_id(base_dir: str, folder_name: str) -> str:
    return hashlib.sha256(f"{base_dir}::{folder_name}".encode()).hexdigest()[:16]


def _parse_folder_name(name: str) -> Tuple[str, Optional[str]]:
    """Parse 'YYYY-MM-DD Event Name' → (event_name, date). Falls back gracefully."""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(.+)$", name)
    if m:
        return m.group(2).strip(), m.group(1)
    return name.strip(), None


# ── Image quality scoring ─────────────────────────────────────────────────────

def _laplacian_var(gray: np.ndarray) -> float:
    """Approximation of Laplacian variance — high = sharp, low = blurry."""
    lap = (
        gray[1:-1, 1:-1].astype(np.float32)
        - 0.25 * (
            gray[:-2, 1:-1].astype(np.float32)
            + gray[2:, 1:-1].astype(np.float32)
            + gray[1:-1, :-2].astype(np.float32)
            + gray[1:-1, 2:].astype(np.float32)
        )
    )
    return float(np.var(lap))


def _score_image(img_path: Path) -> Dict[str, float]:
    file_bytes = img_path.stat().st_size
    with Image.open(img_path) as img:
        w, h = img.size
        gray = np.array(img.convert("L"))

    # Resolution: 3 MP → 100
    res = min(w * h / 3_000_000, 1.0) * 100

    # Sharpness: Laplacian variance; 1000+ = sharp
    lap = _laplacian_var(gray)
    sharpness = min(lap / 1_000, 1.0) * 100

    # Exposure: mean brightness should be 80–180 for good exposure
    mean_b = float(np.mean(gray))
    if 80 <= mean_b <= 180:
        exposure = 100.0
    elif mean_b < 80:
        exposure = (mean_b / 80) * 100
    else:
        exposure = max(0.0, (1.0 - (mean_b - 180) / 75) * 100)

    # File size: 1 MB → 100
    filesize = min(file_bytes / 1_048_576, 1.0) * 100

    total = 0.35 * res + 0.40 * sharpness + 0.15 * exposure + 0.10 * filesize

    return {
        "total":     round(total, 1),
        "resolution": round(res, 1),
        "sharpness":  round(sharpness, 1),
        "exposure":   round(exposure, 1),
        "filesize":   round(filesize, 1),
        "width":      w,
        "height":     h,
        "bytes":      file_bytes,
    }


# ── Perceptual duplicate detection (dhash) ────────────────────────────────────

def _dhash(img: Image.Image, size: int = 8) -> np.ndarray:
    img_gray = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.array(img_gray, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def _hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.count_nonzero(h1 != h2))


def _find_duplicate_groups(hashes: Dict[str, np.ndarray], threshold: int = 8) -> List[List[str]]:
    names = list(hashes)
    visited: set = set()
    groups: List[List[str]] = []
    for i, a in enumerate(names):
        if a in visited:
            continue
        grp = [a]
        visited.add(a)
        for b in names[i + 1:]:
            if b not in visited and _hamming(hashes[a], hashes[b]) <= threshold:
                grp.append(b)
                visited.add(b)
        if len(grp) > 1:
            groups.append(grp)
    return groups


# ── Background scan ───────────────────────────────────────────────────────────

def _run_scan(event_id: str) -> None:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        return

    folder = Path(row["base_dir"]) / row["folder_name"]
    images = [f for f in folder.iterdir() if f.is_file() and f.suffix in IMAGE_EXTS]

    scores: Dict[str, Any]     = {}
    hashes: Dict[str, Any]     = {}
    rejected: Dict[str, str]   = {}

    for img_path in images:
        try:
            sc = _score_image(img_path)
            scores[img_path.name] = sc
            with Image.open(img_path) as img:
                hashes[img_path.name] = _dhash(img)
        except (UnidentifiedImageError, Exception) as e:
            scores[img_path.name] = {"total": 0, "error": str(e)}

    # Duplicate detection
    dup_groups = _find_duplicate_groups(hashes)
    for grp in dup_groups:
        best = max(grp, key=lambda f: scores.get(f, {}).get("total", 0))
        for f in grp:
            if f != best:
                rejected[f] = "duplicate"

    # Sort candidates by score
    candidates = sorted(
        [f for f in scores if f not in rejected and scores[f].get("total", 0) > 0],
        key=lambda f: scores[f]["total"],
        reverse=True,
    )

    hero    = candidates[0] if candidates else None
    gallery = candidates[1:10]

    # Mark anything outside top 10 AND score < 15 as low quality
    for f in candidates[10:]:
        if scores[f].get("total", 0) < 15:
            rejected[f] = f"low_quality"

    conn = get_conn()
    with conn:
        conn.execute(
            """UPDATE media_events
               SET status=?, image_count=?, scores_json=?, duplicates_json=?,
                   hero_filename=?, gallery_json=?, rejected_json=?, updated_at=?
               WHERE event_id=?""",
            (
                "scored", len(images),
                json.dumps(scores), json.dumps(dup_groups),
                hero, json.dumps(gallery), json.dumps(rejected),
                datetime.utcnow().isoformat(), event_id,
            ),
        )
    conn.close()


# ── Background AI analysis ────────────────────────────────────────────────────

def _prepare_for_vision(img_path: Path, max_px: int = 1024) -> Tuple[str, str]:
    with Image.open(img_path) as img:
        if img.mode not in ("RGB",):
            img = img.convert("RGB")
        if img.width > max_px or img.height > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def _run_ai_analysis(event_id: str) -> None:
    import anthropic

    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        return

    folder   = Path(row["base_dir"]) / row["folder_name"]
    scores   = json.loads(row["scores_json"] or "{}")
    rejected = json.loads(row["rejected_json"] or "{}")
    hero     = row["hero_filename"]

    # Top 5 candidates (hero first, then gallery)
    gallery   = json.loads(row["gallery_json"] or "[]")
    top_names = ([hero] if hero else []) + [g for g in gallery if g != hero]
    top_names = top_names[:5]

    client  = anthropic.Anthropic()
    results: Dict[str, Any] = {}

    PROMPT = (
        "Analyze this image from a Maharat Construction Training Center event. "
        "Reply with JSON only — no prose:\n"
        '{\n'
        '  "subjects": ["list of who/what is in the image"],\n'
        '  "composition_score": <1-10>,\n'
        '  "hero_confidence": <0-100>,\n'
        '  "hero_reason": "one sentence",\n'
        '  "alt_text_en": "concise alt text",\n'
        '  "alt_text_ar": "وصف مختصر بالعربية",\n'
        '  "caption_en": "one sentence caption",\n'
        '  "caption_ar": "تعليق بجملة واحدة",\n'
        '  "tags": ["tag1", "tag2"]\n'
        "}"
    )

    for filename in top_names:
        img_path = folder / filename
        if not img_path.exists():
            continue
        try:
            b64, mtype = _prepare_for_vision(img_path)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": mtype, "data": b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            )
            text  = resp.content[0].text.strip()
            start = text.find("{")
            end   = text.rfind("}") + 1
            results[filename] = json.loads(text[start:end])
        except Exception as e:
            results[filename] = {"error": str(e)}

    # Ranking call — send thumbnails together
    if len(top_names) > 1:
        content: List[Any] = []
        for i, filename in enumerate(top_names):
            img_path = folder / filename
            if not img_path.exists():
                continue
            try:
                b64, mtype = _prepare_for_vision(img_path, max_px=512)
                content.append({"type": "text", "text": f"Image {i + 1}: {filename}"})
                content.append({"type": "image",
                                 "source": {"type": "base64", "media_type": mtype, "data": b64}})
            except Exception:
                pass
        content.append({
            "type": "text",
            "text": (
                f"These {len(top_names)} images are from a Maharat event. "
                "Which should be the hero image? Reply JSON only:\n"
                '{"recommended_hero": "filename", '
                '"ranking": ["file1", "file2"], '
                '"reason": "brief explanation"}'
            ),
        })
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": content}],
            )
            text  = resp.content[0].text.strip()
            start = text.find("{")
            end   = text.rfind("}") + 1
            results["_ranking"] = json.loads(text[start:end])
        except Exception as e:
            results["_ranking"] = {"error": str(e)}

    # Update hero if AI disagrees
    ai_hero = results.get("_ranking", {}).get("recommended_hero")
    if ai_hero and ai_hero in scores and ai_hero not in rejected:
        current_gallery = json.loads(row["gallery_json"] or "[]")
        if hero and hero != ai_hero:
            new_gallery = [ai_hero if g == ai_hero else g for g in current_gallery]
            if hero not in new_gallery:
                new_gallery = [hero] + [g for g in new_gallery if g != ai_hero][:8]
            new_hero = ai_hero
        else:
            new_hero    = hero
            new_gallery = current_gallery
    else:
        new_hero    = hero
        new_gallery = json.loads(row["gallery_json"] or "[]")

    conn = get_conn()
    with conn:
        conn.execute(
            """UPDATE media_events
               SET status=?, hero_filename=?, gallery_json=?, ai_json=?, updated_at=?
               WHERE event_id=?""",
            ("ai_analyzed", new_hero, json.dumps(new_gallery),
             json.dumps(results), datetime.utcnow().isoformat(), event_id),
        )
    conn.close()


# ── Export ────────────────────────────────────────────────────────────────────

def _run_export(event_id: str) -> None:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        return

    folder   = Path(row["base_dir"]) / row["folder_name"]
    meta     = json.loads(row["metadata_json"] or "{}")
    hero     = row["hero_filename"]
    gallery  = json.loads(row["gallery_json"] or "[]")

    slug = meta.get("eventSlug") or re.sub(r"[^\w-]", "-", row["event_name"].lower())[:60]
    out_dir = OUTPUT_MEDIA / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy hero
    if hero:
        src = folder / hero
        if src.exists():
            shutil.copy2(str(src), str(out_dir / f"hero{src.suffix}"))

    # Copy gallery
    for i, fname in enumerate(gallery, start=1):
        src = folder / fname
        if src.exists():
            shutil.copy2(str(src), str(out_dir / f"gallery-{i:02d}{src.suffix}"))

    # Write metadata YAML
    import yaml
    scores = json.loads(row["scores_json"] or "{}")
    hero_score = scores.get(hero, {}).get("total", 0) if hero else 0
    ai         = json.loads(row["ai_json"] or "{}")
    hero_ai    = ai.get(hero, {}) if hero else {}

    metadata = {
        "eventSlug":    meta.get("eventSlug", slug),
        "eventDate":    meta.get("eventDate", row.get("event_date") or ""),
        "eventName":    meta.get("eventName", row["event_name"]),
        "imageRole":    "hero",
        "altTextEn":    meta.get("altTextEn") or hero_ai.get("alt_text_en", ""),
        "altTextAr":    meta.get("altTextAr") or hero_ai.get("alt_text_ar", ""),
        "captionEn":    meta.get("captionEn") or hero_ai.get("caption_en", ""),
        "captionAr":    meta.get("captionAr") or hero_ai.get("caption_ar", ""),
        "qualityScore": round(hero_score),
        "heroFilename": f"hero{Path(hero).suffix}" if hero else "",
        "galleryFiles": [f"gallery-{i:02d}{Path(g).suffix}" for i, g in enumerate(gallery, 1)],
        "tags":         meta.get("tags") or hero_ai.get("tags", []),
        "exportedAt":   datetime.utcnow().isoformat(),
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    conn = get_conn()
    with conn:
        conn.execute(
            "UPDATE media_events SET status='approved', updated_at=? WHERE event_id=?",
            (datetime.utcnow().isoformat(), event_id),
        )
    conn.close()


# ── Image serving ─────────────────────────────────────────────────────────────

def _thumb(abs_path: Path, w: int) -> bytes:
    key = f"{abs_path}:{w}"
    if key in _thumb_cache:
        return _thumb_cache[key]
    with Image.open(str(abs_path)) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        ratio = w / img.width
        h = max(1, int(img.height * ratio))
        resized = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
    if len(_thumb_cache) >= MAX_THUMB_CACHE:
        _thumb_cache.pop(next(iter(_thumb_cache)))
    _thumb_cache[key] = data
    return data


# ── Pydantic models ───────────────────────────────────────────────────────────

class SettingsRequest(BaseModel):
    base_dir: str


class SelectionRequest(BaseModel):
    hero:    Optional[str] = None
    gallery: Optional[List[str]] = None


class MetadataRequest(BaseModel):
    eventSlug:  Optional[str] = None
    eventDate:  Optional[str] = None
    eventName:  Optional[str] = None
    altTextEn:  Optional[str] = None
    altTextAr:  Optional[str] = None
    captionEn:  Optional[str] = None
    captionAr:  Optional[str] = None
    tags:       Optional[List[str]] = None


# ── Helper: row → dict ────────────────────────────────────────────────────────

def _row_to_summary(row: Any) -> Dict[str, Any]:
    gallery = json.loads(row["gallery_json"] or "[]")
    return {
        "event_id":    row["event_id"],
        "folder_name": row["folder_name"],
        "event_name":  row["event_name"],
        "event_date":  row["event_date"],
        "status":      row["status"],
        "image_count": row["image_count"],
        "hero":        row["hero_filename"],
        "gallery_count": len(gallery),
    }


def _row_to_detail(row: Any) -> Dict[str, Any]:
    d = _row_to_summary(row)
    d.update({
        "base_dir":  row["base_dir"],
        "scores":    json.loads(row["scores_json"]    or "{}"),
        "duplicates":json.loads(row["duplicates_json"] or "[]"),
        "gallery":   json.loads(row["gallery_json"]   or "[]"),
        "rejected":  json.loads(row["rejected_json"]  or "{}"),
        "metadata":  json.loads(row["metadata_json"]  or "{}"),
        "ai":        json.loads(row["ai_json"]        or "{}"),
    })
    return d


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/settings")
def get_settings() -> Dict[str, str]:
    return {"base_dir": _get_setting("media_base_dir")}


@router.put("/settings")
def update_settings(body: SettingsRequest) -> Dict[str, str]:
    p = Path(body.base_dir)
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, f"Directory does not exist: {body.base_dir}")
    _set_setting("media_base_dir", str(p))
    return {"base_dir": str(p)}


@router.get("/events")
def list_events() -> List[Dict[str, Any]]:
    base_dir = _get_setting("media_base_dir")
    if not base_dir:
        return []

    base = Path(base_dir)
    if not base.exists():
        raise HTTPException(400, f"Base directory not found: {base_dir}")

    # Discover subfolders
    folders = sorted(
        [f for f in base.iterdir() if f.is_dir() and not f.name.startswith(".")],
        key=lambda f: f.name,
        reverse=True,
    )

    # Upsert each folder into DB (idempotent)
    conn = get_conn()
    result = []
    for folder in folders:
        eid        = _event_id(base_dir, folder.name)
        event_name, event_date = _parse_folder_name(folder.name)
        existing   = conn.execute(
            "SELECT * FROM media_events WHERE event_id=?", (eid,)
        ).fetchone()
        if not existing:
            with conn:
                conn.execute(
                    """INSERT INTO media_events
                       (event_id, folder_name, event_name, event_date, base_dir, status)
                       VALUES (?, ?, ?, ?, ?, 'pending')""",
                    (eid, folder.name, event_name, event_date, base_dir),
                )
            existing = conn.execute(
                "SELECT * FROM media_events WHERE event_id=?", (eid,)
            ).fetchone()
        result.append(_row_to_summary(existing))
    conn.close()
    return result


@router.get("/events/{event_id}")
def get_event(event_id: str) -> Dict[str, Any]:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Event not found")
    return _row_to_detail(row)


@router.post("/events/{event_id}/scan")
def scan_event(event_id: str) -> Dict[str, Any]:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Event not found")
    if row["status"] == "scanning":
        raise HTTPException(409, "Scan already in progress")

    conn = get_conn()
    with conn:
        conn.execute(
            "UPDATE media_events SET status='scanning', updated_at=? WHERE event_id=?",
            (datetime.utcnow().isoformat(), event_id),
        )
    conn.close()

    def _worker():
        lock = _scan_locks.setdefault(event_id, threading.Lock())
        with lock:
            try:
                _run_scan(event_id)
            except Exception as e:
                conn2 = get_conn()
                with conn2:
                    conn2.execute(
                        "UPDATE media_events SET status='error', updated_at=? WHERE event_id=?",
                        (f"scan_error: {e}", event_id),
                    )
                conn2.close()

    threading.Thread(target=_worker, daemon=True).start()
    return {"event_id": event_id, "status": "scanning"}


@router.post("/events/{event_id}/analyze")
def analyze_event(event_id: str) -> Dict[str, Any]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(400, "ANTHROPIC_API_KEY not set")

    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Event not found")
    if row["status"] not in ("scored", "ai_analyzed", "approved"):
        raise HTTPException(409, "Run quality scan first")

    conn = get_conn()
    with conn:
        conn.execute(
            "UPDATE media_events SET status='analyzing', updated_at=? WHERE event_id=?",
            (datetime.utcnow().isoformat(), event_id),
        )
    conn.close()

    def _worker():
        try:
            _run_ai_analysis(event_id)
        except Exception as e:
            conn2 = get_conn()
            with conn2:
                conn2.execute(
                    "UPDATE media_events SET status='scored', updated_at=? WHERE event_id=?",
                    (datetime.utcnow().isoformat(), event_id),
                )
            conn2.close()

    threading.Thread(target=_worker, daemon=True).start()
    return {"event_id": event_id, "status": "analyzing"}


@router.put("/events/{event_id}/selection")
def update_selection(event_id: str, body: SelectionRequest) -> Dict[str, Any]:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Event not found")

    hero    = body.hero    if body.hero    is not None else row["hero_filename"]
    gallery = body.gallery if body.gallery is not None else json.loads(row["gallery_json"] or "[]")

    conn = get_conn()
    with conn:
        conn.execute(
            "UPDATE media_events SET hero_filename=?, gallery_json=?, updated_at=? WHERE event_id=?",
            (hero, json.dumps(gallery), datetime.utcnow().isoformat(), event_id),
        )
    conn.close()
    return {"ok": True, "hero": hero, "gallery": gallery}


@router.put("/events/{event_id}/metadata")
def update_metadata(event_id: str, body: MetadataRequest) -> Dict[str, Any]:
    conn = get_conn()
    row  = conn.execute("SELECT metadata_json FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Event not found")

    existing = json.loads(row["metadata_json"] or "{}")
    updates  = {k: v for k, v in body.dict().items() if v is not None}
    existing.update(updates)

    conn = get_conn()
    with conn:
        conn.execute(
            "UPDATE media_events SET metadata_json=?, updated_at=? WHERE event_id=?",
            (json.dumps(existing), datetime.utcnow().isoformat(), event_id),
        )
    conn.close()
    return {"ok": True, "metadata": existing}


@router.post("/events/{event_id}/approve")
def approve_event(event_id: str) -> Dict[str, Any]:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM media_events WHERE event_id=?", (event_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Event not found")
    if not row["hero_filename"]:
        raise HTTPException(400, "Set a hero image before approving")

    threading.Thread(target=_run_export, args=(event_id,), daemon=True).start()
    return {"event_id": event_id, "status": "exporting"}


@router.get("/image")
def serve_image(rel: str, w: Optional[int] = None) -> Response:
    base_dir = _get_setting("media_base_dir")
    if not base_dir:
        raise HTTPException(400, "Media base directory not configured")

    abs_path = (Path(base_dir) / rel).resolve()
    base_abs = Path(base_dir).resolve()

    # Security: must stay inside base_dir
    try:
        abs_path.relative_to(base_abs)
    except ValueError:
        raise HTTPException(403, "Path outside base directory")

    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(404, "Image not found")

    if w:
        try:
            data = _thumb(abs_path, w)
            return Response(content=data, media_type="image/jpeg")
        except Exception:
            pass  # fall through to full image

    return FileResponse(str(abs_path))
