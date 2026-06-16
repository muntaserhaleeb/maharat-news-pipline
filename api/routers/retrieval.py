import ast
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from api.models.schemas import SearchRequest

router = APIRouter()

ROOT = Path(__file__).resolve().parent.parent.parent

# Lazy service cache — fastembed loads models on first use (~30 s cold start)
_svc_cache: Dict[str, Any] = {}
_svc_loading: Dict[str, bool] = {}


def _get_service(collection_key: str):
    if collection_key not in _svc_cache:
        _svc_loading[collection_key] = True
        try:
            from services.retrieval_service import RetrievalService
            _svc_cache[collection_key] = RetrievalService.from_config(collection_key=collection_key)
        finally:
            _svc_loading[collection_key] = False
    return _svc_cache[collection_key]


def _parse_list(value: Any) -> List[str]:
    """Parse a field that may be a list or a Python-repr string of a list."""
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ("", "None", "null", "[]"):
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None]
        except Exception:
            pass
        return [stripped]
    return []


def _str_or_none(value: Any) -> Optional[str]:
    if value is None or str(value).strip() in ("None", "null", ""):
        return None
    return str(value)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_result(r: Any, rank: int) -> Dict[str, Any]:
    p = r.payload or {}
    return {
        "rank":          rank,
        "id":            str(r.id),
        "score":         round(r.score, 4),
        "chunk_type":    _str_or_none(p.get("chunk_type")),
        "chunk_index":   _int_or_none(p.get("chunk_index")),
        "title":         _str_or_none(p.get("title")) or "",
        "slug":          _str_or_none(p.get("slug"))  or "",
        "text":          _str_or_none(p.get("chunk_text")) or "",
        "word_count":    _int_or_none(p.get("word_count")),
        "heading_path":  _str_or_none(p.get("heading_path")),
        "date":          _str_or_none(p.get("date")),
        "year":          _int_or_none(p.get("year")),
        "quarter":       _str_or_none(p.get("quarter")),
        "month":         _int_or_none(p.get("month")),
        "category":      _str_or_none(p.get("category")) or "",
        "tags":          _parse_list(p.get("tags")),
        "source_document": _str_or_none(p.get("source_document")),
        "language":      _str_or_none(p.get("language")),
        "status":        _str_or_none(p.get("status")),
        "entities": {
            "organizations": _parse_list(p.get("entities_organizations")),
            "programs":      _parse_list(p.get("entities_programs")),
            "locations":     _parse_list(p.get("entities_locations")),
            "credentials":   _parse_list(p.get("entities_credentials")),
            "people":        _parse_list(p.get("entities_people")),
        },
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/status")
def get_status() -> Dict[str, Any]:
    """Return which collection services are initialised (models loaded)."""
    return {
        "primary":   {"ready": "primary"   in _svc_cache, "loading": _svc_loading.get("primary",   False)},
        "knowledge": {"ready": "knowledge" in _svc_cache, "loading": _svc_loading.get("knowledge", False)},
        "note": "First search per collection loads fastembed models (~30 s).",
    }


@router.get("/facets")
def get_facets() -> Dict[str, Any]:
    """Return available filter options so the UI can populate dropdowns."""
    try:
        from services.config_service import load_taxonomy
        tax = load_taxonomy()
        categories: List[str] = [
            r["name"]
            for r in tax.get("category_rules", {}).get("ordered_rules", [])
        ]
    except Exception:
        categories = []

    return {
        "collections": [
            {"key": "primary",   "label": "News (primary)",  "description": "Ingested news posts"},
            {"key": "knowledge", "label": "Knowledge base",  "description": "Curated knowledge documents"},
        ],
        "categories":  categories,
        "chunk_types": ["body", "summary", "title", "quote", "caption"],
        "quarters":    ["Q1", "Q2", "Q3", "Q4"],
        "years":       list(range(2022, 2027)),
        "languages":   ["en", "ar"],
    }


@router.post("/search")
def search(body: SearchRequest) -> Dict[str, Any]:
    valid = ("primary", "knowledge")
    if body.collection not in valid:
        raise HTTPException(400, f"collection must be one of {valid}")

    t0 = time.time()
    try:
        svc = _get_service(body.collection)
    except Exception as exc:
        raise HTTPException(503, f"Retrieval service unavailable: {exc}")

    try:
        query_filter = svc.build_filter(
            category=body.category,
            year=body.year,
            quarter=body.quarter,
            chunk_type=body.chunk_type,
            language=body.language,
        )
        raw = svc.search(
            query_text=body.query,
            limit=body.limit,
            score_threshold=body.score_threshold,
            query_filter=query_filter,
        )
    except Exception as exc:
        raise HTTPException(500, f"Search failed: {exc}")

    elapsed_ms = round((time.time() - t0) * 1000)
    results = [_format_result(r, i + 1) for i, r in enumerate(raw)]

    return {
        "query":      body.query,
        "collection": body.collection,
        "total":      len(results),
        "elapsed_ms": elapsed_ms,
        "results":    results,
    }
