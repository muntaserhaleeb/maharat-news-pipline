import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

router = APIRouter()

TAXONOMY_PATH = ROOT / "config" / "taxonomy.yaml"


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _load() -> Dict[str, Any]:
    with open(TAXONOMY_PATH) as f:
        return yaml.safe_load(f) or {}


def _save(data: Dict[str, Any]) -> None:
    from api.db import get_conn
    with open(TAXONOMY_PATH) as f:
        old = f.read()
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO config_versions (config_name, content, saved_at, note) VALUES (?, ?, ?, ?)",
            ("taxonomy.yaml", old, datetime.utcnow().isoformat(), "taxonomy-manager edit"),
        )
    conn.close()
    with open(TAXONOMY_PATH, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _post_counts() -> Dict[str, int]:
    posts_path = ROOT / "data" / "posts.json"
    if not posts_path.exists():
        return {}
    try:
        with open(posts_path) as f:
            raw = json.load(f)
        posts = raw.get("posts", raw) if isinstance(raw, dict) else raw
        counts: Dict[str, int] = {}
        for p in posts:
            cat = (p.get("category") or "General").strip() or "General"
            counts[cat] = counts.get(cat, 0) + 1
        return counts
    except Exception:
        return {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class CategoryRule(BaseModel):
    name:     str
    keywords: List[str]


class SaveCategoriesRequest(BaseModel):
    default:       str
    ordered_rules: List[CategoryRule]


class TagActionRequest(BaseModel):
    tag: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def get_taxonomy() -> Dict[str, Any]:
    data      = _load()
    cat_cfg   = data.get("category_rules", {})
    ordered   = cat_cfg.get("ordered_rules") or []
    counts    = _post_counts()
    default   = cat_cfg.get("default", "General")

    categories = [
        {
            "name":       r.get("name", ""),
            "keywords":   r.get("keywords") or [],
            "post_count": counts.get(r.get("name", ""), 0),
        }
        for r in ordered
    ]

    raw_tags = data.get("tags", {})
    tag_groups = {k: v for k, v in raw_tags.items() if isinstance(v, list)}

    return {
        "categories": {
            "default":       default,
            "default_count": counts.get(default, 0),
            "ordered_rules": categories,
        },
        "tags":              tag_groups,
        "total_categories":  len(categories),
        "total_tags":        sum(len(v) for v in tag_groups.values()),
    }


@router.put("/categories")
def save_categories(body: SaveCategoriesRequest) -> Dict[str, Any]:
    data = _load()
    data.setdefault("category_rules", {})
    data["category_rules"]["default"] = body.default
    data["category_rules"]["ordered_rules"] = [
        {"name": r.name, "keywords": r.keywords} for r in body.ordered_rules
    ]
    _save(data)
    return {"ok": True, "total": len(body.ordered_rules)}


@router.post("/tags/{group}")
def add_tag(group: str, body: TagActionRequest) -> Dict[str, Any]:
    tag  = body.tag.strip()
    if not tag:
        raise HTTPException(400, "tag must not be empty")
    data = _load()
    tags = data.setdefault("tags", {})
    grp  = tags.setdefault(group, []) or []
    if tag in grp:
        raise HTTPException(409, f"'{tag}' already exists in group '{group}'")
    grp.append(tag)
    tags[group] = grp
    _save(data)
    return {"ok": True, "group": group, "tag": tag}


@router.post("/tags/{group}/remove")
def remove_tag(group: str, body: TagActionRequest) -> Dict[str, Any]:
    tag  = body.tag.strip()
    data = _load()
    tags = data.get("tags", {})
    grp  = tags.get(group) or []
    if tag not in grp:
        raise HTTPException(404, f"'{tag}' not found in group '{group}'")
    grp.remove(tag)
    tags[group] = grp
    _save(data)
    return {"ok": True, "group": group, "removed": tag}
