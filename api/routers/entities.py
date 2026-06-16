import ast
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

CONFIG_PATH = ROOT / "config" / "entities.yaml"
ENTITY_TYPES = ["organizations", "programs", "locations", "credentials", "people"]


# ── YAML helpers ──────────────────────────────────────────────────────────────

def _load() -> Dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _save(data: Dict[str, Any]) -> None:
    from api.db import get_conn
    with open(CONFIG_PATH) as f:
        old = f.read()
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO config_versions (config_name, content, saved_at, note) VALUES (?, ?, ?, ?)",
            ("entities.yaml", old, datetime.utcnow().isoformat(), "entity-manager edit"),
        )
    conn.close()
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _normalise_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {"canonical": str(item), "aliases": []}
    return {
        "canonical":   item.get("canonical", ""),
        "aliases":     item.get("aliases") or [],
        "title":       item.get("title"),
        "affiliation": item.get("affiliation"),
    }


# ── Duplicate detection ───────────────────────────────────────────────────────

def _find_duplicates(entities: Dict[str, List]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, str]] = {}
    dupes: List[Dict[str, Any]] = []
    for etype, items in entities.items():
        if not isinstance(items, list):
            continue
        for item in items:
            canonical = item.get("canonical", "") if isinstance(item, dict) else str(item)
            aliases   = (item.get("aliases") or []) if isinstance(item, dict) else []
            for name in [canonical] + aliases:
                if not name:
                    continue
                key = name.lower().strip()
                entry = {"type": etype, "canonical": canonical}
                if key in seen and seen[key]["canonical"] != canonical:
                    dupes.append({"alias": name, "entity_1": seen[key], "entity_2": entry})
                else:
                    seen[key] = entry
    return dupes


# ── Qdrant mention counts (no embedding models needed) ────────────────────────

def _parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        s = value.strip()
        if s in ("", "None", "null", "[]"):
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except Exception:
            pass
        return [s]
    return []


def _get_counts() -> Dict[str, int]:
    try:
        from services.config_service import load_qdrant_config, make_client
        cfg    = load_qdrant_config()
        client = make_client(cfg)
        pcfg   = cfg.get("collections", {}).get("primary", {})
        coll   = pcfg.get("live_alias") or pcfg.get("name", "maharat_content_live")

        entity_fields = [
            "entities_organizations", "entities_programs",
            "entities_locations", "entities_credentials", "entities_people",
        ]
        bucket: Dict[str, set] = {}
        offset = None
        while True:
            points, offset = client.scroll(coll, limit=200, with_payload=True, offset=offset)
            for p in points:
                payload = p.payload or {}
                slug    = str(payload.get("slug", ""))
                for field in entity_fields:
                    for name in _parse_list(payload.get(field, [])):
                        key = name.lower()
                        bucket.setdefault(key, set())
                        if slug:
                            bucket[key].add(slug)
            if offset is None:
                break
        return {k: len(v) for k, v in bucket.items()}
    except Exception:
        return {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class EntityItem(BaseModel):
    canonical:   str
    aliases:     List[str] = []
    title:       Optional[str] = None
    affiliation: Optional[str] = None


class EntityBody(BaseModel):
    entity: EntityItem


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def list_entities() -> Dict[str, Any]:
    data     = _load()
    raw      = data.get("entities", {})
    entities = {etype: [_normalise_item(i) for i in (raw.get(etype) or [])] for etype in ENTITY_TYPES}
    return {
        "entities":     entities,
        "duplicates":   _find_duplicates(entities),
        "entity_types": ENTITY_TYPES,
    }


@router.get("/counts")
def get_counts() -> Dict[str, int]:
    return _get_counts()


@router.post("/{entity_type}")
def add_entity(entity_type: str, body: EntityBody) -> Dict[str, Any]:
    if entity_type not in ENTITY_TYPES:
        raise HTTPException(400, f"Unknown entity type: {entity_type}")
    data  = _load()
    items = (data.setdefault("entities", {}).setdefault(entity_type, []) or [])
    for item in items:
        if isinstance(item, dict) and item.get("canonical", "").lower() == body.entity.canonical.lower():
            raise HTTPException(409, f"Canonical name already exists: {body.entity.canonical}")
    new_item: Dict[str, Any] = {"canonical": body.entity.canonical, "aliases": body.entity.aliases or []}
    if body.entity.title:
        new_item["title"] = body.entity.title
    if body.entity.affiliation:
        new_item["affiliation"] = body.entity.affiliation
    items.append(new_item)
    data["entities"][entity_type] = items
    _save(data)
    return {"ok": True, "entity_type": entity_type, "canonical": body.entity.canonical}


@router.put("/{entity_type}/{index}")
def update_entity(entity_type: str, index: int, body: EntityBody) -> Dict[str, Any]:
    if entity_type not in ENTITY_TYPES:
        raise HTTPException(400, f"Unknown entity type: {entity_type}")
    data  = _load()
    items = data.get("entities", {}).get(entity_type) or []
    if index < 0 or index >= len(items):
        raise HTTPException(404, "Index out of range")
    updated: Dict[str, Any] = {"canonical": body.entity.canonical, "aliases": body.entity.aliases or []}
    if body.entity.title:
        updated["title"] = body.entity.title
    if body.entity.affiliation:
        updated["affiliation"] = body.entity.affiliation
    items[index] = updated
    data["entities"][entity_type] = items
    _save(data)
    return {"ok": True, "entity_type": entity_type, "index": index}


@router.delete("/{entity_type}/{index}")
def delete_entity(entity_type: str, index: int) -> Dict[str, Any]:
    if entity_type not in ENTITY_TYPES:
        raise HTTPException(400, f"Unknown entity type: {entity_type}")
    data  = _load()
    items = data.get("entities", {}).get(entity_type) or []
    if index < 0 or index >= len(items):
        raise HTTPException(404, "Index out of range")
    removed = items.pop(index)
    data["entities"][entity_type] = items
    _save(data)
    canonical = removed.get("canonical", "") if isinstance(removed, dict) else str(removed)
    return {"ok": True, "entity_type": entity_type, "removed": canonical}
