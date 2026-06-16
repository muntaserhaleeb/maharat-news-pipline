from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.db import get_conn
from api.models.schemas import ConfigSaveRequest

router = APIRouter()

ROOT       = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT / "config"

AVAILABLE_CONFIGS = [
    "entities.yaml",
    "taxonomy.yaml",
    "generation.yaml",
    "editorial_style.yaml",
    "chunking.yaml",
    "knowledge_chunking.yaml",
    "qdrant.yaml",
]

# Expected top-level keys per config — used for structural warnings
_EXPECTED_KEYS: Dict[str, List[str]] = {
    "entities.yaml":           ["entities"],
    "taxonomy.yaml":           ["category_rules", "tags"],
    "generation.yaml":         ["generation"],
    "chunking.yaml":           ["chunking"],
    "knowledge_chunking.yaml": ["chunking"],
    "qdrant.yaml":             ["qdrant", "collections"],
}


class ValidateRequest(BaseModel):
    content: str
    name: Optional[str] = None


@router.post("/validate")
def validate_config(body: ValidateRequest) -> Dict[str, Any]:
    """Validate YAML syntax and structure without saving."""
    errors: List[str] = []
    warnings: List[str] = []

    # 1 — YAML syntax
    try:
        parsed = yaml.safe_load(body.content)
    except yaml.YAMLError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}

    if parsed is None:
        errors.append("File is empty or contains only comments.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if not isinstance(parsed, dict):
        errors.append(f"Expected a YAML mapping at the top level, got {type(parsed).__name__}.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # 2 — structural check when a known config name is supplied
    if body.name and body.name in _EXPECTED_KEYS:
        expected = _EXPECTED_KEYS[body.name]
        missing = [k for k in expected if k not in parsed]
        if missing:
            warnings.append(
                f"Expected top-level key(s) not found: {', '.join(missing)}"
            )

    return {"valid": True, "errors": errors, "warnings": warnings}


@router.get("")
def list_configs() -> List[Dict[str, Any]]:
    result = []
    for name in AVAILABLE_CONFIGS:
        path = CONFIG_DIR / name
        result.append({
            "name":   name,
            "exists": path.exists(),
            "size":   path.stat().st_size if path.exists() else 0,
        })
    return result


@router.get("/{name}")
def get_config(name: str) -> Dict[str, Any]:
    if name not in AVAILABLE_CONFIGS:
        raise HTTPException(404, f"Config '{name}' not managed by this UI")
    path = CONFIG_DIR / name
    if not path.exists():
        raise HTTPException(404, f"File '{name}' not found on disk")
    content = path.read_text(encoding="utf-8")
    try:
        parsed: Any = yaml.safe_load(content)
    except yaml.YAMLError:
        parsed = None
    return {"name": name, "content": content, "parsed": parsed}


@router.put("/{name}")
def save_config(name: str, body: ConfigSaveRequest) -> Dict[str, str]:
    if name not in AVAILABLE_CONFIGS:
        raise HTTPException(404, f"Config '{name}' not managed by this UI")
    try:
        yaml.safe_load(body.content)
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"Invalid YAML: {exc}")
    path = CONFIG_DIR / name
    old_content = path.read_text(encoding="utf-8") if path.exists() else ""
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO config_versions (config_name, content, saved_at, note) VALUES (?, ?, ?, ?)",
            (name, old_content, datetime.utcnow().isoformat(), body.note),
        )
    conn.close()
    path.write_text(body.content, encoding="utf-8")
    return {"status": "saved", "name": name}


@router.get("/{name}/history")
def get_history(name: str) -> List[Dict[str, Any]]:
    if name not in AVAILABLE_CONFIGS:
        raise HTTPException(404, f"Config '{name}' not managed by this UI")
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, config_name, saved_at, note
           FROM config_versions
           WHERE config_name = ?
           ORDER BY id DESC
           LIMIT 30""",
        (name,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/{name}/rollback/{version_id}")
def rollback_config(name: str, version_id: int) -> Dict[str, Any]:
    if name not in AVAILABLE_CONFIGS:
        raise HTTPException(404, f"Config '{name}' not managed by this UI")
    conn = get_conn()
    row = conn.execute(
        "SELECT content FROM config_versions WHERE id = ? AND config_name = ?",
        (version_id, name),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Version not found")
    path = CONFIG_DIR / name
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO config_versions (config_name, content, saved_at, note) VALUES (?, ?, ?, ?)",
            (name, current, datetime.utcnow().isoformat(), f"pre-rollback snapshot (restoring v{version_id})"),
        )
    conn.close()
    path.write_text(row["content"], encoding="utf-8")
    return {"status": "rolled_back", "name": name, "restored_version_id": version_id}
