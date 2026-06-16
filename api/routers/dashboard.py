from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter()

ROOT         = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR   = ROOT / "config"
POSTS_DIR    = ROOT / "data" / "posts"
IMAGES_DIR   = ROOT / "data" / "images"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

MANAGED_CONFIGS = [
    "entities.yaml",
    "taxonomy.yaml",
    "generation.yaml",
    "editorial_style.yaml",
    "chunking.yaml",
    "knowledge_chunking.yaml",
    "qdrant.yaml",
]


def _config_status() -> List[Dict[str, Any]]:
    result = []
    for name in MANAGED_CONFIGS:
        path = CONFIG_DIR / name
        result.append({
            "name":   name,
            "exists": path.exists(),
            "size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
        })
    return result


def _data_counts() -> Dict[str, int]:
    posts  = len(list(POSTS_DIR.glob("*.md")))   if POSTS_DIR.exists()    else 0
    images = len([f for f in IMAGES_DIR.iterdir() if f.is_file()]) if IMAGES_DIR.exists() else 0
    knowledge = len(list(KNOWLEDGE_DIR.rglob("*.md"))) if KNOWLEDGE_DIR.exists() else 0
    return {"posts": posts, "images": images, "knowledge_docs": knowledge}


@router.get("/status")
def get_status() -> Dict[str, Any]:
    counts = _data_counts()
    configs = _config_status()
    missing = [c["name"] for c in configs if not c["exists"]]
    return {
        "phase":      "1",
        "data":       counts,
        "configs": {
            "managed":  len(configs),
            "present":  sum(1 for c in configs if c["exists"]),
            "missing":  missing,
            "files":    configs,
        },
    }
