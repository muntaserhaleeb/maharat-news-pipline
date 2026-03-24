"""
Config service — loads all YAML config files.
Single source of truth for configuration across services and pipelines.
scripts/config.py re-exports from here for backwards compatibility with one-off scripts.
"""

import yaml
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_qdrant_config() -> dict:
    return _load("qdrant.yaml")


def load_chunking_config() -> dict:
    return _load("chunking.yaml")


def load_taxonomy() -> dict:
    return _load("taxonomy.yaml")


def load_generation_config() -> dict:
    return _load("generation.yaml")


def load_entities_config() -> dict:
    return _load("entities.yaml")


def load_editorial_style_config() -> dict:
    return _load("editorial_style.yaml")


def get_collection_cfg(qdrant_cfg: dict = None) -> dict:
    if qdrant_cfg is None:
        qdrant_cfg = load_qdrant_config()
    return qdrant_cfg["collections"]["primary"]


def get_all_valid_tags(taxonomy: dict = None) -> set:
    if taxonomy is None:
        taxonomy = load_taxonomy()
    valid = set()
    for group_values in taxonomy.get("tags", {}).values():
        if isinstance(group_values, list):
            for v in group_values:
                valid.add(v)
    return valid


def get_valid_categories(taxonomy: dict = None) -> list:
    if taxonomy is None:
        taxonomy = load_taxonomy()
    return taxonomy.get("categories", [])


def get_taxonomy_rules(taxonomy: dict = None) -> dict:
    if taxonomy is None:
        taxonomy = load_taxonomy()
    return taxonomy.get("rules", {})


def make_client(qdrant_cfg: dict = None):
    """Return a QdrantClient using HTTP (url) or embedded local (path) mode."""
    from qdrant_client import QdrantClient

    if qdrant_cfg is None:
        qdrant_cfg = load_qdrant_config()

    qcfg = qdrant_cfg["qdrant"]
    url  = qcfg.get("url")
    path = qcfg.get("path")

    if url:
        return QdrantClient(url=url, api_key=qcfg.get("api_key"))

    if path:
        resolved = ROOT / path if not Path(path).is_absolute() else Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(resolved))

    raise ValueError("qdrant config must have either 'url' or 'path' set")
