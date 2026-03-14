#!/usr/bin/env python3
"""
Load and expose qdrant.yaml, chunking.yaml, and taxonomy.yaml configs.
"""

import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _load(name):
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_qdrant_config():
    return _load("qdrant.yaml")


def load_chunking_config():
    return _load("chunking.yaml")


def load_taxonomy():
    return _load("taxonomy.yaml")


def get_collection_cfg(qdrant_cfg=None):
    """Return the primary collection config block."""
    if qdrant_cfg is None:
        qdrant_cfg = load_qdrant_config()
    return qdrant_cfg["collections"]["primary"]


def get_all_valid_tags(taxonomy=None):
    """Return a flat set of all allowed tag values from taxonomy.yaml."""
    if taxonomy is None:
        taxonomy = load_taxonomy()
    tags_section = taxonomy.get("tags", {})
    valid = set()
    for group_values in tags_section.values():
        if isinstance(group_values, list):
            for v in group_values:
                valid.add(v)
    return valid


def get_valid_categories(taxonomy=None):
    """Return list of allowed category names."""
    if taxonomy is None:
        taxonomy = load_taxonomy()
    return taxonomy.get("categories", [])


def get_taxonomy_rules(taxonomy=None):
    if taxonomy is None:
        taxonomy = load_taxonomy()
    return taxonomy.get("rules", {})
