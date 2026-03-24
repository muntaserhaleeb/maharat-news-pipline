#!/usr/bin/env python3
"""
Config loader — thin re-export wrapper.
All logic lives in services/config_service.py.
This file exists for backwards compatibility with one-off scripts in scripts/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.config_service import (  # noqa: F401  (re-export)
    _load,
    get_all_valid_tags,
    get_collection_cfg,
    get_taxonomy_rules,
    get_valid_categories,
    load_chunking_config,
    load_generation_config,
    load_qdrant_config,
    load_taxonomy,
    make_client,
)
