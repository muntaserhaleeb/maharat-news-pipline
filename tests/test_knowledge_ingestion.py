#!/usr/bin/env python3
"""
Knowledge ingestion validation tests.

Tests validation logic, chunking, and the full pipeline (dry-run).
Does not require a running Qdrant instance — dry_run=True skips all network calls.

Usage:
    python tests/test_knowledge_ingestion.py
    python tests/test_knowledge_ingestion.py --verbose
"""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.knowledge_ingest_pipeline import (
    KnowledgeIngestPipeline,
    _build_knowledge_payload,
    _map_chunk_type,
    _validate_front,
)
from services.chunk_service import make_chunks, parse_markdown
from services.config_service import load_knowledge_chunking_config, load_qdrant_config
from services.embedding_service import EmbeddingService
from services.entity_service import EntityService
from services.qdrant_service import QdrantService


# ── sample data ────────────────────────────────────────────────────────────────

_GOOD_FRONT = {
    "title":          "Maharat Overview",
    "slug":           "maharat-overview",
    "knowledge_type": "institutional_profile",
    "status":         "approved",
    "published":      True,
    "language":       "en",
    "priority":       "high",
    "memory_layer":   "knowledge",
}

_SAMPLE_MD = """\
---
title: "Maharat Construction Training Center"
slug: "maharat-overview-test"
knowledge_type: "institutional_profile"
status: "approved"
published: true
language: "en"
priority: "high"
memory_layer: "knowledge"
---

Maharat is a leading construction training center in Saudi Arabia.

## Training Programs

Maharat offers technical training across welding, pipefitting, and electrical trades.

## FAQ

### What is Maharat?
Maharat is a national construction training institution.

## Key Terms

**Competency-based training** — a methodology that evaluates learners against defined skills.
"""


# ── test helpers ───────────────────────────────────────────────────────────────

def _ok(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


# ── tests ──────────────────────────────────────────────────────────────────────

def test_validate_front_valid() -> bool:
    errors = _validate_front(_GOOD_FRONT)
    return _ok("valid front matter passes", errors == [], str(errors))


def test_validate_front_missing_field() -> bool:
    bad = {k: v for k, v in _GOOD_FRONT.items() if k != "knowledge_type"}
    errors = _validate_front(bad)
    return _ok("missing field is caught", any("knowledge_type" in e for e in errors))


def test_validate_front_wrong_memory_layer() -> bool:
    bad = {**_GOOD_FRONT, "memory_layer": "news"}
    errors = _validate_front(bad)
    return _ok("wrong memory_layer is caught", any("memory_layer" in e for e in errors))


def test_chunk_type_mapping() -> bool:
    results = [
        _map_chunk_type("summary", "Maharat Overview") == "overview",
        _map_chunk_type("body", "FAQ") == "faq",
        _map_chunk_type("body", "Frequently Asked Questions") == "faq",
        _map_chunk_type("body", "Key Terms") == "definition",
        _map_chunk_type("body", "Training Programs") == "section",
    ]
    return _ok("chunk type mapping", all(results), f"{results}")


def test_parse_and_chunk_sample(tmp_dir: Path) -> bool:
    md_path = tmp_dir / "sample.md"
    md_path.write_text(_SAMPLE_MD, encoding="utf-8")
    front, body = parse_markdown(md_path)
    if not _ok("parse front matter", bool(front), str(front)):
        return False
    chunks = make_chunks(front, body, max_tokens=585, overlap_tokens=65)
    return _ok("produces chunks", len(chunks) > 0, f"{len(chunks)} chunks")


def test_payload_structure(tmp_dir: Path) -> bool:
    md_path = tmp_dir / "payload_test.md"
    md_path.write_text(_SAMPLE_MD, encoding="utf-8")
    front, body = parse_markdown(md_path)
    chunks = make_chunks(front, body, max_tokens=585, overlap_tokens=65)
    payload = _build_knowledge_payload(front, chunks[0], {}, md_path)

    required_keys = (
        "doc_id", "chunk_id", "title", "slug", "knowledge_type",
        "section", "chunk_index", "chunk_type", "chunk_text",
        "priority", "language", "status", "published", "memory_layer",
        "source_file", "word_count",
        "entities_organizations", "entities_programs", "entities_locations",
        "entities_credentials", "entities_people", "ingested_at",
    )
    missing = [k for k in required_keys if k not in payload]
    return _ok("all payload fields present", not missing, f"missing={missing}")


def test_real_files_dry_run() -> bool:
    """Run the pipeline in dry-run mode against the real data/knowledge/ directory."""
    knowledge_dir = ROOT / "data" / "knowledge"
    if not knowledge_dir.exists():
        print("  [SKIP] test_real_files_dry_run — data/knowledge/ not found")
        return True

    qdrant_cfg = load_qdrant_config()
    chunk_cfg  = load_knowledge_chunking_config()
    col_cfg    = qdrant_cfg["collections"]["knowledge"]

    pipeline = KnowledgeIngestPipeline(
        qdrant_service=QdrantService.from_config(qdrant_cfg, collection_key="knowledge"),
        embedding_service=EmbeddingService.from_config(col_cfg),
        entity_service=EntityService.from_config(),
        chunk_cfg=chunk_cfg,
        knowledge_dir=knowledge_dir,
    )
    summary = pipeline.run(dry_run=True)
    ok = (
        summary["files_found"] > 0
        and summary["files_valid"] > 0
        and summary["chunks_produced"] > 0
    )
    return _ok(
        "dry-run over real knowledge files",
        ok,
        f"found={summary['files_found']} valid={summary['files_valid']} "
        f"failed={summary['files_failed']} chunks={summary['chunks_produced']}",
    )


# ── runner ─────────────────────────────────────────────────────────────────────

def run_tests(verbose: bool = False) -> bool:
    print("\nKnowledge ingestion tests\n" + "─" * 50)
    passed = failed = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        unit_tests = [
            test_validate_front_valid,
            test_validate_front_missing_field,
            test_validate_front_wrong_memory_layer,
            test_chunk_type_mapping,
            lambda: test_parse_and_chunk_sample(tmp_dir),
            lambda: test_payload_structure(tmp_dir),
        ]
        integration_tests = [
            test_real_files_dry_run,
        ]

        for fn in unit_tests:
            if fn():
                passed += 1
            else:
                failed += 1

        print()
        for fn in integration_tests:
            if fn():
                passed += 1
            else:
                failed += 1

    print(f"\n{'─' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Knowledge ingestion tests")
    parser.add_argument("--verbose", action="store_true")
    args   = parser.parse_args()
    ok     = run_tests(verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
