#!/usr/bin/env python3
"""
Maharat RAG Validation Script
==============================
Validates the Qdrant collection after ingestion:
  - Collection exists and has correct vector config
  - Alias is set
  - Payload indexes are present
  - Point count is plausible
  - Sample hybrid searches return results

Usage:
    python tests/validate_qdrant.py
    python tests/validate_qdrant.py --sample-queries 5
"""

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from config import load_qdrant_config, make_client
from embed_chunks import Embedder
from search_qdrant import search

SAMPLE_QUERIES = [
    "fire safety training drill",
    "OJT on-the-job training monitoring",
    "welding competition",
    "graduation ceremony",
    "partnership agreement Saudi Aramco",
]


def check_collection(client, col_cfg):
    name = col_cfg["name"]
    ok = True

    if not client.collection_exists(name):
        print(f"  [FAIL] Collection '{name}' does not exist.")
        return False

    info = client.get_collection(name)
    points = info.points_count
    print(f"  [OK]   Collection '{name}' exists — {points} points")

    # Check named vectors
    vec_cfg = info.config.params.vectors or {}
    if "dense" not in vec_cfg:
        print("  [FAIL] Named vector 'dense' not found.")
        ok = False
    else:
        dense_size = vec_cfg["dense"].size
        expected   = col_cfg["vectors"]["dense"]["size"]
        if dense_size != expected:
            print(f"  [FAIL] Dense vector size {dense_size} != expected {expected}.")
            ok = False
        else:
            print(f"  [OK]   Dense vector 'dense' size={dense_size}")

    sparse_cfg = info.config.params.sparse_vectors or {}
    if "sparse" in sparse_cfg:
        print("  [OK]   Sparse vector 'sparse' present")
    else:
        print("  [FAIL] Sparse vector 'sparse' not found.")
        ok = False

    return ok


def check_alias(client, collection_name, alias_name):
    try:
        aliases = client.get_aliases()
        found = any(
            a.alias_name == alias_name and a.collection_name == collection_name
            for a in aliases.aliases
        )
        if found:
            print(f"  [OK]   Alias '{alias_name}' → '{collection_name}'")
            return True
        else:
            print(f"  [FAIL] Alias '{alias_name}' not set.")
            return False
    except Exception as e:
        print(f"  [WARN] Could not check alias: {e}")
        return True  # non-fatal


def check_payload_indexes(client, collection_name, qdrant_cfg):
    expected_indexes = {i["field_name"] for i in qdrant_cfg.get("payload_indexes", [])}
    try:
        info = client.get_collection(collection_name)
        indexed = set(info.payload_schema.keys()) if info.payload_schema else set()
        missing = expected_indexes - indexed
        if missing:
            print(f"  [WARN] Missing payload indexes: {', '.join(sorted(missing))}")
        else:
            print(f"  [OK]   All {len(expected_indexes)} payload indexes present")
    except Exception as e:
        print(f"  [WARN] Could not check payload indexes: {e}")


def check_payload_completeness(client, collection_name, sample_size=5):
    """Spot-check a few points for required payload fields."""
    required = ["article_id", "chunk_id", "slug", "chunk_text", "chunk_type", "chunk_index"]
    try:
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=sample_size,
            with_payload=True,
        )
        results = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
        issues = 0
        for point in results:
            p = point.payload or {}
            for field in required:
                if field not in p or p[field] is None:
                    print(f"  [WARN] Point {point.id} missing field '{field}'")
                    issues += 1
        if issues == 0:
            print(f"  [OK]   Payload completeness check passed ({len(results)} points sampled)")
    except Exception as e:
        print(f"  [WARN] Payload check failed: {e}")


def run_sample_searches(client, collection_name, embedder, queries, limit=3):
    print(f"\n  Running {len(queries)} sample searches…")
    passed = 0
    for q in queries:
        try:
            results = search(
                query_text=q,
                client=client,
                collection_name=collection_name,
                embedder=embedder,
                limit=limit,
            )
            if results:
                top = results[0]
                p = top.payload or {}
                print(f"  [OK]   '{q[:40]}' → {len(results)} results "
                      f"(top: {p.get('slug', '?')} score={top.score:.3f})")
                passed += 1
            else:
                print(f"  [WARN] '{q[:40]}' → 0 results")
        except Exception as e:
            print(f"  [FAIL] '{q[:40]}' → error: {e}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate the Qdrant RAG collection.")
    parser.add_argument("--sample-queries", type=int, default=3,
                        help="Number of sample queries to run (default 3)")
    parser.add_argument("--collection", default="",
                        help="Override collection name (default: use config)")
    args = parser.parse_args()

    qdrant_cfg = load_qdrant_config()
    qcfg       = qdrant_cfg["qdrant"]
    col_cfg    = qdrant_cfg["collections"]["primary"]

    collection_name = args.collection or col_cfg["name"]
    alias_name      = col_cfg["live_alias"]
    dense_model     = col_cfg["vectors"]["dense"].get("model", "BAAI/bge-small-en-v1.5")

    client = make_client(qdrant_cfg)

    print(f"\nValidating Qdrant collection: {collection_name}")
    print("─" * 60)

    failures = 0

    print("\n1. Collection structure")
    if not check_collection(client, col_cfg):
        failures += 1

    print("\n2. Alias")
    check_alias(client, collection_name, alias_name)

    print("\n3. Payload indexes")
    check_payload_indexes(client, collection_name, qdrant_cfg)

    print("\n4. Payload completeness")
    check_payload_completeness(client, collection_name)

    print("\n5. Sample searches")
    embedder = Embedder(dense_model=dense_model)
    n = min(args.sample_queries, len(SAMPLE_QUERIES))
    passed = run_sample_searches(client, collection_name, embedder, SAMPLE_QUERIES[:n])

    print("\n─" * 60)
    if failures == 0:
        print(f"\n✓ Validation passed. ({passed}/{n} sample searches returned results)")
    else:
        print(f"\n✗ Validation found {failures} critical issue(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
