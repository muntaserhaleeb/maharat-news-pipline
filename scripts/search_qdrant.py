#!/usr/bin/env python3
"""
Maharat Hybrid Search CLI
==========================
Searches the Qdrant collection using dense + sparse hybrid retrieval (RRF fusion).

Usage:
    python scripts/search_qdrant.py "fire safety training"
    python scripts/search_qdrant.py "welding" --limit 5
    python scripts/search_qdrant.py "OJT" --category "Staff Development"
    python scripts/search_qdrant.py "safety drill" --year 2026 --quarter Q1
    python scripts/search_qdrant.py "graduation" --category "Graduations" --score-threshold 0.2
"""

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_qdrant_config
from embed_chunks import Embedder


def build_filter(category=None, year=None, quarter=None, chunk_type=None,
                 language=None, status=None, published=None):
    """Build a Qdrant Filter from optional field constraints."""
    must = []

    if category:
        must.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if year:
        must.append(FieldCondition(key="year", match=MatchValue(value=int(year))))
    if quarter:
        must.append(FieldCondition(key="quarter", match=MatchValue(value=quarter)))
    if chunk_type:
        must.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
    if language:
        must.append(FieldCondition(key="language", match=MatchValue(value=language)))
    if status:
        must.append(FieldCondition(key="status", match=MatchValue(value=status)))
    if published is not None:
        must.append(FieldCondition(key="published", match=MatchValue(value=published)))

    return Filter(must=must) if must else None


def search(
    query_text,
    client,
    collection_name,
    embedder,
    limit=8,
    candidate_limit=24,
    score_threshold=0.0,
    query_filter=None,
):
    """
    Hybrid search using Qdrant Prefetch + FusionQuery (RRF).
    Returns list of ScoredPoint results.
    """
    dense_vec  = embedder.embed_query_dense(query_text)
    sparse_emb = embedder.embed_query_sparse(query_text)
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense_vec.tolist(),
                using="dense",
                limit=candidate_limit,
                filter=query_filter,
            ),
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=candidate_limit,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        score_threshold=score_threshold if score_threshold > 0 else None,
        with_payload=True,
    )
    return results.points


def main():
    parser = argparse.ArgumentParser(description="Hybrid search over Maharat content.")
    parser.add_argument("query",                                  help="Search query text")
    parser.add_argument("--limit",           type=int, default=8, help="Number of results (default 8)")
    parser.add_argument("--candidates",      type=int, default=0, help="Candidate pool per vector (default from config)")
    parser.add_argument("--category",        default="",          help="Filter by category")
    parser.add_argument("--year",            type=int, default=0, help="Filter by year (e.g. 2026)")
    parser.add_argument("--quarter",         default="",          help="Filter by quarter (e.g. Q1)")
    parser.add_argument("--chunk-type",      default="",          help="Filter by chunk_type (summary/body)")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum relevance score")
    parser.add_argument("--json",            action="store_true", help="Output raw JSON")
    parser.add_argument("--collection",      default="",          help="Override collection name")
    args = parser.parse_args()

    qdrant_cfg = load_qdrant_config()
    qcfg       = qdrant_cfg["qdrant"]
    col_cfg    = qdrant_cfg["collections"]["primary"]
    retrieval  = qdrant_cfg.get("retrieval", {})

    collection_name  = args.collection or col_cfg.get("live_alias") or col_cfg["name"]
    candidate_limit  = args.candidates or retrieval.get("candidate_limit", 24)
    score_threshold  = args.score_threshold or retrieval.get("score_threshold", 0.0)
    dense_model      = col_cfg["vectors"]["dense"].get("model", "BAAI/bge-small-en-v1.5")

    client   = QdrantClient(url=qcfg["url"], api_key=qcfg.get("api_key"))
    embedder = Embedder(dense_model=dense_model)

    query_filter = build_filter(
        category=args.category or None,
        year=args.year or None,
        quarter=args.quarter or None,
        chunk_type=args.chunk_type or None,
    )

    results = search(
        query_text=args.query,
        client=client,
        collection_name=collection_name,
        embedder=embedder,
        limit=args.limit,
        candidate_limit=candidate_limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

    if args.json:
        output = []
        for r in results:
            output.append({"id": r.id, "score": r.score, "payload": r.payload})
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print(f"\nQuery: \"{args.query}\"")
    if query_filter:
        print(f"Filter: category={args.category or '-'} year={args.year or '-'} "
              f"quarter={args.quarter or '-'} chunk_type={args.chunk_type or '-'}")
    print(f"Results: {len(results)}\n")
    print("─" * 72)

    for i, r in enumerate(results, 1):
        p = r.payload or {}
        print(f"{i:2d}. [{r.score:.4f}] {p.get('title', '—')}")
        print(f"     slug      : {p.get('slug', '')}")
        print(f"     chunk_id  : {p.get('chunk_id', '')}")
        print(f"     category  : {p.get('category', '')}  |  date: {p.get('date', '')}")
        tags = p.get("tags") or []
        if tags:
            print(f"     tags      : {', '.join(tags)}")
        chunk_text = (p.get("chunk_text") or "")[:200]
        if chunk_text:
            print(f"     excerpt   : {chunk_text}…")
        print()


if __name__ == "__main__":
    main()
