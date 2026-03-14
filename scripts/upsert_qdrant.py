#!/usr/bin/env python3
"""
Maharat RAG Ingestion Pipeline
================================
Six-stage pipeline: load → validate → chunk → embed → upsert → verify.

Reads normalized markdown posts from data/posts/, chunks each article,
generates dense + sparse embeddings, and upserts into Qdrant.

Usage:
    python scripts/upsert_qdrant.py
    python scripts/upsert_qdrant.py --dry-run
    python scripts/upsert_qdrant.py --recreate      # drop & recreate collection
    python scripts/upsert_qdrant.py --slug mctc-hosts-fire-drill  # single post
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    WalConfigDiff,
)

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_markdown import make_chunks
from config import load_chunking_config, load_qdrant_config, load_taxonomy, get_taxonomy_rules, make_client
from embed_chunks import Embedder
from ingest_markdown import load_posts, validate_all

POSTS_DIR = ROOT / "data" / "posts"
LOGS_DIR  = ROOT / "logs"


# ── helpers ────────────────────────────────────────────────────────────────

def _chunk_id_to_uuid(chunk_id):
    """Deterministic UUID5 from a chunk_id string."""
    ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace
    return str(uuid.uuid5(ns, chunk_id))


def _derive_month(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").month
    except ValueError:
        return None


def _iso_datetime(date_str):
    """Convert YYYY-MM-DD to RFC 3339 UTC string."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(str(date_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _null(val):
    if val in (None, "", 0, [], {}):
        return None
    return val


# ── collection setup ───────────────────────────────────────────────────────

def setup_collection(client, col_cfg, recreate=False):
    """Create (or recreate) the Qdrant collection with named dense + sparse vectors."""
    name = col_cfg["name"]

    if recreate and client.collection_exists(name):
        print(f"  Dropping existing collection '{name}'")
        client.delete_collection(name)

    if client.collection_exists(name):
        print(f"  Collection '{name}' already exists — skipping creation.")
        return

    dense_cfg  = col_cfg["vectors"]["dense"]
    hnsw_cfg   = col_cfg.get("hnsw", {})
    opt_cfg    = col_cfg.get("optimization", {})

    dist_map = {"Cosine": Distance.COSINE, "Euclid": Distance.EUCLID, "Dot": Distance.DOT}
    distance = dist_map.get(dense_cfg.get("distance", "Cosine"), Distance.COSINE)

    print(f"  Creating collection '{name}' (dense={dense_cfg['size']}d cosine + sparse BM25)…")

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(
                size=dense_cfg["size"],
                distance=distance,
                on_disk=dense_cfg.get("on_disk", False),
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        },
        hnsw_config=HnswConfigDiff(
            m=hnsw_cfg.get("m", 16),
            ef_construct=hnsw_cfg.get("ef_construct", 100),
            full_scan_threshold=hnsw_cfg.get("full_scan_threshold", 10000),
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=opt_cfg.get("indexing_threshold", 20000),
        ),
        on_disk_payload=col_cfg.get("storage", {}).get("on_disk_payload", True),
    )
    print("  Collection created.")


def setup_payload_indexes(client, collection_name, qdrant_cfg):
    """Create all payload indexes defined in qdrant.yaml."""
    from qdrant_client.models import PayloadSchemaType

    type_map = {
        "keyword":  PayloadSchemaType.KEYWORD,
        "integer":  PayloadSchemaType.INTEGER,
        "float":    PayloadSchemaType.FLOAT,
        "bool":     PayloadSchemaType.BOOL,
        "datetime": PayloadSchemaType.DATETIME,
        "geo":      PayloadSchemaType.GEO,
        "text":     PayloadSchemaType.TEXT,
    }

    indexes = qdrant_cfg.get("payload_indexes", [])
    print(f"  Creating {len(indexes)} payload indexes…")
    for idx_def in indexes:
        field   = idx_def["field_name"]
        schema  = idx_def["field_schema"]
        ptype   = type_map.get(schema)
        if ptype is None:
            print(f"    Warning: unknown index type '{schema}' for '{field}', skipping.")
            continue
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=ptype,
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"    Warning: index '{field}': {e}")


def setup_alias(client, collection_name, alias_name):
    """Point alias_name → collection_name (creates or updates)."""
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=collection_name,
                    alias_name=alias_name,
                )
            )
        ]
    )
    print(f"  Alias '{alias_name}' → '{collection_name}'")


# ── payload builder ────────────────────────────────────────────────────────

def build_payload(front, chunk, ingest_cfg, taxonomy_rules, version="1.0"):
    slug      = front.get("slug", "")
    date_str  = str(front.get("date", "") or "")
    gallery   = front.get("gallery_images") or []
    img       = front.get("featured_image") or ""
    tags_raw  = front.get("tags") or []

    return {
        # ── article identity
        "article_id":       slug,
        "chunk_id":         chunk["chunk_id"],
        "title":            _null(front.get("title", "")),
        "slug":             slug,
        "url":              f"posts/{slug}" if slug else None,
        "source_type":      "news_archive",
        "content_type":     "news_post",

        # ── chunk fields
        "chunk_index":      chunk["chunk_index"],
        "chunk_type":       chunk["chunk_type"],
        "heading_path":     _null(chunk.get("heading_path", "")),
        "chunk_text":       chunk["chunk_text"],
        "word_count":       chunk["word_count"],

        # ── article content
        "summary":          _null(front.get("summary", "")),

        # ── dates
        "date":             _iso_datetime(date_str),
        "year":             _null(front.get("year", 0)),
        "quarter":          _null(front.get("quarter", "")),
        "month":            _derive_month(date_str),

        # ── taxonomy
        "category":         _null(front.get("category", "")),
        "tags":             tags_raw if isinstance(tags_raw, list) else [],
        "partner":          _null(front.get("partner", "")),
        "location":         _null(front.get("location", "")),
        "audience":         ["Trainees", "Stakeholders"],

        # ── media
        "featured_image":   _null(img),
        "image_names":      ([img] + gallery) if img else (gallery or []),

        # ── provenance
        "language":         taxonomy_rules.get("default_language", "en"),
        "status":           taxonomy_rules.get("default_status", "approved"),
        "visibility":       taxonomy_rules.get("default_visibility", "public"),
        "published":        True,
        "source_document":  _null(front.get("source_document", "")),
        "source_section":   _null(front.get("source_section", "")),
        "source_page_start": _null(front.get("source_page", 0)),
        "source_page_end":   _null(front.get("source_page", 0)),

        # ── system
        "version":          version,
        "ingested_at":      datetime.now(timezone.utc).isoformat(),
    }


# ── main ingestion ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest markdown posts into Qdrant.")
    parser.add_argument("--dry-run",  action="store_true", help="Parse and chunk without writing to Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection.")
    parser.add_argument("--slug",     default="",          help="Ingest only a single post by slug.")
    args = parser.parse_args()

    # ── load config ──────────────────────────────────────────────────────
    qdrant_cfg  = load_qdrant_config()
    chunk_cfg   = load_chunking_config()
    taxonomy    = load_taxonomy()

    col_cfg     = qdrant_cfg["collections"]["primary"]
    ingest_cfg  = qdrant_cfg.get("ingestion", {})
    tax_rules   = get_taxonomy_rules(taxonomy)

    collection_name = col_cfg["name"]
    alias_name      = col_cfg["live_alias"]
    batch_size      = ingest_cfg.get("upsert_batch_size", 64)

    max_tokens     = chunk_cfg.get("chunking", {}).get("max_tokens", 700)
    overlap_tokens = chunk_cfg.get("chunking", {}).get("overlap_tokens", 100)

    # ── connect ──────────────────────────────────────────────────────────
    client = make_client(qdrant_cfg)

    if not args.dry_run:
        setup_collection(client, col_cfg, recreate=args.recreate)
        setup_payload_indexes(client, collection_name, qdrant_cfg)
        setup_alias(client, collection_name, alias_name)

    # ── STAGE 1: load markdown posts ─────────────────────────────────────
    print("\n[1/6] Loading markdown posts…")
    parsed = load_posts(posts_dir=POSTS_DIR, slug_filter=args.slug or None)
    if not parsed:
        target = f"slug '{args.slug}'" if args.slug else str(POSTS_DIR)
        print(f"  No posts found in {target}")
        sys.exit(1)
    print(f"  Loaded {len(parsed)} post(s)  [dry_run={args.dry_run}]")

    # ── STAGE 2: metadata validation ─────────────────────────────────────
    print("\n[2/6] Validating metadata…")
    valid, failed = validate_all(parsed, taxonomy, ingest_cfg)
    print(f"  Valid: {len(valid)}  |  Failed: {len(failed)}")

    # ── STAGE 3: chunking ────────────────────────────────────────────────
    print("\n[3/6] Chunking posts…")
    pending = []  # list of {"id", "text", "payload"}
    for front, body in valid:
        chunks = make_chunks(front, body, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for chunk in chunks:
            pending.append({
                "id":      _chunk_id_to_uuid(chunk["chunk_id"]),
                "text":    chunk["chunk_text"],
                "payload": build_payload(front, chunk, ingest_cfg, tax_rules),
            })

    print(f"  Total chunks: {len(pending)}")

    # ── STAGE 4: embedding generation ───────────────────────────────────
    print("\n[4/6] Generating embeddings…")
    embedder = Embedder(
        dense_model=col_cfg["vectors"]["dense"].get("model", "BAAI/bge-small-en-v1.5"),
    )

    points = []  # list of PointStruct
    for batch_start in range(0, len(pending), batch_size):
        batch       = pending[batch_start: batch_start + batch_size]
        texts       = [item["text"] for item in batch]
        dense_vecs  = embedder.embed_dense(texts)
        sparse_vecs = embedder.embed_sparse(texts)
        for item, dv, sv in zip(batch, dense_vecs, sparse_vecs):
            points.append(
                PointStruct(
                    id=item["id"],
                    vector={
                        "dense": dv.tolist(),
                        "sparse": SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                    },
                    payload=item["payload"],
                )
            )
        print(f"  Embedded {min(batch_start + batch_size, len(pending))}/{len(pending)}")

    # ── STAGE 5: Qdrant upsert ───────────────────────────────────────────
    print("\n[5/6] Upserting to Qdrant…")
    if args.dry_run:
        print(f"  [dry-run] would upsert {len(points)} points")
    else:
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start: batch_start + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            print(f"  Upserted {min(batch_start + batch_size, len(points))}/{len(points)}")

    # ── STAGE 6: verification ────────────────────────────────────────────
    print("\n[6/6] Verifying…")
    print(f"  Posts loaded    : {len(parsed)}")
    print(f"  Posts valid     : {len(valid)}")
    print(f"  Posts failed    : {len(failed)}")
    print(f"  Chunks produced : {len(pending)}")
    print(f"  Points embedded : {len(points)}")
    if not args.dry_run:
        info = client.get_collection(collection_name)
        print(f"  Points in Qdrant: {info.points_count}")

    if failed:
        LOGS_DIR.mkdir(exist_ok=True)
        fail_path = LOGS_DIR / "failed_records.jsonl"
        with open(fail_path, "a", encoding="utf-8") as fh:
            for rec in failed:
                fh.write(json.dumps(rec) + "\n")
        print(f"\n  Failed records written to {fail_path}")

    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    main()
