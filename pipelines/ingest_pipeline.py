"""
Ingest pipeline — orchestrates: load → validate → chunk → embed → upsert → verify.
Imports from services only.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from qdrant_client.models import PointStruct, SparseVector

from services.chunk_service import load_posts, make_chunks, validate_all
from services.config_service import (
    get_taxonomy_rules,
    load_chunking_config,
    load_qdrant_config,
    load_taxonomy,
)
from services.embedding_service import EmbeddingService
from services.entity_service import EntityService
from services.qdrant_service import QdrantService, build_payload, chunk_id_to_uuid

ROOT      = Path(__file__).resolve().parent.parent
POSTS_DIR = ROOT / "data" / "posts"
LOGS_DIR  = ROOT / "logs"


class IngestPipeline:
    """Full ingestion pipeline: load markdown → validate → chunk → embed → upsert → verify."""

    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        entity_service: EntityService,
        chunk_cfg: dict,
        ingest_cfg: dict,
        taxonomy: dict,
        taxonomy_rules: dict,
        posts_dir: Optional[Path] = None,
    ):
        self.qdrant_service    = qdrant_service
        self.embedding_service = embedding_service
        self.entity_service    = entity_service
        self.chunk_cfg         = chunk_cfg
        self.ingest_cfg        = ingest_cfg
        self.taxonomy          = taxonomy
        self.taxonomy_rules    = taxonomy_rules
        self.posts_dir         = posts_dir or POSTS_DIR

    @classmethod
    def from_config(cls) -> "IngestPipeline":
        qdrant_cfg     = load_qdrant_config()
        chunk_cfg_dict = load_chunking_config()
        taxonomy       = load_taxonomy()
        col_cfg        = qdrant_cfg["collections"]["primary"]
        return cls(
            qdrant_service=QdrantService.from_config(qdrant_cfg),
            embedding_service=EmbeddingService.from_config(col_cfg),
            entity_service=EntityService.from_config(),
            chunk_cfg=chunk_cfg_dict,
            ingest_cfg=qdrant_cfg.get("ingestion", {}),
            taxonomy=taxonomy,
            taxonomy_rules=get_taxonomy_rules(taxonomy),
        )

    def run(
        self,
        slug_filter: Optional[str] = None,
        dry_run: bool = False,
        recreate: bool = False,
    ) -> Dict:
        """
        Execute the full pipeline.
        Returns summary dict: loaded, valid, failed, chunks_produced, points_upserted.
        """
        max_tokens     = self.chunk_cfg.get("chunking", {}).get("max_tokens", 700)
        overlap_tokens = self.chunk_cfg.get("chunking", {}).get("overlap_tokens", 100)
        batch_size     = self.ingest_cfg.get("upsert_batch_size", 64)

        # collection setup
        if not dry_run:
            self.qdrant_service.setup_collection(recreate=recreate)
            self.qdrant_service.setup_payload_indexes()
            self.qdrant_service.setup_alias()

        # [1/6] load
        print("\n[1/6] Loading markdown posts…")
        parsed = load_posts(posts_dir=self.posts_dir, slug_filter=slug_filter)
        if not parsed:
            target = f"slug '{slug_filter}'" if slug_filter else str(self.posts_dir)
            print(f"  No posts found in {target}")
            return {"loaded": 0, "valid": 0, "failed": 0,
                    "chunks_produced": 0, "points_upserted": 0}
        print(f"  Loaded {len(parsed)} post(s)  [dry_run={dry_run}]")

        # [2/6] validate
        print("\n[2/6] Validating metadata…")
        valid, failed = validate_all(parsed, self.taxonomy, self.ingest_cfg)
        print(f"  Valid: {len(valid)}  |  Failed: {len(failed)}")

        # [3/6] chunk + entity extraction
        print("\n[3/6] Chunking posts and extracting entities…")
        pending = []
        for front, body in valid:
            entities = self.entity_service.extract_from_article(front, body)
            for chunk in make_chunks(front, body,
                                     max_tokens=max_tokens,
                                     overlap_tokens=overlap_tokens):
                pending.append({
                    "id":      chunk_id_to_uuid(chunk["chunk_id"]),
                    "text":    chunk["chunk_text"],
                    "payload": build_payload(
                        front, chunk, self.ingest_cfg, self.taxonomy_rules,
                        entities=entities,
                    ),
                })
        print(f"  Total chunks: {len(pending)}")

        # [4/6] embed
        print("\n[4/6] Generating embeddings…")
        points = []
        for start in range(0, len(pending), batch_size):
            batch            = pending[start: start + batch_size]
            texts            = [item["text"] for item in batch]
            dense_vecs, sparse_vecs = self.embedding_service.embed_documents(texts)
            for item, dv, sv in zip(batch, dense_vecs, sparse_vecs):
                points.append(PointStruct(
                    id=item["id"],
                    vector={
                        "dense": dv.tolist(),
                        "sparse": SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                    },
                    payload=item["payload"],
                ))
            print(f"  Embedded {min(start + batch_size, len(pending))}/{len(pending)}")

        # [5/6] upsert
        print("\n[5/6] Upserting to Qdrant…")
        upserted = 0
        if dry_run:
            print(f"  [dry-run] would upsert {len(points)} points")
        else:
            upserted = self.qdrant_service.upsert_points(points, batch_size=batch_size)

        # [6/6] verify
        print("\n[6/6] Verifying…")
        print(f"  Posts loaded    : {len(parsed)}")
        print(f"  Posts valid     : {len(valid)}")
        print(f"  Posts failed    : {len(failed)}")
        print(f"  Chunks produced : {len(pending)}")
        print(f"  Points embedded : {len(points)}")
        if not dry_run:
            info = self.qdrant_service.get_collection_info()
            print(f"  Points in Qdrant: {info.points_count}")

        if failed:
            LOGS_DIR.mkdir(exist_ok=True)
            fail_path = LOGS_DIR / "failed_records.jsonl"
            with open(fail_path, "a", encoding="utf-8") as fh:
                for rec in failed:
                    fh.write(json.dumps(rec) + "\n")
            print(f"\n  Failed records written to {fail_path}")

        print("\n✓ Ingestion complete.")
        return {
            "loaded":          len(parsed),
            "valid":           len(valid),
            "failed":          len(failed),
            "chunks_produced": len(pending),
            "points_upserted": upserted,
        }
