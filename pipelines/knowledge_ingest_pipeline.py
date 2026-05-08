"""
Knowledge ingest pipeline — reads data/knowledge/**/*.md, validates front matter,
chunks, embeds, and upserts into maharat_knowledge_memory_v1.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qdrant_client.models import PointStruct, SparseVector

from services.chunk_service import make_chunks, parse_markdown
from services.config_service import (
    load_knowledge_chunking_config,
    load_qdrant_config,
)
from services.embedding_service import EmbeddingService
from services.entity_service import EntityService
from services.qdrant_service import QdrantService, chunk_id_to_uuid

ROOT          = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
LOGS_DIR      = ROOT / "logs"

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = (
    "title", "slug", "knowledge_type", "status",
    "published", "language", "priority", "memory_layer",
)

_FAQ_KEYWORDS        = ("faq", "frequently asked", "questions & answers", "q&a")
_DEFINITION_KEYWORDS = ("definition", "glossary", "terminology", "key terms", "terms")


# ── helpers ────────────────────────────────────────────────────────────────────

def _discover_files(knowledge_dir: Path) -> List[Path]:
    return sorted(knowledge_dir.rglob("*.md"))


def _validate_front(front: dict) -> List[str]:
    """Return list of validation error strings; empty = valid."""
    errors = []
    for field in _REQUIRED_FIELDS:
        if front.get(field) in (None, ""):
            errors.append(f"missing field: {field}")
    if front.get("memory_layer") != "knowledge":
        errors.append(f"memory_layer must be 'knowledge', got '{front.get('memory_layer')}'")
    return errors


def _map_chunk_type(original_type: str, heading_path: str) -> str:
    """Remap make_chunks types to knowledge vocabulary."""
    if original_type == "summary":
        return "overview"
    hp = (heading_path or "").lower()
    if any(kw in hp for kw in _FAQ_KEYWORDS):
        return "faq"
    if any(kw in hp for kw in _DEFINITION_KEYWORDS):
        return "definition"
    return "section"


def _build_knowledge_payload(
    front: dict,
    chunk: dict,
    entities: dict,
    source_file: Path,
) -> dict:
    slug = front.get("slug", "")
    return {
        "doc_id":                  slug,
        "chunk_id":                chunk["chunk_id"],
        "title":                   front.get("title") or None,
        "slug":                    slug,
        "knowledge_type":          front.get("knowledge_type") or None,
        "section":                 chunk.get("heading_path") or None,
        "chunk_index":             chunk["chunk_index"],
        "chunk_type":              _map_chunk_type(
                                       chunk["chunk_type"],
                                       chunk.get("heading_path", ""),
                                   ),
        "chunk_text":              chunk["chunk_text"],
        "priority":                front.get("priority") or None,
        "language":                front.get("language") or "en",
        "status":                  front.get("status") or "approved",
        "published":               bool(front.get("published", True)),
        "memory_layer":            front.get("memory_layer") or "knowledge",
        "source_file":             (
                                       str(source_file.relative_to(ROOT))
                                       if source_file.is_relative_to(ROOT)
                                       else str(source_file)
                                   ),
        "word_count":              chunk["word_count"],
        "entities_organizations":  entities.get("organizations", []),
        "entities_programs":       entities.get("programs", []),
        "entities_locations":      entities.get("locations", []),
        "entities_credentials":    entities.get("credentials", []),
        "entities_people":         entities.get("people", []),
        "ingested_at":             datetime.now(timezone.utc).isoformat(),
    }


# ── pipeline ───────────────────────────────────────────────────────────────────

class KnowledgeIngestPipeline:
    """
    Full knowledge ingestion pipeline:
    discover → parse → validate → chunk → embed → upsert → verify.
    """

    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        entity_service: EntityService,
        chunk_cfg: dict,
        knowledge_dir: Optional[Path] = None,
        batch_size: int = 64,
    ):
        self.qdrant_service    = qdrant_service
        self.embedding_service = embedding_service
        self.entity_service    = entity_service
        self.chunk_cfg         = chunk_cfg
        self.knowledge_dir     = knowledge_dir or KNOWLEDGE_DIR
        self.batch_size        = batch_size

    @classmethod
    def from_config(cls) -> "KnowledgeIngestPipeline":
        qdrant_cfg = load_qdrant_config()
        chunk_cfg  = load_knowledge_chunking_config()
        col_cfg    = qdrant_cfg["collections"]["knowledge"]
        return cls(
            qdrant_service=QdrantService.from_config(
                qdrant_cfg, collection_key="knowledge"
            ),
            embedding_service=EmbeddingService.from_config(col_cfg),
            entity_service=EntityService.from_config(),
            chunk_cfg=chunk_cfg,
            batch_size=qdrant_cfg.get("ingestion", {}).get("upsert_batch_size", 64),
        )

    def run(self, dry_run: bool = False, recreate: bool = False) -> Dict:
        k_cfg          = self.chunk_cfg.get("chunking", {})
        max_tokens     = int(k_cfg.get("max_words", 450) * 1.3)
        overlap_tokens = int(k_cfg.get("overlap_words", 50) * 1.3)

        # collection setup
        if not dry_run:
            self.qdrant_service.setup_collection(recreate=recreate)
            self.qdrant_service.setup_payload_indexes()
            self.qdrant_service.setup_alias()

        # [1/6] discover
        print("\n[1/6] Discovering knowledge files…")
        md_files = _discover_files(self.knowledge_dir)
        if not md_files:
            print(f"  No .md files found under {self.knowledge_dir}")
            return {"files_found": 0, "files_valid": 0, "files_failed": 0,
                    "chunks_produced": 0, "points_upserted": 0}
        print(f"  Found {len(md_files)} file(s)  [dry_run={dry_run}]")

        # [2/6] parse + validate
        print("\n[2/6] Parsing and validating front matter…")
        valid_docs: List[Tuple[dict, str, Path]] = []
        failed: List[dict] = []

        for md_path in md_files:
            front, body = parse_markdown(md_path)
            if not front:
                reason = "no front matter"
                print(f"  [skip] {md_path.name}: {reason}")
                failed.append({"file": str(md_path), "reason": reason})
                continue
            errors = _validate_front(front)
            if errors:
                reason = "; ".join(errors)
                print(f"  [fail] {md_path.name}: {reason}")
                failed.append({"file": str(md_path), "reason": reason})
                continue
            valid_docs.append((front, body, md_path))

        print(f"  Valid: {len(valid_docs)}  |  Failed: {len(failed)}")

        # [3/6] chunk + entity extraction
        print("\n[3/6] Chunking and extracting entities…")
        pending = []
        for front, body, md_path in valid_docs:
            entities = self.entity_service.extract_from_article(front, body)
            for chunk in make_chunks(front, body,
                                     max_tokens=max_tokens,
                                     overlap_tokens=overlap_tokens):
                pending.append({
                    "id":      chunk_id_to_uuid(chunk["chunk_id"]),
                    "text":    chunk["chunk_text"],
                    "payload": _build_knowledge_payload(front, chunk, entities, md_path),
                })
        print(f"  Total chunks: {len(pending)}")

        # [4/6] embed
        print("\n[4/6] Generating embeddings…")
        points = []
        for start in range(0, len(pending), self.batch_size):
            batch    = pending[start: start + self.batch_size]
            texts    = [item["text"] for item in batch]
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
            print(f"  Embedded {min(start + self.batch_size, len(pending))}/{len(pending)}")

        # [5/6] upsert
        print("\n[5/6] Upserting to Qdrant…")
        upserted = 0
        if dry_run:
            print(f"  [dry-run] would upsert {len(points)} points")
        else:
            upserted = self.qdrant_service.upsert_points(points, batch_size=self.batch_size)

        # [6/6] verify + log
        print("\n[6/6] Verifying…")
        print(f"  Files found    : {len(md_files)}")
        print(f"  Files valid    : {len(valid_docs)}")
        print(f"  Files failed   : {len(failed)}")
        print(f"  Chunks produced: {len(pending)}")
        print(f"  Points embedded: {len(points)}")
        if not dry_run:
            info = self.qdrant_service.get_collection_info()
            print(f"  Points in Qdrant: {info.points_count}")

        if failed:
            LOGS_DIR.mkdir(exist_ok=True)
            fail_path = LOGS_DIR / "failed_knowledge.jsonl"
            with open(fail_path, "a", encoding="utf-8") as fh:
                for rec in failed:
                    fh.write(json.dumps(rec) + "\n")
            print(f"\n  Failed records written to {fail_path}")

        print("\n✓ Knowledge ingestion complete.")
        return {
            "files_found":     len(md_files),
            "files_valid":     len(valid_docs),
            "files_failed":    len(failed),
            "chunks_produced": len(pending),
            "points_upserted": upserted,
        }
