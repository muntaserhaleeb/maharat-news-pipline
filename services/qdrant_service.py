"""
Qdrant service — collection management and point upsert.
Extracts the collection setup and payload-building logic from scripts/upsert_qdrant.py.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    WalConfigDiff,
)

from services.config_service import get_taxonomy_rules, load_qdrant_config, make_client


# ── module-level helpers (public) ──────────────────────────────────────────

def chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID5 from a chunk_id string."""
    ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(ns, chunk_id))


def derive_month(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").month
    except ValueError:
        return None


def iso_datetime(date_str: str) -> Optional[str]:
    """Convert YYYY-MM-DD to RFC 3339 UTC string."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(str(date_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def null_if_empty(val) -> Optional[object]:
    if val in (None, "", 0, [], {}):
        return None
    return val


def build_payload(front: dict, chunk: dict, ingest_cfg: dict,
                  taxonomy_rules: dict, version: str = "1.0",
                  entities: Optional[dict] = None) -> dict:
    slug     = front.get("slug", "")
    date_str = str(front.get("date", "") or "")
    gallery  = front.get("gallery_images") or []
    img      = front.get("featured_image") or ""
    tags_raw = front.get("tags") or []
    ents     = entities or {}

    return {
        "article_id":        slug,
        "chunk_id":          chunk["chunk_id"],
        "title":             null_if_empty(front.get("title", "")),
        "slug":              slug,
        "url":               f"posts/{slug}" if slug else None,
        "source_type":       "news_archive",
        "content_type":      "news_post",
        "chunk_index":       chunk["chunk_index"],
        "chunk_type":        chunk["chunk_type"],
        "heading_path":      null_if_empty(chunk.get("heading_path", "")),
        "chunk_text":        chunk["chunk_text"],
        "word_count":        chunk["word_count"],
        "summary":           null_if_empty(front.get("summary", "")),
        "date":              iso_datetime(date_str),
        "year":              null_if_empty(front.get("year", 0)),
        "quarter":           null_if_empty(front.get("quarter", "")),
        "month":             derive_month(date_str),
        "category":          null_if_empty(front.get("category", "")),
        "tags":              tags_raw if isinstance(tags_raw, list) else [],
        "partner":           null_if_empty(front.get("partner", "")),
        "location":          null_if_empty(front.get("location", "")),
        "audience":          ["Trainees", "Stakeholders"],
        "featured_image":    null_if_empty(img),
        "image_names":       ([img] + gallery) if img else (gallery or []),
        "language":          taxonomy_rules.get("default_language", "en"),
        "status":            taxonomy_rules.get("default_status", "approved"),
        "visibility":        taxonomy_rules.get("default_visibility", "public"),
        "published":         True,
        "source_document":   null_if_empty(front.get("source_document", "")),
        "source_section":    null_if_empty(front.get("source_section", "")),
        "source_page_start": null_if_empty(front.get("source_page", 0)),
        "source_page_end":   null_if_empty(front.get("source_page", 0)),
        "version":           version,
        "ingested_at":       datetime.now(timezone.utc).isoformat(),

        # ── extracted entities (set per-article, attached to every chunk)
        "entities_organizations": ents.get("organizations", []),
        "entities_programs":      ents.get("programs", []),
        "entities_locations":     ents.get("locations", []),
        "entities_credentials":   ents.get("credentials", []),
        "entities_people":        ents.get("people", []),
    }


# ── service class ──────────────────────────────────────────────────────────

class QdrantService:
    """Collection management and point upsert operations."""

    def __init__(self, client, col_cfg: dict, qdrant_cfg: dict):
        self.client          = client
        self.col_cfg         = col_cfg
        self.qdrant_cfg      = qdrant_cfg
        self.collection_name = col_cfg["name"]
        self.alias_name      = col_cfg["live_alias"]

    @classmethod
    def from_config(cls, qdrant_cfg: Optional[dict] = None) -> "QdrantService":
        if qdrant_cfg is None:
            qdrant_cfg = load_qdrant_config()
        client  = make_client(qdrant_cfg)
        col_cfg = qdrant_cfg["collections"]["primary"]
        return cls(client=client, col_cfg=col_cfg, qdrant_cfg=qdrant_cfg)

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def get_collection_info(self):
        return self.client.get_collection(self.collection_name)

    def setup_collection(self, recreate: bool = False) -> None:
        name      = self.collection_name
        col_cfg   = self.col_cfg
        client    = self.client

        if recreate and client.collection_exists(name):
            print(f"  Dropping existing collection '{name}'")
            client.delete_collection(name)

        if client.collection_exists(name):
            print(f"  Collection '{name}' already exists — skipping creation.")
            return

        dense_cfg = col_cfg["vectors"]["dense"]
        hnsw_cfg  = col_cfg.get("hnsw", {})
        opt_cfg   = col_cfg.get("optimization", {})
        dist_map  = {"Cosine": Distance.COSINE, "Euclid": Distance.EUCLID, "Dot": Distance.DOT}
        distance  = dist_map.get(dense_cfg.get("distance", "Cosine"), Distance.COSINE)

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
            sparse_vectors_config={"sparse": SparseVectorParams()},
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

    def setup_payload_indexes(self) -> None:
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
        indexes = self.qdrant_cfg.get("payload_indexes", [])
        print(f"  Creating {len(indexes)} payload indexes…")
        for idx_def in indexes:
            field  = idx_def["field_name"]
            schema = idx_def["field_schema"]
            ptype  = type_map.get(schema)
            if ptype is None:
                print(f"    Warning: unknown index type '{schema}' for '{field}', skipping.")
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=ptype,
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"    Warning: index '{field}': {e}")

    def setup_alias(self) -> None:
        self.client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=self.collection_name,
                        alias_name=self.alias_name,
                    )
                )
            ]
        )
        print(f"  Alias '{self.alias_name}' → '{self.collection_name}'")

    def upsert_points(self, points: list, batch_size: int = 64) -> int:
        """Upsert points in batches. Returns total count upserted."""
        total = 0
        for start in range(0, len(points), batch_size):
            batch = points[start: start + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            total += len(batch)
            print(f"  Upserted {min(start + batch_size, len(points))}/{len(points)}")
        return total
