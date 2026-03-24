"""
Retrieval service — hybrid search over Qdrant using dense + sparse (RRF).
Self-contained: does not import from scripts/search_qdrant.py to avoid
module-level side effects in that file.
"""

from typing import List, Optional

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from services.config_service import load_qdrant_config, make_client
from services.embedding_service import EmbeddingService


class RetrievalService:
    """Hybrid search: dense Prefetch + sparse Prefetch fused with RRF."""

    def __init__(self, client, collection_name: str, embedding_service: EmbeddingService):
        self.client            = client
        self.collection_name   = collection_name
        self.embedding_service = embedding_service

    @classmethod
    def from_config(cls, qdrant_cfg: Optional[dict] = None) -> "RetrievalService":
        if qdrant_cfg is None:
            qdrant_cfg = load_qdrant_config()
        col_cfg     = qdrant_cfg["collections"]["primary"]
        client      = make_client(qdrant_cfg)
        emb_service = EmbeddingService.from_config(col_cfg)
        coll_name   = col_cfg.get("live_alias") or col_cfg["name"]
        return cls(client=client, collection_name=coll_name, embedding_service=emb_service)

    def build_filter(
        self,
        category: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
        status: Optional[str] = None,
        published: Optional[bool] = None,
        visibility: Optional[str] = None,
    ) -> Optional[Filter]:
        must = []
        if category:
            must.append(FieldCondition(key="category",   match=MatchValue(value=category)))
        if year:
            must.append(FieldCondition(key="year",       match=MatchValue(value=int(year))))
        if quarter:
            must.append(FieldCondition(key="quarter",    match=MatchValue(value=quarter)))
        if chunk_type:
            must.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        if language:
            must.append(FieldCondition(key="language",   match=MatchValue(value=language)))
        if status:
            must.append(FieldCondition(key="status",     match=MatchValue(value=status)))
        if published is not None:
            must.append(FieldCondition(key="published",  match=MatchValue(value=published)))
        if visibility:
            must.append(FieldCondition(key="visibility", match=MatchValue(value=visibility)))
        return Filter(must=must) if must else None

    def search(
        self,
        query_text: str,
        limit: int = 8,
        candidate_limit: int = 24,
        score_threshold: float = 0.0,
        query_filter: Optional[Filter] = None,
    ) -> list:
        """
        Hybrid search using Prefetch + FusionQuery (RRF).
        Returns list of ScoredPoint.
        """
        dense_vec, sparse_emb = self.embedding_service.embed_query(query_text)
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
        results = self.client.query_points(
            collection_name=self.collection_name,
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
