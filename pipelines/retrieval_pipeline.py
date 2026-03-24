"""
Retrieval pipeline — query → apply prefilters → hybrid search → ranked results.
Imports from services only.
"""

from typing import List, Optional

from services.config_service import load_generation_config, load_qdrant_config
from services.retrieval_service import RetrievalService


class RetrievalPipeline:
    """Orchestrates: query → inject prefilters → hybrid search → ranked results."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        retrieval_cfg: dict,
        gen_cfg: dict,
    ):
        self.retrieval_service = retrieval_service
        self.retrieval_cfg     = retrieval_cfg
        self.gen_cfg           = gen_cfg

    @classmethod
    def from_config(
        cls,
        qdrant_cfg: Optional[dict] = None,
        gen_cfg: Optional[dict] = None,
    ) -> "RetrievalPipeline":
        if qdrant_cfg is None:
            qdrant_cfg = load_qdrant_config()
        if gen_cfg is None:
            gen_cfg = load_generation_config().get("generation", {})
        return cls(
            retrieval_service=RetrievalService.from_config(qdrant_cfg),
            retrieval_cfg=qdrant_cfg.get("retrieval", {}),
            gen_cfg=gen_cfg,
        )

    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        chunk_type: Optional[str] = None,
        apply_prefilters: bool = True,
        score_threshold: Optional[float] = None,
    ) -> list:
        """
        Run hybrid retrieval; return up to `limit` ScoredPoint objects.
        apply_prefilters=True injects published/status/language/visibility
        from qdrant.yaml retrieval.prefilters.
        """
        # Resolve settings from new nested paths with flat-field fallbacks
        limit = limit or (
            self.gen_cfg.get("default_settings", {}).get("max_context_chunks")
            or self.gen_cfg.get("max_context_chunks", 8)
        )
        candidate_multiplier = (
            self.gen_cfg.get("retrieval_rules", {}).get("candidate_multiplier")
            or self.gen_cfg.get("candidate_multiplier", 3)
        )
        candidate_limit = limit * candidate_multiplier
        threshold = (
            score_threshold
            if score_threshold is not None
            else (
                self.gen_cfg.get("retrieval_rules", {}).get("score_threshold")
                or self.retrieval_cfg.get("score_threshold", 0.0)
            )
        )

        prefilters = self.retrieval_cfg.get("prefilters", {}) if apply_prefilters else {}
        query_filter = self.retrieval_service.build_filter(
            category=category,
            year=year,
            quarter=quarter,
            chunk_type=chunk_type,
            language=prefilters.get("language"),
            status=prefilters.get("status"),
            published=prefilters.get("published"),
            visibility=prefilters.get("visibility"),
        )

        return self.retrieval_service.search(
            query_text=query,
            limit=limit,
            candidate_limit=candidate_limit,
            score_threshold=threshold,
            query_filter=query_filter,
        )
