"""
Memory router — classifies queries AND orchestrates retrieval across collections.

Collections:
  maharat_content_live   — news, events, announcements, MoUs, graduations
  maharat_knowledge_live — institutional knowledge, programs, accreditations,
                           campus, strategy, FAQ, editorial rules

Route values:  "news" | "knowledge" | "both"

Usage (classify only):
    router = MemoryRouter()
    result = router.route_query("Write article about Maharat and Sinopec")
    print(result.route)          # "both"

Usage (full retrieval — requires from_config()):
    router = MemoryRouter.from_config()
    rr     = router.retrieve("Write article about Maharat and Sinopec")
    rr.news_chunks          # ScoredPoint list from content collection
    rr.knowledge_chunks     # ScoredPoint list from knowledge collection
    rr.editorial_chunks     # ScoredPoint list filtered to editorial_guidelines
    rr.graph_entities       # entity dicts from KnowledgeGraphService
    rr.to_debug_dict()      # full debug record for retrieval_debug.json
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


# ── routing result ────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    route: str                              # "news" | "knowledge" | "both"
    intent: str
    reasoning: str
    news_filters: Dict[str, Any] = field(default_factory=dict)
    knowledge_filters: Dict[str, Any] = field(default_factory=dict)


# ── retrieval result ──────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Full retrieval output: route decision + all chunk lists + graph context."""

    route: str
    route_reasoning: str
    collections_searched: List[str]
    news_chunks: List[Any]
    knowledge_chunks: List[Any]
    editorial_chunks: List[Any]
    graph_entities: List[Dict]          # entity dicts detected from query
    graph_context_blocks: List[str]     # formatted context strings for prompt

    @property
    def all_chunks(self) -> List[Any]:
        return self.news_chunks + self.knowledge_chunks + self.editorial_chunks

    @property
    def has_content(self) -> bool:
        return bool(self.news_chunks or self.knowledge_chunks or self.editorial_chunks)

    def to_debug_dict(self) -> Dict[str, Any]:
        """Build the retrieval_debug.json payload."""
        def _chunk_records(chunks: list, lane: str) -> list:
            records = []
            for i, pt in enumerate(chunks):
                p = pt.payload or {}
                records.append({
                    "lane":         lane,
                    "rank":         i + 1,
                    "score":        round(float(pt.score), 4),
                    "slug":         p.get("slug", ""),
                    "chunk_id":     p.get("chunk_id", ""),
                    "chunk_index":  p.get("chunk_index", 0),
                    "chunk_type":   p.get("chunk_type", ""),
                    "knowledge_type": p.get("knowledge_type", ""),
                    "title":        p.get("title", ""),
                    "date":         (p.get("date") or "")[:10],
                    "category":     p.get("category", ""),
                    "tags":         p.get("tags", []),
                    "word_count":   p.get("word_count", 0),
                    "chunk_text":   (p.get("chunk_text") or "")[:300],
                })
            return records

        return {
            "routing": {
                "selected_route":       self.route,
                "reasoning":            self.route_reasoning,
                "collections_searched": self.collections_searched,
            },
            "graph_entities_used": [
                {
                    "id":   e.get("id", ""),
                    "name": e.get("name", ""),
                    "type": e.get("type", ""),
                }
                for e in self.graph_entities
            ],
            "news_chunks":      _chunk_records(self.news_chunks,     "news"),
            "knowledge_chunks": _chunk_records(self.knowledge_chunks, "knowledge"),
            "editorial_chunks": _chunk_records(self.editorial_chunks, "editorial"),
            "totals": {
                "news":      len(self.news_chunks),
                "knowledge": len(self.knowledge_chunks),
                "editorial": len(self.editorial_chunks),
                "total":     len(self.all_chunks),
            },
        }


# ── main router class ─────────────────────────────────────────────────────

class MemoryRouter:
    """
    Keyword-signal classifier + optional retrieval orchestrator.

    When constructed with no arguments, route_query() works standalone.
    Call from_config() to load retrieval services and enable retrieve().
    """

    # ── signal lists ──────────────────────────────────────────────────────

    _DRAFT_SIGNALS: List[str] = [
        "draft", "write", "generate", "article", "linkedin", "post",
        "magazine", "press release", "news story", "feature story",
        "create content", "create article",
    ]

    _NEWS_SIGNALS: List[str] = [
        "event", "ceremony", "graduation", "mou", "agreement", "signed",
        "collaboration", "drill", "competition", "award", "visit",
        "workshop", "conference", "seminar", "celebration",
        "achievement", "activity", "highlights", "recap", "announcement",
        "hosted", "welcomed", "completed", "launched", "held", "attended",
        "sinopec", "samsung", "nesma", "saudi aramco", "sabic",
        "ojt", "on-the-job", "intake", "job fair",
        "recent", "latest", "this year", "2024", "2025", "2026",
    ]

    _KNOWLEDGE_SIGNALS: List[str] = [
        "what is maharat", "who is maharat", "about maharat",
        "mission", "vision", "history", "established", "founded",
        "accreditation", "certification", "iso", "abet", "ncaaa",
        "program", "course", "curriculum", "short course",
        "methodology", "training approach", "campus", "facility", "facilities",
        "faq", "frequently asked",
        "describe maharat", "explain maharat", "overview",
        "institutional", "governance", "strategy", "strategic",
        "brand", "editorial style", "credential", "qualification",
        "who are", "tell me about maharat",
    ]

    # ── init ──────────────────────────────────────────────────────────────

    def __init__(
        self,
        news_service=None,
        knowledge_service=None,
        graph_service=None,
        gen_cfg: Optional[dict] = None,
    ):
        self._news_svc      = news_service
        self._knowledge_svc = knowledge_service
        self._graph_svc     = graph_service
        self._gen_cfg       = gen_cfg or {}

    @classmethod
    def from_config(
        cls,
        qdrant_cfg: Optional[dict] = None,
        gen_cfg: Optional[dict] = None,
    ) -> "MemoryRouter":
        """
        Build a fully-loaded router with retrieval services and graph context.
        Uses a single shared Qdrant client to avoid embedded-storage locking.
        """
        from services.config_service import (
            load_qdrant_config, load_generation_config, make_client,
        )
        from services.embedding_service import EmbeddingService
        from services.retrieval_service import RetrievalService
        from services.knowledge_graph_service import KnowledgeGraphService

        if qdrant_cfg is None:
            qdrant_cfg = load_qdrant_config()
        if gen_cfg is None:
            gen_cfg = load_generation_config().get("generation", {})

        # One client to avoid embedded storage lock
        try:
            shared_client = make_client(qdrant_cfg)
        except Exception as exc:
            print(f"[memory_router] Qdrant unavailable: {exc}", file=sys.stderr)
            return cls(gen_cfg=gen_cfg)

        # News / content service
        primary_cfg  = qdrant_cfg["collections"]["primary"]
        news_svc     = RetrievalService(
            client=shared_client,
            collection_name=primary_cfg.get("live_alias") or primary_cfg["name"],
            embedding_service=EmbeddingService.from_config(primary_cfg),
        )

        # Knowledge service (optional)
        knowledge_svc = None
        knowledge_cfg = qdrant_cfg.get("collections", {}).get("knowledge")
        if knowledge_cfg:
            try:
                knowledge_svc = RetrievalService(
                    client=shared_client,
                    collection_name=knowledge_cfg.get("live_alias") or knowledge_cfg["name"],
                    embedding_service=EmbeddingService.from_config(knowledge_cfg),
                )
            except Exception as exc:
                print(f"[memory_router] Knowledge collection unavailable: {exc}", file=sys.stderr)

        # Graph service (optional)
        graph_svc = None
        try:
            graph_svc = KnowledgeGraphService.from_config()
        except Exception as exc:
            print(f"[memory_router] Knowledge graph unavailable: {exc}", file=sys.stderr)

        return cls(
            news_service=news_svc,
            knowledge_service=knowledge_svc,
            graph_service=graph_svc,
            gen_cfg=gen_cfg,
        )

    # ── classification ────────────────────────────────────────────────────

    def route_query(
        self,
        query: str,
        intent: Optional[str] = None,
    ) -> RouteResult:
        """
        Classify the query and return a RouteResult.
        intent — explicit override: "news" | "knowledge" | "both"
        """
        if intent in ("news", "knowledge", "both"):
            return RouteResult(
                route=intent,
                intent=intent,
                reasoning=f"Explicit intent override: {intent}",
            )

        q               = query.lower()
        draft_score     = sum(1 for kw in self._DRAFT_SIGNALS     if kw in q)
        news_score      = sum(1 for kw in self._NEWS_SIGNALS      if kw in q)
        knowledge_score = sum(1 for kw in self._KNOWLEDGE_SIGNALS if kw in q)

        if draft_score > 0:
            route     = "both"
            reasoning = (
                f"Drafting/authoring task detected "
                f"(draft={draft_score}, news={news_score}, knowledge={knowledge_score})"
            )
        elif news_score > 0 and knowledge_score > 0:
            route     = "both"
            reasoning = (
                f"Mixed signals — querying both collections "
                f"(news={news_score}, knowledge={knowledge_score})"
            )
        elif knowledge_score > news_score:
            route     = "knowledge"
            reasoning = (
                f"Institutional knowledge query "
                f"(knowledge={knowledge_score}, news={news_score})"
            )
        elif news_score > 0:
            route     = "news"
            reasoning = (
                f"News/event query "
                f"(news={news_score}, knowledge={knowledge_score})"
            )
        else:
            route     = "both"
            reasoning = "No strong signal — defaulting to both collections"

        return RouteResult(
            route=route,
            intent=intent or route,
            reasoning=reasoning,
        )

    # ── entity detection ──────────────────────────────────────────────────

    def _detect_graph_entities(self, query: str, extra_names: List[str] = None) -> List[Dict]:
        """
        Find graph entities whose names/aliases appear in the query.
        Also checks any extra_names list (e.g. names extracted from chunk payloads).
        Returns list of entity dicts.
        """
        if self._graph_svc is None:
            return []

        q         = query.lower()
        found_ids: set = set()
        results   = []

        # Scan graph name index against query
        name_index = self._graph_svc._graph["name_index"]
        for alias, entity_id in name_index.items():
            if len(alias) >= 3 and alias in q:
                if entity_id not in found_ids:
                    entity = self._graph_svc._graph["entity_index"].get(entity_id)
                    if entity:
                        found_ids.add(entity_id)
                        results.append(entity)

        # Also check explicitly supplied names (from chunk payloads)
        for name in (extra_names or []):
            if not name:
                continue
            entity = self._graph_svc.find_entity(name)
            if entity and entity["id"] not in found_ids:
                found_ids.add(entity["id"])
                results.append(entity)

        return results

    def _build_graph_context_blocks(self, entities: List[Dict]) -> List[str]:
        """Format graph entity profiles into compact context strings."""
        if not entities or self._graph_svc is None:
            return []
        blocks = []
        for ent in entities:
            block = self._graph_svc.build_context_block(ent.get("name", ""), max_rels=8)
            if block:
                blocks.append(block)
        return blocks

    # ── retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        use_knowledge: bool = True,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        intent: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Orchestrate retrieval across collections based on route classification.

        Args:
            query           — query string (also used for entity detection)
            filters         — optional dict with keys: category, year, quarter
            use_knowledge   — when False, only retrieve news regardless of route
            limit           — max news/knowledge chunks (editorial is always 2)
            score_threshold — minimum relevance score
            intent          — explicit route override: "news"|"knowledge"|"both"

        Returns:
            RetrievalResult with news_chunks, knowledge_chunks, editorial_chunks,
            graph_entities, and to_debug_dict().
        """
        filters = filters or {}

        # Resolve routing
        route_result = self.route_query(query, intent=intent)
        # use_knowledge=False forces news-only; True upgrades to at least "both"
        if not use_knowledge:
            effective_route = "news"
        elif route_result.route == "news":
            effective_route = "both"
        else:
            effective_route = route_result.route

        # Resolve limits from gen_cfg
        max_ctx = (
            self._gen_cfg.get("default_settings", {}).get("max_context_chunks")
            or self._gen_cfg.get("max_context_chunks", 8)
        )
        news_limit = limit or max_ctx
        know_limit = max(3, news_limit // 2)
        edit_limit = 2

        # Resolve threshold
        threshold = (
            score_threshold
            if score_threshold is not None
            else (
                self._gen_cfg.get("retrieval_rules", {}).get("score_threshold")
                or 0.0
            )
        )

        collections_searched: List[str] = []
        news_chunks:      list = []
        knowledge_chunks: list = []
        editorial_chunks: list = []

        # ── news lane ─────────────────────────────────────────────────────
        if effective_route in ("news", "both") and self._news_svc is not None:
            news_filter = self._news_svc.build_filter(
                category=filters.get("category"),
                year=filters.get("year"),
                quarter=filters.get("quarter"),
                published=True,
                status="approved",
                language="en",
                visibility="public",
            )
            news_chunks = self._news_svc.search(
                query_text=query,
                limit=news_limit,
                candidate_limit=news_limit * 3,
                score_threshold=threshold,
                query_filter=news_filter,
            )
            collections_searched.append("maharat_content_live")

        # ── knowledge lane ────────────────────────────────────────────────
        if (
            effective_route in ("knowledge", "both")
            and use_knowledge
            and self._knowledge_svc is not None
        ):
            know_filter = self._knowledge_svc.build_filter(
                knowledge_type=filters.get("knowledge_type"),
            )
            knowledge_chunks = self._knowledge_svc.search(
                query_text=query,
                limit=know_limit,
                candidate_limit=know_limit * 3,
                score_threshold=threshold,
                query_filter=know_filter,
            )
            collections_searched.append("maharat_knowledge_live")

            # ── editorial guidance lane (always retrieve for drafting) ──────
            if effective_route == "both" and self._knowledge_svc is not None:
                edit_filter = self._knowledge_svc.build_filter(
                    knowledge_type="editorial_guidelines",
                )
                editorial_chunks = self._knowledge_svc.search(
                    query_text=query,
                    limit=edit_limit,
                    candidate_limit=edit_limit * 4,
                    score_threshold=0.0,
                    query_filter=edit_filter,
                )
                # editorial already inside knowledge collection — don't re-add to list

        # ── graph entity detection ────────────────────────────────────────
        # Collect explicit entity names from chunk payloads
        extra_names: List[str] = []
        for pt in news_chunks + knowledge_chunks:
            p = pt.payload or {}
            for field_key in ("entities_organizations", "entities_programs",
                              "entities_people", "entities_credentials"):
                for name in (p.get(field_key) or []):
                    if name:
                        extra_names.append(name)

        graph_entities  = self._detect_graph_entities(query, extra_names=extra_names)
        context_blocks  = self._build_graph_context_blocks(graph_entities)

        return RetrievalResult(
            route=effective_route,
            route_reasoning=route_result.reasoning,
            collections_searched=collections_searched,
            news_chunks=news_chunks,
            knowledge_chunks=knowledge_chunks,
            editorial_chunks=editorial_chunks,
            graph_entities=graph_entities,
            graph_context_blocks=context_blocks,
        )
