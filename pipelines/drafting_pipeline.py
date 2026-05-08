"""
Drafting pipeline — request -> retrieve -> prompt -> generate -> save.
Imports from services and retrieval_pipeline only.
"""

import re
import sys
from typing import Dict, List, Optional

import services.citation_service as citation_service
import services.prompt_service as prompt_service
from services.config_service import (
    load_editorial_style_config,
    load_generation_config,
    load_qdrant_config,
)
from services.generation_service import DraftResult, GenerationService
from services.memory_router import MemoryRouter
from services.retrieval_service import RetrievalService
from services.style_service import StyleService, VALID_ARTICLE_TYPES
from pipelines.retrieval_pipeline import RetrievalPipeline


def _topic_to_slug(topic: str, year: Optional[int] = None) -> str:
    """Convert a topic string to a safe folder name."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    if len(slug) > 50:
        slug = slug[:50].rsplit("-", 1)[0]
    if year:
        slug = f"{slug}-{year}"
    return slug


def _aggregate_entities_from_chunks(chunks: list) -> Dict[str, List[str]]:
    """
    Collect and deduplicate entity values from retrieved chunk payloads.
    Returns {entity_type: sorted_unique_list}.
    """
    buckets: Dict[str, set] = {
        "organizations": set(),
        "programs":      set(),
        "credentials":   set(),
        "locations":     set(),
        "people":        set(),
    }
    field_map = {
        "entities_organizations": "organizations",
        "entities_programs":      "programs",
        "entities_credentials":   "credentials",
        "entities_locations":     "locations",
        "entities_people":        "people",
    }
    for point in chunks:
        p = point.payload or {}
        for field, etype in field_map.items():
            vals = p.get(field) or []
            if isinstance(vals, list):
                buckets[etype].update(vals)
    return {k: sorted(v) for k, v in buckets.items()}


class DraftingPipeline:
    """Orchestrates the full RAG drafting flow."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        generation_service: GenerationService,
        gen_cfg: dict,
        style_service: Optional[StyleService] = None,
        knowledge_retrieval_pipeline: Optional[RetrievalPipeline] = None,
    ):
        self.retrieval_pipeline           = retrieval_pipeline
        self.generation_service           = generation_service
        self.gen_cfg                      = gen_cfg
        self.style_service                = style_service
        self.knowledge_retrieval_pipeline = knowledge_retrieval_pipeline
        self._router                      = MemoryRouter()

    @classmethod
    def from_config(
        cls,
        qdrant_cfg: Optional[dict] = None,
        gen_cfg: Optional[dict] = None,
        api_key: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> "DraftingPipeline":
        if qdrant_cfg is None:
            qdrant_cfg = load_qdrant_config()
        if gen_cfg is None:
            gen_cfg = load_generation_config().get("generation", {})
        if model_override:
            gen_cfg = dict(gen_cfg)
            gen_cfg["model"] = model_override

        style_service = StyleService.from_config()

        # Build optional knowledge retrieval pipeline if collection is configured
        knowledge_pipeline = None
        if qdrant_cfg.get("collections", {}).get("knowledge"):
            try:
                knowledge_pipeline = RetrievalPipeline(
                    retrieval_service=RetrievalService.from_config(
                        qdrant_cfg, collection_key="knowledge"
                    ),
                    retrieval_cfg=qdrant_cfg.get("retrieval", {}),
                    gen_cfg=gen_cfg,
                )
            except Exception as exc:
                print(
                    f"[warn] Could not init knowledge retrieval pipeline: {exc}",
                    file=sys.stderr,
                )

        return cls(
            retrieval_pipeline=RetrievalPipeline.from_config(
                qdrant_cfg=qdrant_cfg, gen_cfg=gen_cfg
            ),
            generation_service=GenerationService.from_config(
                gen_cfg=gen_cfg,
                api_key=api_key,
                style_service=style_service,
            ),
            gen_cfg=gen_cfg,
            style_service=style_service,
            knowledge_retrieval_pipeline=knowledge_pipeline,
        )

    def _resolve_mode(self, generation_mode: Optional[str]):
        """
        Look up the mode spec from generation.yaml generation_modes.
        Returns (mode_name, mode_spec) or (None, None).
        """
        if not generation_mode:
            return None, None
        modes = self.gen_cfg.get("generation_modes", {})
        spec  = modes.get(generation_mode)
        if spec is None:
            valid = ", ".join(modes.keys()) if modes else "none configured"
            raise ValueError(
                f"Unknown generation mode '{generation_mode}'. "
                f"Valid modes: {valid}"
            )
        return generation_mode, spec

    def draft(
        self,
        topic: str,
        generation_mode: Optional[str] = None,
        article_type: Optional[str] = None,
        category: Optional[str] = None,
        year: Optional[int] = None,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        dry_run: bool = False,
        stream: bool = True,
        use_knowledge: bool = False,
    ) -> Optional[DraftResult]:
        """
        Full RAG drafting pipeline.

        1. Resolve generation mode from generation.yaml generation_modes.
        2. Retrieve news chunks; optionally also retrieve knowledge chunks.
        3. Aggregate entities from all retrieved chunks.
        4. Build single or dual prompt depending on use_knowledge.
        5. If dry_run: print chunks and return None.
        6. Generate via GenerationService (JSON -> structured fields + QA).
        7. Print QA warnings, headline/slug, citations, save path to stderr.
        8. Return DraftResult.
        """
        # step 1 — resolve mode
        mode_name, mode_spec = self._resolve_mode(generation_mode)

        if article_type and article_type not in VALID_ARTICLE_TYPES:
            raise ValueError(
                f"Unknown article type '{article_type}'. "
                f"Valid types: {', '.join(VALID_ARTICLE_TYPES)}"
            )

        # step 2 — retrieve news chunks
        retrieval_rules     = self.gen_cfg.get("retrieval_rules", {})
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else retrieval_rules.get("score_threshold")
        )
        chunks = self.retrieval_pipeline.retrieve(
            query=topic,
            limit=limit,
            category=category,
            year=year,
            score_threshold=effective_threshold,
        )
        print(f"Retrieved {len(chunks)} news chunk(s) for: \"{topic}\"", file=sys.stderr)

        # step 2b — optionally retrieve knowledge chunks
        knowledge_chunks: list = []
        if use_knowledge:
            if self.knowledge_retrieval_pipeline is None:
                print(
                    "[warn] --use-knowledge requested but knowledge pipeline is "
                    "unavailable. Run: python app/cli.py ingest-knowledge",
                    file=sys.stderr,
                )
            else:
                k_limit = max(4, (limit or 8) // 2)
                knowledge_chunks = self.knowledge_retrieval_pipeline.retrieve(
                    query=topic,
                    limit=k_limit,
                    score_threshold=effective_threshold,
                    apply_prefilters=False,
                )
                print(
                    f"Retrieved {len(knowledge_chunks)} knowledge chunk(s)",
                    file=sys.stderr,
                )

        # step 3 — dry run
        if dry_run:
            print("\n" + "─" * 72)
            print("NEWS CHUNKS (dry-run — Claude not called)")
            print("─" * 72)
            print(prompt_service.format_chunks_as_context(chunks))
            if knowledge_chunks:
                print("\n" + "─" * 72)
                print("KNOWLEDGE CHUNKS")
                print("─" * 72)
                print(prompt_service.format_knowledge_chunks_as_context(knowledge_chunks))
            return None

        if not chunks and not knowledge_chunks:
            print(
                "\nNo relevant chunks found. "
                "Try a broader topic or lower score_threshold.",
                file=sys.stderr,
            )
            return None

        # step 4 — aggregate entities
        entity_usage       = self.gen_cfg.get("entity_usage", {})
        entities_detected: Dict[str, List[str]] = {}
        if entity_usage.get("detect_entities_from_query") or \
                entity_usage.get("include_entity_context_in_prompt"):
            entities_detected = _aggregate_entities_from_chunks(
                chunks + knowledge_chunks
            )

        # step 5 — build prompt (single or dual)
        if use_knowledge and knowledge_chunks:
            prompt_package = prompt_service.build_prompt_package_dual(
                topic=topic,
                news_chunks=chunks,
                knowledge_chunks=knowledge_chunks,
                gen_cfg=self.gen_cfg,
                style_service=self.style_service,
                article_type=article_type,
                mode_name=mode_name,
                mode_spec=mode_spec,
                entities_detected=entities_detected if entities_detected else None,
            )
        else:
            prompt_package = prompt_service.build_prompt_package(
                topic=topic,
                chunks=chunks,
                gen_cfg=self.gen_cfg,
                style_service=self.style_service,
                article_type=article_type,
                mode_name=mode_name,
                mode_spec=mode_spec,
                entities_detected=entities_detected if entities_detected else None,
            )

        # step 6 — generate
        all_chunks = chunks + knowledge_chunks
        retrieval_context = {
            "query":            topic,
            "generation_mode":  generation_mode,
            "article_type":     article_type,
            "use_knowledge":    use_knowledge,
            "news_chunks":      len(chunks),
            "knowledge_chunks": len(knowledge_chunks),
            "filters": {
                "category": category,
                "year":     year,
            },
        }
        draft_slug = _topic_to_slug(topic, year=year)

        mode_label = f" [{generation_mode}]" if generation_mode else ""
        print(
            f"\nDrafting with {self.generation_service.model}{mode_label}…\n",
            file=sys.stderr,
        )
        print("─" * 72, file=sys.stderr)

        result = self.generation_service.generate(
            topic=topic,
            prompt_package=prompt_package,
            chunks=all_chunks,
            retrieval_context=retrieval_context,
            draft_slug=draft_slug,
            article_type=article_type,
            generation_mode=generation_mode,
            mode_spec=mode_spec,
            entities_detected=entities_detected,
            stream=stream,
        )

        # step 7 — post-generation output to stderr
        print(f"\n\n{'─' * 72}", file=sys.stderr)
        if result.include_token_usage:
            print(
                f"Tokens — input: {result.input_tokens}  "
                f"output: {result.output_tokens}",
                file=sys.stderr,
            )

        # QA
        if result.qa_warnings:
            print("\nQA Warnings:", file=sys.stderr)
            for w in result.qa_warnings:
                print(f"  ⚠  {w}", file=sys.stderr)
        else:
            print("\n✓ QA passed.", file=sys.stderr)

        # structured preview
        if result.headline:
            print(f"\nHeadline : {result.headline}", file=sys.stderr)
        if result.suggested_slug:
            print(f"Slug     : {result.suggested_slug}", file=sys.stderr)
        if result.seo_summary:
            print(f"SEO      : {result.seo_summary}", file=sys.stderr)

        # citations
        if result.include_sources_used:
            sources_block = citation_service.format_sources_block(result.sources_used)
            if sources_block:
                print(sources_block, file=sys.stderr)

        print(
            f"\nDraft saved → {self.generation_service.drafts_dir / result.folder_name}/",
            file=sys.stderr,
        )

        return result
