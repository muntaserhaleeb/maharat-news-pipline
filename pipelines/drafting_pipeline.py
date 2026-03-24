"""
Drafting pipeline — request → retrieve → prompt → generate → save.
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
    Used when entity_usage.include_entity_context_in_prompt is true.
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
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_service = generation_service
        self.gen_cfg            = gen_cfg
        self.style_service      = style_service

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
        )

    def _resolve_mode(
        self, generation_mode: Optional[str]
    ):
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
    ) -> Optional[DraftResult]:
        """
        Full RAG drafting pipeline.

        1. Resolve generation mode from generation.yaml generation_modes.
        2. Retrieve relevant chunks (applying retrieval_rules defaults).
        3. Aggregate entities from chunk payloads.
        4. Build prompt (StyleService + mode + entity context + grounding rules).
        5. If dry_run: print chunks and return None.
        6. Generate via GenerationService (JSON → structured fields + QA).
        7. Print QA warnings, headline/slug, citations, save path to stderr.
        8. Return DraftResult.
        """
        # step 1 — resolve mode spec
        mode_name, mode_spec = self._resolve_mode(generation_mode)

        # validate article_type if given
        if article_type and article_type not in VALID_ARTICLE_TYPES:
            raise ValueError(
                f"Unknown article type '{article_type}'. "
                f"Valid types: {', '.join(VALID_ARTICLE_TYPES)}"
            )

        # step 2 — retrieve
        # Apply retrieval_rules defaults from generation.yaml
        retrieval_rules  = self.gen_cfg.get("retrieval_rules", {})
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
        print(
            f"Retrieved {len(chunks)} chunk(s) for: \"{topic}\"",
            file=sys.stderr,
        )

        # step 3 — dry run
        if dry_run:
            context = prompt_service.format_chunks_as_context(chunks)
            print("\n" + "\u2500" * 72)
            print("SOURCE CHUNKS (dry-run \u2014 Claude not called)")
            print("\u2500" * 72)
            print(context)
            return None

        if not chunks:
            print(
                "\nNo relevant chunks found. "
                "Try a broader topic or lower score_threshold.",
                file=sys.stderr,
            )
            return None

        # step 4 — aggregate entities (used when entity_usage config requests it)
        entity_usage       = self.gen_cfg.get("entity_usage", {})
        entities_detected: Dict[str, List[str]] = {}
        if entity_usage.get("detect_entities_from_query") or \
                entity_usage.get("include_entity_context_in_prompt"):
            entities_detected = _aggregate_entities_from_chunks(chunks)

        # step 5 — build prompt
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
        retrieval_context = {
            "query":           topic,
            "generation_mode": generation_mode,
            "article_type":    article_type,
            "filters": {
                "category": category,
                "year":     year,
            },
        }
        draft_slug = _topic_to_slug(topic, year=year)

        mode_label = f" [{generation_mode}]" if generation_mode else ""
        print(
            f"\nDrafting with {self.generation_service.model}{mode_label}\u2026\n",
            file=sys.stderr,
        )
        print("\u2500" * 72, file=sys.stderr)

        result = self.generation_service.generate(
            topic=topic,
            prompt_package=prompt_package,
            chunks=chunks,
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
                f"Tokens \u2014 input: {result.input_tokens}  "
                f"output: {result.output_tokens}",
                file=sys.stderr,
            )

        # QA
        if result.qa_warnings:
            print("\nQA Warnings:", file=sys.stderr)
            for w in result.qa_warnings:
                print(f"  \u26a0  {w}", file=sys.stderr)
        else:
            print("\n\u2713 QA passed.", file=sys.stderr)

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
            f"\nDraft saved \u2192 {self.generation_service.drafts_dir / result.folder_name}/",
            file=sys.stderr,
        )

        return result
