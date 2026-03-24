"""
Generation service — calls Claude API with a grounded prompt and saves
structured draft output: draft.md, sources.json, retrieval_debug.json.

Claude is instructed to respond with a JSON object whose fields depend on
the generation mode (headline/summary/body/slug/seo_summary for website_news;
body/hashtags only for linkedin_post; etc.).

After generation:
  - JSON is parsed from the raw response
  - Editorial QA runs via StyleService (headline/summary/style guards)
  - Mode QA runs via quality_controls config (body word count, duplicate sentences)
  - draft.md is written as a clean formatted article
  - sources.json respects include_sources_used and include_token_usage flags
  - retrieval_debug.json goes to debug_dir (or draft folder if not configured)
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import anthropic

ROOT = Path(__file__).resolve().parent.parent


# ── config helpers ────────────────────────────────────────────────────────

def _get(cfg: dict, *keys, default=None):
    """Safe nested get: _get(cfg, 'a', 'b', default=0)."""
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k)
        if val is None:
            return default
    return val


# ── JSON extraction ───────────────────────────────────────────────────────

def _parse_draft_json(text: str) -> dict:
    """
    Extract the structured JSON object from Claude's response.
    Tries, in order:
      1. ```json ... ``` fenced block
      2. First { ... } object in the text
      3. Fallback — treat entire text as body, leave other fields empty
    """
    fallback = {
        "headline":       "",
        "summary":        "",
        "body":           text.strip(),
        "suggested_slug": "",
        "seo_summary":    "",
        "hashtags":       [],
    }

    # 1 — fenced block
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        try:
            parsed = json.loads(m.group(1))
            parsed.setdefault("hashtags", [])
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # 2 — raw JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            parsed.setdefault("hashtags", [])
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return fallback


# ── mode-level QA ─────────────────────────────────────────────────────────

def _run_mode_qa(
    draft: dict,
    mode_spec: Optional[dict],
    sources_used: List[dict],
    quality_controls: dict,
) -> List[str]:
    """
    Mode-specific and quality_controls QA checks.
    Returns list of warning strings.

    Checks driven by quality_controls config:
      check_summary_length      → body word count vs mode length bounds
      check_duplicate_sentences → repeated sentence fragments in body
      check_partner_consistency → body references at least one detected entity
      check_unsupported_claims  → flagged if no sources but body present
    """
    warnings = []
    body = (draft.get("body") or "").strip()

    # ── body word count against mode bounds ───────────────────────────────
    if quality_controls.get("check_summary_length") and mode_spec:
        length  = mode_spec.get("length", {})
        min_w   = length.get("min_words", 0)
        max_w   = length.get("max_words", 0)
        b_words = len(body.split()) if body else 0
        if min_w and b_words < min_w:
            warnings.append(
                f"QA [mode/length]: body too short "
                f"({b_words} words, min {min_w})."
            )
        if max_w and b_words > max_w:
            warnings.append(
                f"QA [mode/length]: body too long "
                f"({b_words} words, max {max_w})."
            )

    # ── duplicate sentences ────────────────────────────────────────────────
    if quality_controls.get("check_duplicate_sentences") and body:
        sentences = re.split(r"(?<=[.!?])\s+", body)
        seen: set = set()
        for s in sentences:
            norm = re.sub(r"\s+", " ", s.strip().lower())
            if len(norm) > 25:
                if norm in seen:
                    warnings.append(
                        "QA [quality]: duplicate sentence detected in body."
                    )
                    break
                seen.add(norm)

    # ── unsupported claims: body present but no sources ────────────────────
    if quality_controls.get("check_unsupported_claims") and body and not sources_used:
        warnings.append(
            "QA [grounding]: body text present but no sources retrieved — "
            "claims cannot be verified."
        )

    return warnings


# ── payload helpers ───────────────────────────────────────────────────────

def _extract_sources_used(chunks: list) -> List[Dict]:
    sources = []
    for point in chunks:
        p = point.payload or {}
        sources.append({
            "slug":        p.get("slug", ""),
            "chunk_id":    p.get("chunk_id", ""),
            "chunk_index": p.get("chunk_index", 0),
            "title":       p.get("title", ""),
            "date":        (p.get("date") or "")[:10],
            "category":    p.get("category", ""),
            "score":       round(float(point.score), 4),
            "source_document": p.get("source_document", ""),
        })
    return sources


def _build_retrieval_debug(
    draft_id: str,
    retrieval_context: dict,
    chunks: list,
) -> dict:
    chunk_records = []
    for i, point in enumerate(chunks):
        p = point.payload or {}
        chunk_records.append({
            "rank":        i + 1,
            "score":       round(float(point.score), 4),
            "slug":        p.get("slug", ""),
            "chunk_id":    p.get("chunk_id", ""),
            "chunk_index": p.get("chunk_index", 0),
            "chunk_type":  p.get("chunk_type", ""),
            "title":       p.get("title", ""),
            "date":        (p.get("date") or "")[:10],
            "category":    p.get("category", ""),
            "tags":        p.get("tags", []),
            "word_count":  p.get("word_count", 0),
            "chunk_text":  p.get("chunk_text", ""),
        })
    return {
        "draft_id":         draft_id,
        "query":            retrieval_context.get("query", ""),
        "generation_mode":  retrieval_context.get("generation_mode"),
        "article_type":     retrieval_context.get("article_type"),
        "filters":          retrieval_context.get("filters", {}),
        "chunks_retrieved": len(chunks),
        "chunks":           chunk_records,
    }


# ── DraftResult ───────────────────────────────────────────────────────────

class DraftResult:
    """Holds the full structured output of a single generation call."""

    def __init__(
        self,
        draft_id: str,
        draft_slug: str,
        topic: str,
        article_type: Optional[str],
        generation_mode: Optional[str],
        mode_spec: Optional[dict],
        # structured fields parsed from Claude's JSON response
        headline: str,
        summary: str,
        body: str,
        suggested_slug: str,
        seo_summary: str,
        hashtags: List[str],
        # raw response text (for audit)
        article_text: str,
        # retrieval
        sources_used: List[Dict],
        entities_detected: dict,
        retrieval_debug: Dict,
        # QA
        qa_warnings: List[str],
        # API metadata
        model: str,
        input_tokens: int,
        output_tokens: int,
        generated_at: str,
        generation_config: Dict,
        # output flags
        include_sources_used: bool,
        include_token_usage: bool,
    ):
        self.draft_id           = draft_id
        self.draft_slug         = draft_slug
        self.topic              = topic
        self.article_type       = article_type
        self.generation_mode    = generation_mode
        self.mode_spec          = mode_spec
        self.headline           = headline
        self.summary            = summary
        self.body               = body
        self.suggested_slug     = suggested_slug
        self.seo_summary        = seo_summary
        self.hashtags           = hashtags
        self.article_text       = article_text
        self.sources_used       = sources_used
        self.entities_detected  = entities_detected
        self.retrieval_debug    = retrieval_debug
        self.qa_warnings        = qa_warnings
        self.model              = model
        self.input_tokens       = input_tokens
        self.output_tokens      = output_tokens
        self.generated_at       = generated_at
        self.generation_config  = generation_config
        self.include_sources_used = include_sources_used
        self.include_token_usage  = include_token_usage

    @property
    def folder_name(self) -> str:
        return self.draft_slug

    # ── serialisers ───────────────────────────────────────────────────────

    def to_sources_dict(self) -> dict:
        """Content of sources.json — respects include_sources_used / include_token_usage."""
        d: dict = {
            "draft_id":       self.draft_id,
            "draft_slug":     self.draft_slug,
            "topic":          self.topic,
            "article_type":   self.article_type,
            "generation_mode": self.generation_mode,
            "headline":       self.headline,
            "summary":        self.summary,
            "suggested_slug": self.suggested_slug,
            "seo_summary":    self.seo_summary,
            "qa_warnings":    self.qa_warnings,
            "entities_detected": self.entities_detected,
            "generated_at":   self.generated_at,
            "generation_config": self.generation_config,
        }
        if self.include_token_usage:
            d["model"]         = self.model
            d["input_tokens"]  = self.input_tokens
            d["output_tokens"] = self.output_tokens
        if self.include_sources_used:
            d["sources_used"] = self.sources_used
        return d

    def to_debug_dict(self) -> dict:
        """Content of retrieval_debug.json."""
        return self.retrieval_debug

    def to_dict(self) -> dict:
        d = self.to_sources_dict()
        d["body"]            = self.body
        d["hashtags"]        = self.hashtags
        d["article_text"]    = self.article_text
        d["retrieval_debug"] = self.retrieval_debug
        return d

    def formatted_article(self) -> str:
        """
        Clean markdown representation of the structured draft.
        Respects mode structure flags (include_headline, include_summary,
        include_slug, include_seo_summary, include_hashtags).
        """
        structure = (self.mode_spec or {}).get("structure", {})
        # Defaults: include everything when no mode spec
        show_headline = structure.get("include_headline", True)
        show_summary  = structure.get("include_summary", True)
        show_slug     = structure.get("include_slug", True)
        show_seo      = structure.get("include_seo_summary", True)
        show_hashtags = structure.get("include_hashtags", False)

        parts = []

        if show_headline and self.headline:
            parts.append(f"# {self.headline}")
            parts.append("")

        if show_summary and self.summary:
            parts.append(f"**{self.summary}**")
            parts.append("")

        if self.body:
            parts.append(self.body)
            parts.append("")

        meta = []
        if show_slug and self.suggested_slug:
            meta.append(f"**Suggested slug:** `{self.suggested_slug}`")
        if show_seo and self.seo_summary:
            meta.append(f"**SEO summary:** {self.seo_summary}")
        if meta:
            parts.append("---")
            parts.extend(meta)
            parts.append("")

        if show_hashtags and self.hashtags:
            tag_str = "  ".join(
                h if h.startswith("#") else f"#{h}" for h in self.hashtags
            )
            parts.append(tag_str)

        return "\n".join(parts).rstrip()


# ── GenerationService ─────────────────────────────────────────────────────

class GenerationService:
    """
    Calls Claude API with a grounded prompt.
    Parses JSON response → structured draft fields.
    Runs editorial QA (via StyleService) and mode QA (via quality_controls config).
    Saves draft.md / sources.json / retrieval_debug.json.
    """

    def __init__(
        self,
        gen_cfg: dict,
        api_key: Optional[str] = None,
        drafts_dir: Optional[Path] = None,
        debug_dir: Optional[Path] = None,
        style_service=None,
    ):
        self.gen_cfg            = gen_cfg
        self.model      = gen_cfg.get("model", "claude-sonnet-4-6")
        self.max_tokens = gen_cfg.get("max_tokens", 4096)
        # thinking: pass to API only when explicitly enabled (type != "disabled" and not null)
        _thinking = gen_cfg.get("thinking") or {}
        self.thinking = (
            _thinking
            if isinstance(_thinking, dict) and _thinking.get("type") not in (None, "disabled")
            else None
        )
        self.require_grounding  = gen_cfg.get("require_grounding", True)
        self.style_service      = style_service
        self.quality_controls   = gen_cfg.get("quality_controls", {})

        # output config — support both output.* dict and legacy flat fields
        out = gen_cfg.get("output", {})
        self.save_drafts          = out.get("save_drafts", gen_cfg.get("save_drafts", True))
        self.include_sources_used = out.get("include_sources_used", True)
        self.include_token_usage  = out.get("include_token_usage", True)

        _drafts_root = ROOT / out.get("drafts_dir", gen_cfg.get("drafts_dir", "outputs/drafts"))
        self.drafts_dir = drafts_dir or _drafts_root
        self.drafts_dir.mkdir(parents=True, exist_ok=True)

        _debug_root = ROOT / out.get("debug_dir", "outputs/debug")
        self.debug_dir  = debug_dir or _debug_root
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    @classmethod
    def from_config(
        cls,
        gen_cfg: dict,
        api_key: Optional[str] = None,
        style_service=None,
    ) -> "GenerationService":
        return cls(gen_cfg=gen_cfg, api_key=api_key, style_service=style_service)

    # ── core generation ────────────────────────────────────────────────────

    def generate(
        self,
        topic: str,
        prompt_package: Dict[str, str],
        chunks: list,
        retrieval_context: Optional[Dict] = None,
        draft_slug: Optional[str] = None,
        article_type: Optional[str] = None,
        generation_mode: Optional[str] = None,
        mode_spec: Optional[dict] = None,
        entities_detected: Optional[dict] = None,
        stream: bool = True,
    ) -> DraftResult:
        """
        Call Claude and return a DraftResult with parsed structured fields.

        Args:
            topic:             The user's topic / request string.
            prompt_package:    {"system": str, "user": str} from prompt_service.
            chunks:            Retrieved ScoredPoint list.
            retrieval_context: Dict with query/filters for retrieval_debug.json.
            draft_slug:        Human-readable folder name (defaults to UUID).
            article_type:      Editorial article type (from editorial_style.yaml).
            generation_mode:   Mode key from generation.yaml generation_modes.
            mode_spec:         Resolved mode dict.
            entities_detected: Aggregated entities from retrieved chunks.
            stream:            Stream tokens to stdout while generating.
        """
        if self.require_grounding and not chunks:
            raise ValueError(
                "require_grounding=True but no chunks were retrieved. "
                "Broaden the query or lower score_threshold."
            )

        draft_id      = str(uuid.uuid4())
        article_parts = []

        api_kwargs: dict = {
            "model":    self.model,
            "max_tokens": self.max_tokens,
            "system":   prompt_package["system"],
            "messages": [{"role": "user", "content": prompt_package["user"]}],
        }
        if self.thinking is not None:
            api_kwargs["thinking"] = self.thinking

        if stream:
            with self._client.messages.stream(**api_kwargs) as stream_ctx:
                for text in stream_ctx.text_stream:
                    print(text, end="", flush=True)
                    article_parts.append(text)
            final         = stream_ctx.get_final_message()
            input_tokens  = final.usage.input_tokens
            output_tokens = final.usage.output_tokens
        else:
            response = self._client.messages.create(**api_kwargs)
            article_parts = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            input_tokens  = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        raw_text     = "".join(article_parts)
        parsed       = _parse_draft_json(raw_text)
        sources_used = _extract_sources_used(chunks)

        # ── QA ────────────────────────────────────────────────────────────
        qa_warnings: List[str] = []
        if self.style_service is not None:
            qa_warnings = self.style_service.run_qa(parsed, sources_used)
        qa_warnings += _run_mode_qa(
            parsed, mode_spec, sources_used, self.quality_controls
        )

        result = DraftResult(
            draft_id=draft_id,
            draft_slug=draft_slug or draft_id,
            topic=topic,
            article_type=article_type,
            generation_mode=generation_mode,
            mode_spec=mode_spec,
            headline=parsed.get("headline", ""),
            summary=parsed.get("summary", ""),
            body=parsed.get("body", ""),
            suggested_slug=parsed.get("suggested_slug", ""),
            seo_summary=parsed.get("seo_summary", ""),
            hashtags=parsed.get("hashtags") or [],
            article_text=raw_text,
            sources_used=sources_used,
            entities_detected=entities_detected or {},
            retrieval_debug=_build_retrieval_debug(
                draft_id,
                retrieval_context or {"query": topic},
                chunks,
            ),
            qa_warnings=qa_warnings,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generated_at=datetime.now(timezone.utc).isoformat(),
            generation_config={
                "model":           self.model,
                "max_tokens":      self.max_tokens,
                "require_grounding": self.require_grounding,
                "article_type":    article_type,
                "generation_mode": generation_mode,
            },
            include_sources_used=self.include_sources_used,
            include_token_usage=self.include_token_usage,
        )

        if self.save_drafts:
            self._save_draft(result)

        return result

    # ── output persistence ─────────────────────────────────────────────────

    def _save_draft(self, result: DraftResult) -> Path:
        """
        Save draft files:
          <drafts_dir>/<draft_slug>/draft.md     — clean formatted article
          <drafts_dir>/<draft_slug>/sources.json — metadata + QA + conditional fields
          <debug_dir>/<draft_slug>/retrieval_debug.json — raw chunk scores
        """
        draft_dir = self.drafts_dir / result.folder_name
        draft_dir.mkdir(parents=True, exist_ok=True)

        # draft.md — clean formatted output (mode-aware)
        (draft_dir / "draft.md").write_text(
            result.formatted_article(), encoding="utf-8"
        )

        # sources.json — respects include_sources_used / include_token_usage
        with open(draft_dir / "sources.json", "w", encoding="utf-8") as fh:
            json.dump(result.to_sources_dict(), fh, indent=2, ensure_ascii=False)

        # retrieval_debug.json — goes to debug_dir
        debug_dir = self.debug_dir / result.folder_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "retrieval_debug.json", "w", encoding="utf-8") as fh:
            json.dump(result.to_debug_dict(), fh, indent=2, ensure_ascii=False)

        return draft_dir
