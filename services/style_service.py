"""
Style service — loads config/editorial_style.yaml and provides:
  - Article-type-specific prompt instructions (structure, emphasis, purpose)
  - Voice, factual controls, style guards, headline/summary rules
  - Lightweight QA checks that run after generation
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT       = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

VALID_ARTICLE_TYPES = (
    "partnership_announcement",
    "graduation_story",
    "training_program_update",
    "event_story",
    "recognition_story",
    "leadership_news",
)


class StyleService:
    """
    Wraps editorial_style.yaml.
    Instantiate once; pass to prompt_service and generation_service.
    """

    def __init__(self, style_cfg: dict):
        # Support both a bare editorial_style dict and the file root dict
        self._cfg = style_cfg.get("editorial_style", style_cfg)

    @classmethod
    def from_config(cls) -> "StyleService":
        path = CONFIG_DIR / "editorial_style.yaml"
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(cfg)

    # ── raw accessors ──────────────────────────────────────────────────────

    def get_organization(self) -> dict:
        return self._cfg.get("organization", {})

    def get_article_type_spec(self, article_type: str) -> dict:
        types = self._cfg.get("article_types", {})
        if article_type not in types:
            raise ValueError(
                f"Unknown article type '{article_type}'. "
                f"Valid types: {', '.join(VALID_ARTICLE_TYPES)}"
            )
        return types[article_type]

    def get_voice(self) -> dict:
        return self._cfg.get("voice", {})

    def get_writing_principles(self) -> List[str]:
        return self._cfg.get("writing_principles", [])

    def get_headline_rules(self) -> dict:
        return self._cfg.get("headline_rules", {})

    def get_summary_rules(self) -> dict:
        return self._cfg.get("summary_rules", {})

    def get_body_rules(self) -> dict:
        return self._cfg.get("body_rules", {})

    def get_factual_controls(self) -> dict:
        return self._cfg.get("factual_controls", {})

    def get_style_guards(self) -> dict:
        return self._cfg.get("style_guards", {})

    def get_generation_rules(self) -> dict:
        return self._cfg.get("generation_rules", {})

    def get_quality_checks(self) -> dict:
        return self._cfg.get("quality_checks", {})

    def get_seo_rules(self) -> dict:
        return self._cfg.get("seo_rules", {})

    def get_partner_language(self) -> dict:
        return self._cfg.get("partner_language", {})

    # ── prompt-section builders ────────────────────────────────────────────

    def build_organization_context(self) -> str:
        org = self.get_organization()
        return (
            f"{org.get('full_name', 'Maharat Construction Training Center')} "
            f"({org.get('short_name', 'Maharat')}) — "
            f"{org.get('descriptor', '')}."
        )

    def build_voice_instructions(self) -> str:
        voice = self.get_voice()
        parts = []
        if voice.get("tone"):
            parts.append("Tone: " + ", ".join(voice["tone"]) + ".")
        if voice.get("style_characteristics"):
            parts.append("Style: " + ", ".join(voice["style_characteristics"]) + ".")
        if voice.get("avoid"):
            parts.append("Avoid: " + "; ".join(voice["avoid"]) + ".")
        return "\n".join(parts)

    def build_writing_principles(self) -> str:
        principles = self.get_writing_principles()
        return "\n".join(f"  • {p}" for p in principles)

    def build_factual_controls_instructions(self) -> str:
        fc = self.get_factual_controls()
        no_inv = fc.get("no_invention", [])
        lines = [
            "Factual controls:",
            "  • Base every claim on the SOURCE CHUNKS provided — nothing else.",
        ]
        if no_inv:
            lines.append(f"  • Do NOT invent: {', '.join(no_inv)}.")
        if fc.get("quote_policy"):
            lines.append(f"  • Quote policy: {fc['quote_policy']}")
        if fc.get("uncertainty_rule"):
            lines.append(f"  • Uncertainty rule: {fc['uncertainty_rule']}")
        if fc.get("conflict_rule"):
            lines.append(f"  • Conflict rule: {fc['conflict_rule']}")
        return "\n".join(lines)

    def build_style_guards_instructions(self) -> str:
        sg = self.get_style_guards()
        lines = []
        if sg.get("banned_phrases"):
            banned = "; ".join(f'"{p}"' for p in sg["banned_phrases"])
            lines.append(f"BANNED phrases (never use): {banned}.")
        if sg.get("discouraged_phrases"):
            disc = "; ".join(f'"{p}"' for p in sg["discouraged_phrases"])
            lines.append(f"Discouraged phrases: {disc}.")
        if sg.get("preferred_replacements"):
            pairs = "; ".join(
                f'"{k}" → "{v}"'
                for k, v in sg["preferred_replacements"].items()
            )
            lines.append(f"Preferred replacements: {pairs}.")
        return "\n".join(lines)

    def build_article_type_instructions(self, article_type: str) -> str:
        spec      = self.get_article_type_spec(article_type)
        label     = article_type.replace("_", " ").title()
        purpose   = spec.get("purpose", "")
        emphasis  = spec.get("emphasis", [])
        structure = spec.get("preferred_structure", [])
        lines = [f"Article type: {label}", f"Purpose: {purpose}"]
        if emphasis:
            lines.append("Emphasise: " + "; ".join(emphasis) + ".")
        if structure:
            lines.append("Structure: " + " → ".join(structure) + ".")
        return "\n".join(lines)

    def build_headline_instructions(self) -> str:
        hr      = self.get_headline_rules()
        length  = hr.get("length", {})
        avoid   = hr.get("avoid", [])
        patterns = hr.get("preferred_patterns", [])
        lines = [
            f"headline: {length.get('min_words', 5)}–{length.get('max_words', 14)} words.",
        ]
        if avoid:
            lines.append("  Avoid: " + "; ".join(avoid) + ".")
        if patterns:
            lines.append("  Preferred patterns: " + " | ".join(f'"{p}"' for p in patterns))
        return "\n".join(lines)

    def build_summary_instructions(self) -> str:
        sr = self.get_summary_rules()
        incl = ", ".join(sr.get("should_include", []))
        avoid = ", ".join(sr.get("avoid", []))
        return (
            f"summary: {sr.get('min_words', 18)}–{sr.get('max_words', 45)} words. "
            f"Include: {incl}. Avoid: {avoid}."
        )

    def build_seo_instructions(self) -> str:
        seo      = self.get_seo_rules()
        slug_r   = seo.get("slug", {})
        seo_sum  = seo.get("seo_summary", {})
        incl     = ", ".join(seo_sum.get("should_include", ["Maharat", "main action"]))
        return (
            f"suggested_slug: lowercase, hyphen-separated, max {slug_r.get('max_length', 80)} chars, "
            f"derived from the headline.\n"
            f"seo_summary: ≤{seo_sum.get('max_words', 30)} words; include {incl}."
        )

    # ── QA ─────────────────────────────────────────────────────────────────

    def run_qa(self, draft: dict, sources_used: List[dict]) -> List[str]:
        """
        Lightweight post-generation quality checks.
        Returns a list of warning strings — empty list means all checks passed.

        Checks:
          - headline: word count bounds, question headline, banned phrases
          - summary: word count bounds, identical to headline
          - body + full draft: banned and discouraged phrases
          - sources: minimum count
        """
        warnings = []
        hr  = self.get_headline_rules()
        sr  = self.get_summary_rules()
        sg  = self.get_style_guards()
        qc  = self.get_quality_checks()

        headline = (draft.get("headline") or "").strip()
        summary  = (draft.get("summary")  or "").strip()
        body     = (draft.get("body")     or "").strip()

        # ── headline ──────────────────────────────────────────────────────
        if not headline:
            warnings.append("QA [headline]: missing.")
        else:
            h_words = len(headline.split())
            h_min   = hr.get("length", {}).get("min_words", 5)
            h_max   = hr.get("length", {}).get("max_words", 14)
            if h_words < h_min:
                warnings.append(
                    f"QA [headline]: too short ({h_words} words, min {h_min})."
                )
            if h_words > h_max:
                warnings.append(
                    f"QA [headline]: too long ({h_words} words, max {h_max})."
                )
            if headline.rstrip().endswith("?"):
                warnings.append("QA [headline]: is a question — avoid question headlines.")
            for phrase in sg.get("banned_phrases", []):
                if phrase.lower() in headline.lower():
                    warnings.append(f'QA [headline]: banned phrase — "{phrase}".')

        # ── summary ───────────────────────────────────────────────────────
        if not summary:
            warnings.append("QA [summary]: missing.")
        else:
            s_words = len(summary.split())
            s_min   = sr.get("min_words", 18)
            s_max   = sr.get("max_words", 45)
            if s_words < s_min:
                warnings.append(
                    f"QA [summary]: too short ({s_words} words, min {s_min})."
                )
            if s_words > s_max:
                warnings.append(
                    f"QA [summary]: too long ({s_words} words, max {s_max})."
                )
            if (headline and
                    summary.strip().rstrip(".") == headline.strip().rstrip(".")):
                warnings.append("QA [summary]: identical to headline.")

        # ── banned / discouraged phrases across full draft ─────────────
        full_text = " ".join([headline, summary, body]).lower()
        for phrase in sg.get("banned_phrases", []):
            if phrase.lower() in full_text:
                warnings.append(f'QA [style]: banned phrase in draft — "{phrase}".')
        for phrase in sg.get("discouraged_phrases", []):
            if phrase.lower() in full_text:
                warnings.append(f'QA [style]: discouraged phrase in draft — "{phrase}".')

        # ── sources ───────────────────────────────────────────────────────
        min_sources = qc.get("sources", {}).get("minimum_sources", 1)
        if len(sources_used) < min_sources:
            warnings.append(
                f"QA [sources]: fewer than {min_sources} source(s) "
                f"(got {len(sources_used)})."
            )

        # ── unsupported claims heuristic ──────────────────────────────────
        # Flag if body is present but sources list is empty — can't verify grounding
        if body and not sources_used:
            warnings.append(
                "QA [grounding]: body text present but no sources retrieved — "
                "claims cannot be verified."
            )

        return warnings
