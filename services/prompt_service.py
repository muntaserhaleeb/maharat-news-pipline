"""
Prompt service — system prompt templates and context formatting.
When a StyleService is supplied the system prompt is built dynamically
from editorial_style.yaml; otherwise it falls back to the embedded default.

generation.yaml drives:
  - generation_mode  — length, structure, required JSON fields, tone adjustments
  - grounding_rules  — what Claude must never invent
  - entity_usage     — whether to include entity context in the user message
"""

from typing import Dict, List, Optional

# ── legacy embedded default (fallback only) ───────────────────────────────

_DEFAULT_SYSTEM_PROMPT = """\
You are a professional news writer for Maharat Construction Training Center (MCTC),
a vocational training institution in Saudi Arabia that prepares trainees for careers
in construction, welding, pipefitting, scaffolding, instrumentation, and related trades.

Draft a polished, factual news article grounded exclusively in the SOURCE CHUNKS
provided. Respond ONLY with a JSON object — no markdown fences, no commentary.

Required JSON fields:
  "headline"       — article headline
  "summary"        — 18–45 word standfirst
  "body"           — full article body (plain paragraphs, no markdown headings)
  "suggested_slug" — lowercase, hyphen-separated URL slug derived from the headline
  "seo_summary"    — ≤30 words for meta description

Do NOT invent facts, names, dates, or statistics not present in the source material.\
"""


# ── dynamic system prompt ─────────────────────────────────────────────────

def build_system_prompt(
    style_service,
    article_type: Optional[str] = None,
    grounding_rules: Optional[dict] = None,
) -> str:
    """
    Build a full system prompt from editorial_style.yaml via StyleService.
    grounding_rules, if supplied, comes from generation.yaml and is merged with
    the editorial style factual controls.
    """
    org = style_service.build_organization_context()

    sections = [
        f"You are a professional news writer for {org}",
        "",
        "─── Voice & Style ───────────────────────────────────────────────",
        style_service.build_voice_instructions(),
        "",
        "─── Writing Principles ──────────────────────────────────────────",
        style_service.build_writing_principles(),
        "",
        "─── Factual Controls ────────────────────────────────────────────",
        style_service.build_factual_controls_instructions(),
    ]

    # Merge grounding_rules from generation.yaml
    if grounding_rules:
        gr_section = _build_grounding_rules_section(grounding_rules)
        if gr_section:
            sections += ["", gr_section]

    sections += [
        "",
        "─── Style Guards ────────────────────────────────────────────────",
        style_service.build_style_guards_instructions(),
    ]

    if article_type:
        sections += [
            "",
            "─── Article Type ────────────────────────────────────────────────",
            style_service.build_article_type_instructions(article_type),
        ]

    sections += [
        "",
        "─── Partner & Programme Language ────────────────────────────────",
        "Partner language: "
        + style_service.get_partner_language().get(
            "first_mention_rule",
            "Use full official partner name on first mention.",
        ),
    ]

    return "\n".join(sections)


def _build_grounding_rules_section(grounding_rules: dict) -> str:
    """Format generation.yaml grounding_rules into a prompt section."""
    lines = ["Grounding rules (from generation config):"]
    if grounding_rules.get("require_source_reference"):
        lines.append("  • Every factual claim must be traceable to a SOURCE CHUNK.")
    prohibited = grounding_rules.get("prohibit_invention", [])
    if prohibited:
        items = ", ".join(str(p).replace("_", " ") for p in prohibited)
        lines.append(f"  • Never invent: {items}.")
    citation_fmt = grounding_rules.get("citation_format")
    if citation_fmt:
        lines.append(f"  • Citation format: {citation_fmt}.")
    return "\n".join(lines)


# ── mode instructions ─────────────────────────────────────────────────────

def build_mode_instructions(mode_name: str, mode_spec: dict) -> str:
    """
    Convert a generation_modes entry from generation.yaml into prompt text
    that tells Claude the length, tone, formatting, and required JSON fields.
    """
    desc        = mode_spec.get("description", mode_name.replace("_", " ").title())
    length      = mode_spec.get("length", {})
    structure   = mode_spec.get("structure", {})
    tone_adj    = mode_spec.get("tone_adjustment", [])
    formatting  = mode_spec.get("formatting", {})

    lines = [
        f"─── Generation Mode: {desc} ─────────────────────────────────────",
    ]

    if tone_adj:
        lines.append("Tone adjustments for this mode: " + ", ".join(tone_adj) + ".")

    target  = length.get("target_words")
    min_w   = length.get("min_words")
    max_w   = length.get("max_words")
    if target or min_w or max_w:
        parts = []
        if target:
            parts.append(f"target ~{target} words")
        if min_w:
            parts.append(f"min {min_w}")
        if max_w:
            parts.append(f"max {max_w}")
        lines.append("Body length: " + ", ".join(parts) + ".")

    # paragraph style
    para_style = formatting.get("paragraphs")
    if para_style:
        lines.append(f"Paragraphs: {para_style}.")

    # subheadings
    if formatting.get("subheadings"):
        lines.append(
            "Use ## markdown subheadings to organise the body "
            "(background, context, significance, future outlook)."
        )

    # additional sections for magazine
    extra_sections = mode_spec.get("additional_sections", [])
    if extra_sections:
        lines.append(
            "Include additional sections: " + ", ".join(extra_sections) + "."
        )

    # required JSON output fields
    required_fields = _build_required_fields(structure, mode_spec)
    lines.append("")
    lines.append("Required JSON output fields for this mode:")
    lines.extend(f"  {f}" for f in required_fields)

    # hashtags spec for LinkedIn
    hashtags_cfg = mode_spec.get("hashtags", {})
    if structure.get("include_hashtags") and hashtags_cfg.get("auto_generate"):
        base_tags = hashtags_cfg.get("base_tags", [])
        lines.append(
            "  \"hashtags\": array of strings; always include base tags: "
            + ", ".join(f"#{t}" for t in base_tags)
            + "; add 2–4 topic-specific tags."
        )

    return "\n".join(lines)


def _build_required_fields(structure: dict, mode_spec: dict) -> List[str]:
    """Return list of field descriptions for the JSON output instruction."""
    fields = []
    if structure.get("include_headline", True):
        fields.append('"headline"       — article headline (5–14 words)')
    if structure.get("include_summary", True):
        fields.append('"summary"        — 18–45 word standfirst')
    fields.append('"body"           — article body text')
    if structure.get("include_slug", True):
        fields.append('"suggested_slug" — lowercase hyphen-separated slug (max 80 chars)')
    if structure.get("include_seo_summary", True):
        fields.append('"seo_summary"    — ≤30 word meta description including "Maharat"')
    if structure.get("include_hashtags", False):
        fields.append('"hashtags"       — array of hashtag strings (see below)')
    return fields


# ── entity context ────────────────────────────────────────────────────────

def build_entity_context(entities_detected: dict) -> str:
    """
    Format aggregated entities from retrieved chunks into a prompt section.
    entities_detected: {entity_type: [canonical_name, ...]}
    """
    lines = ["─── Detected Entities (from retrieved sources) ──────────────────"]
    any_found = False
    type_labels = {
        "organizations": "Organizations",
        "programs":      "Programs",
        "credentials":   "Credentials",
        "locations":     "Locations",
        "people":        "People",
    }
    for etype, label in type_labels.items():
        names = entities_detected.get(etype, [])
        if names:
            lines.append(f"  {label}: {', '.join(names)}")
            any_found = True
    if not any_found:
        return ""
    lines.append(
        "Use full official names on first mention. "
        "Do not introduce entity names not present in the source chunks."
    )
    return "\n".join(lines)


# ── context formatting ────────────────────────────────────────────────────

def format_chunks_as_context(chunks: list) -> str:
    """Format a list of ScoredPoint objects into a SOURCE CHUNKS block."""
    lines = []
    for i, point in enumerate(chunks, 1):
        p = point.payload or {}
        lines.append(f"--- Chunk {i} (score={point.score:.4f}) ---")
        if p.get("title"):
            lines.append(f"Title   : {p['title']}")
        if p.get("date"):
            lines.append(f"Date    : {p['date'][:10]}")
        if p.get("category"):
            lines.append(f"Category: {p['category']}")
        if p.get("tags"):
            lines.append(f"Tags    : {', '.join(p['tags'])}")
        lines.append(f"Text    :\n{p.get('chunk_text', '').strip()}")
        lines.append("")
    return "\n".join(lines)


# ── user message ──────────────────────────────────────────────────────────

def build_user_message(
    topic: str,
    context_block: str,
    article_type: Optional[str] = None,
    mode_spec: Optional[dict] = None,
    mode_name: Optional[str] = None,
    entities_detected: Optional[dict] = None,
    include_entity_context: bool = False,
) -> str:
    type_hint = (
        f" (article type: {article_type.replace('_', ' ')})"
        if article_type else ""
    )
    mode_hint = (
        f" (mode: {mode_name.replace('_', ' ')})"
        if mode_name else ""
    )

    parts = [
        f"Draft a professional Maharat news article about: "
        f"{topic}{type_hint}{mode_hint}",
        "",
    ]

    # Entity context (from generation.yaml entity_usage.include_entity_context_in_prompt)
    if include_entity_context and entities_detected:
        entity_block = build_entity_context(entities_detected)
        if entity_block:
            parts += [entity_block, ""]

    # Mode-specific output instructions
    if mode_spec and mode_name:
        parts += [build_mode_instructions(mode_name, mode_spec), ""]
    else:
        # fallback output spec
        parts += [
            "Respond with a JSON object containing: "
            "headline, summary, body, suggested_slug, seo_summary.",
            "",
        ]

    parts += ["SOURCE CHUNKS:", "", context_block]
    return "\n".join(parts)


# ── package builder ───────────────────────────────────────────────────────

def build_prompt_package(
    topic: str,
    chunks: list,
    gen_cfg: dict,
    style_service=None,
    article_type: Optional[str] = None,
    mode_name: Optional[str] = None,
    mode_spec: Optional[dict] = None,
    entities_detected: Optional[dict] = None,
) -> Dict[str, str]:
    """
    Return {"system": str, "user": str} ready to pass to the Claude API.

    style_service  — builds system prompt from editorial_style.yaml
    gen_cfg        — provides grounding_rules, entity_usage config
    mode_name      — generation mode key (e.g. "website_news")
    mode_spec      — resolved mode dict from gen_cfg["generation_modes"]
    entities_detected — aggregated from retrieved chunks
    """
    grounding_rules = gen_cfg.get("grounding_rules")
    entity_usage    = gen_cfg.get("entity_usage", {})
    include_entity  = (
        entity_usage.get("include_entity_context_in_prompt", False)
        and bool(entities_detected)
    )

    if style_service is not None:
        system = build_system_prompt(
            style_service,
            article_type=article_type,
            grounding_rules=grounding_rules,
        )
    else:
        system = _DEFAULT_SYSTEM_PROMPT

    context = format_chunks_as_context(chunks)
    user    = build_user_message(
        topic=topic,
        context_block=context,
        article_type=article_type,
        mode_spec=mode_spec,
        mode_name=mode_name,
        entities_detected=entities_detected,
        include_entity_context=include_entity,
    )
    return {"system": system, "user": user}


# ── legacy accessor ───────────────────────────────────────────────────────

def get_system_prompt(template_name: str = "maharat_news_v1") -> str:
    return _DEFAULT_SYSTEM_PROMPT
