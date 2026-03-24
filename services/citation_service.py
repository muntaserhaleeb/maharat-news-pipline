"""
Citation service — formats source references from DraftResult.sources_used.
Pure string utilities; no external dependencies.
"""

from typing import Dict, List


def format_inline_citations(sources_used: List[Dict]) -> str:
    """
    Return a numbered reference list:
      [1] Title (slug) — date — category  score=0.xxxx
    """
    lines = []
    for i, src in enumerate(sources_used, 1):
        date_part = f" \u2014 {src['date']}" if src.get("date") else ""
        cat_part  = f" \u2014 {src['category']}" if src.get("category") else ""
        lines.append(
            f"[{i}] {src.get('title', src.get('slug', '?'))}"
            f" ({src.get('slug', '')})"
            f"{date_part}{cat_part}"
            f"  score={src.get('score', 0.0):.4f}"
        )
    return "\n".join(lines)


def format_sources_block(sources_used: List[Dict]) -> str:
    """
    Return a Markdown Sources section suitable for appending to a draft.
    """
    if not sources_used:
        return ""
    ref_list = format_inline_citations(sources_used)
    return f"\n\n## Sources\n\n{ref_list}\n"


def unique_slugs(sources_used: List[Dict]) -> List[str]:
    """Return deduplicated list of slugs in order of first appearance."""
    seen = []
    seen_set: set = set()
    for src in sources_used:
        slug = src.get("slug", "")
        if slug and slug not in seen_set:
            seen.append(slug)
            seen_set.add(slug)
    return seen
