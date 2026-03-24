"""
Chunk service — markdown parsing, chunking, post loading, and metadata validation.
Self-contained: no imports from scripts/.
Consolidates scripts/chunk_markdown.py + scripts/ingest_markdown.py.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_FM_RE      = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


# ── token estimation ───────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count: ~1.3 tokens per word."""
    return max(1, int(len(text.split()) * 1.3))


# ── markdown parsing ───────────────────────────────────────────────────────

def parse_markdown(path) -> Tuple[dict, str]:
    """Return (front_matter_dict, body_str) from a markdown file."""
    text = Path(path).read_text(encoding="utf-8")
    m    = _FM_RE.match(text)
    if not m:
        return {}, text.strip()
    return yaml.safe_load(m.group(1)) or {}, m.group(2).strip()


# ── chunking internals ─────────────────────────────────────────────────────

def _build_heading_path(stack: list) -> str:
    return " / ".join(h for _, h in stack) if stack else ""


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks   = []
    current  = []
    curr_tok = 0

    for para in paragraphs:
        pt = estimate_tokens(para)
        if current and curr_tok + pt > max_tokens:
            chunks.append("\n\n".join(current))
            overlap    = []
            overlap_tok = 0
            for p in reversed(current):
                pt2 = estimate_tokens(p)
                if overlap_tok + pt2 <= overlap_tokens:
                    overlap.insert(0, p)
                    overlap_tok += pt2
                else:
                    break
            current  = overlap
            curr_tok = overlap_tok
        current.append(para)
        curr_tok += pt

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_into_sections(body: str) -> List[Tuple[int, str, str]]:
    """Split markdown body at heading lines. Returns [(level, heading, body)]."""
    lines    = body.splitlines()
    sections = []
    cur_level   = 0
    cur_heading = ""
    cur_lines   = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            if cur_lines or cur_level > 0:
                sections.append((cur_level, cur_heading, "\n".join(cur_lines).strip()))
            cur_level   = len(m.group(1))
            cur_heading = m.group(2).strip()
            cur_lines   = []
        else:
            cur_lines.append(line)

    if cur_lines or cur_level > 0:
        sections.append((cur_level, cur_heading, "\n".join(cur_lines).strip()))

    return sections


# ── public chunking API ────────────────────────────────────────────────────

def make_chunks(
    front: dict,
    body: str,
    max_tokens: int = 700,
    overlap_tokens: int = 100,
) -> List[dict]:
    """
    Generate all chunks for one article.
    Returns list of dicts: chunk_id, chunk_index, chunk_type,
    heading_path, chunk_text, word_count.
    """
    slug    = front.get("slug", "")
    title   = front.get("title", "") or ""
    summary = front.get("summary", "") or ""
    results = []
    idx     = 0

    # chunk 0 — summary
    intro_parts = [p for p in [title, summary] if p.strip()]
    intro_text  = "\n\n".join(intro_parts)
    if intro_text.strip():
        results.append({
            "chunk_id":     f"{slug}__{idx:03d}",
            "chunk_index":  idx,
            "chunk_type":   "summary",
            "heading_path": title,
            "chunk_text":   intro_text,
            "word_count":   len(intro_text.split()),
        })
        idx += 1

    if not body:
        return results

    sections     = _split_into_sections(body)
    has_headings = any(level > 0 for level, _, _ in sections)

    if not has_headings:
        for tc in _chunk_text(body.strip(), max_tokens, overlap_tokens):
            if not tc.strip():
                continue
            results.append({
                "chunk_id":     f"{slug}__{idx:03d}",
                "chunk_index":  idx,
                "chunk_type":   "body",
                "heading_path": title,
                "chunk_text":   tc,
                "word_count":   len(tc.split()),
            })
            idx += 1
        return results

    heading_stack: list = []

    for level, heading, section_body in sections:
        if level == 0:
            section_text = section_body.strip()
            hpath        = title
        else:
            heading_stack = [(l, h) for l, h in heading_stack if l < level]
            heading_stack.append((level, heading))
            hpath        = _build_heading_path(heading_stack)
            section_text = heading
            if section_body.strip():
                section_text += "\n\n" + section_body.strip()

        if not section_text.strip():
            continue

        for tc in _chunk_text(section_text, max_tokens, overlap_tokens):
            if not tc.strip():
                continue
            results.append({
                "chunk_id":     f"{slug}__{idx:03d}",
                "chunk_index":  idx,
                "chunk_type":   "body",
                "heading_path": hpath,
                "chunk_text":   tc,
                "word_count":   len(tc.split()),
            })
            idx += 1

    return results


# ── post loading and validation ────────────────────────────────────────────

def load_posts(
    posts_dir,
    slug_filter: Optional[str] = None,
) -> List[Tuple[dict, str]]:
    """
    Read all markdown files in posts_dir.
    Returns list of (front, body) tuples — skips files with no front matter.
    """
    posts_dir = Path(posts_dir)
    md_files  = sorted(posts_dir.glob("*.md"))

    if slug_filter:
        md_files = [f for f in md_files if f.stem == slug_filter]
        if not md_files:
            return []

    parsed = []
    for md_path in md_files:
        front, body = parse_markdown(md_path)
        if not front:
            print(f"  [skip] {md_path.name}: no front matter")
        else:
            parsed.append((front, body))

    return parsed


def validate_post(
    front: dict,
    taxonomy: dict,
    ingest_cfg: dict,
    warnings: list,
) -> bool:
    """
    Check front matter against taxonomy rules.
    Appends warning strings to warnings list. Returns True if can proceed.
    """
    rules    = taxonomy.get("rules", {})
    cats     = set(taxonomy.get("categories", []))
    all_tags: set = set()
    for g in taxonomy.get("tags", {}).values():
        if isinstance(g, list):
            all_tags.update(g)

    slug  = front.get("slug", "")
    title = front.get("title", "")

    if rules.get("require_slug") and not slug:
        warnings.append("missing slug")
        if ingest_cfg.get("fail_on_missing_slug"):
            return False

    if rules.get("require_title") and not title:
        warnings.append("missing title")

    if rules.get("require_summary") and not front.get("summary", "").strip():
        warnings.append("missing summary")

    if not front.get("date") and ingest_cfg.get("fail_on_missing_date"):
        warnings.append("missing date")
        return False

    cat = front.get("category", "")
    if cat and cats and cat not in cats:
        warnings.append(
            f"category '{cat}' not in taxonomy (pipeline categories differ — continuing)"
        )

    return True


def validate_all(
    parsed: List[Tuple[dict, str]],
    taxonomy: dict,
    ingest_cfg: dict,
) -> Tuple[List[Tuple[dict, str]], List[dict]]:
    """
    Validate all loaded (front, body) pairs.
    Returns (valid, failed) where failed is a list of {"slug": ..., "reason": ...}.
    """
    valid  = []
    failed = []
    for front, body in parsed:
        slug     = front.get("slug", "")
        warnings: list = []
        if not validate_post(front, taxonomy, ingest_cfg, warnings):
            print(f"  [fail] {slug}: {'; '.join(warnings)}")
            failed.append({"slug": slug, "reason": "; ".join(warnings)})
            continue
        if warnings:
            print(f"  [warn] {slug}: {'; '.join(warnings)}")
        valid.append((front, body))
    return valid, failed
