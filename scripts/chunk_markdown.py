#!/usr/bin/env python3
"""
Markdown → chunks converter.

Splits a markdown document (with YAML front matter) into retrieval chunks:
  - chunk 0: summary chunk  (title + summary, chunk_type="summary")
  - remaining: body chunks  (one per section, split if too long, chunk_type="body")

Heading hierarchy is tracked so each chunk carries a heading_path string.

Usage:
    python scripts/chunk_markdown.py data/posts/some-post.md
    python scripts/chunk_markdown.py data/posts/some-post.md --max-tokens 500
"""

import argparse
import re
from pathlib import Path

import yaml

_FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


# ── helpers ────────────────────────────────────────────────────────────────

def estimate_tokens(text):
    """Rough token count: ~1.3 tokens per word."""
    return max(1, int(len(text.split()) * 1.3))


def parse_markdown(path):
    """Return (front_matter_dict, body_str) from a markdown file."""
    text = Path(path).read_text(encoding="utf-8")
    m = _FM_RE.match(text)
    if not m:
        return {}, text.strip()
    return yaml.safe_load(m.group(1)) or {}, m.group(2).strip()


def _build_heading_path(stack):
    return " / ".join(h for _, h in stack) if stack else ""


def _split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _chunk_text(text, max_tokens, overlap_tokens):
    """
    Split text into chunks at paragraph boundaries.
    Each chunk is at most max_tokens; consecutive chunks share
    up to overlap_tokens worth of content from the previous chunk.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    current = []
    current_tok = 0

    for para in paragraphs:
        pt = estimate_tokens(para)
        if current and current_tok + pt > max_tokens:
            chunks.append("\n\n".join(current))
            # Build overlap from tail of current
            overlap = []
            overlap_tok = 0
            for p in reversed(current):
                pt2 = estimate_tokens(p)
                if overlap_tok + pt2 <= overlap_tokens:
                    overlap.insert(0, p)
                    overlap_tok += pt2
                else:
                    break
            current = overlap
            current_tok = overlap_tok
        current.append(para)
        current_tok += pt

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_into_sections(body):
    """
    Split markdown body at heading lines.
    Returns list of (level, heading_text, section_body_text).
    level=0 means pre-heading text before any heading.
    """
    lines = body.splitlines()
    sections = []
    current_level = 0
    current_heading = ""
    current_lines = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # flush current section
            if current_lines or current_level > 0:
                sections.append((
                    current_level,
                    current_heading,
                    "\n".join(current_lines).strip(),
                ))
            current_level = len(m.group(1))
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # flush last section
    if current_lines or current_level > 0:
        sections.append((current_level, current_heading, "\n".join(current_lines).strip()))

    return sections


# ── public API ─────────────────────────────────────────────────────────────

def make_chunks(front, body, max_tokens=700, overlap_tokens=100):
    """
    Generate all chunks for one article.

    Returns a list of dicts, each with:
      chunk_id, chunk_index, chunk_type, heading_path, chunk_text, word_count
    """
    slug = front.get("slug", "")
    title = front.get("title", "") or ""
    summary = front.get("summary", "") or ""

    results = []
    idx = 0

    # ── chunk 0: summary chunk ────────────────────────────────────────────
    intro_parts = [p for p in [title, summary] if p.strip()]
    intro_text = "\n\n".join(intro_parts)
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

    sections = _split_into_sections(body)

    # If the body has no headings, treat it as one flat section
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

    # Headed sections — track heading hierarchy
    heading_stack = []  # [(level, text), ...]

    for level, heading, section_body in sections:
        if level == 0:
            # pre-heading body text
            section_text = section_body.strip()
            hpath = title
        else:
            # Pop deeper or same-level entries, then push current
            heading_stack = [(l, h) for l, h in heading_stack if l < level]
            heading_stack.append((level, heading))
            hpath = _build_heading_path(heading_stack)
            # Section text starts with the heading itself
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


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chunk a markdown post and print the results.")
    parser.add_argument("path", help="Path to a markdown file")
    parser.add_argument("--max-tokens",    type=int, default=700, help="Max tokens per chunk (default 700)")
    parser.add_argument("--overlap-tokens",type=int, default=100, help="Overlap tokens between chunks (default 100)")
    args = parser.parse_args()

    front, body = parse_markdown(args.path)
    if not front:
        print(f"No front matter found in {args.path}")
        return

    chunks = make_chunks(front, body, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
    print(f"Slug : {front.get('slug', '?')}")
    print(f"Chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"  [{c['chunk_index']:03d}] type={c['chunk_type']:<8} words={c['word_count']:3d}  path={c['heading_path'][:60]}")


if __name__ == "__main__":
    main()
