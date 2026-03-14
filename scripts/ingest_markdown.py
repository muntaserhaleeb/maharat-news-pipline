#!/usr/bin/env python3
"""
Stage 1 & 2: Load markdown posts from data/posts/ and validate metadata.

Provides importable functions used by upsert_qdrant.py, and a standalone
CLI for inspecting/validating posts without running the full ingestion.

Usage:
    python scripts/ingest_markdown.py
    python scripts/ingest_markdown.py --slug mctc-hosts-fire-drill
    python scripts/ingest_markdown.py --strict     # exit 1 if any post fails
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = ROOT / "data" / "posts"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_markdown import parse_markdown
from config import load_qdrant_config, load_taxonomy


# ── validation ─────────────────────────────────────────────────────────────

def validate_post(front, taxonomy, ingest_cfg, warnings):
    """
    Check front matter against taxonomy rules.
    Appends warning strings to `warnings`. Returns True if can proceed.
    """
    rules    = taxonomy.get("rules", {})
    cats     = set(taxonomy.get("categories", []))
    all_tags = set()
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
        warnings.append(f"category '{cat}' not in taxonomy (pipeline categories differ — continuing)")

    return True


# ── loaders ────────────────────────────────────────────────────────────────

def load_posts(posts_dir=None, slug_filter=None):
    """
    Read all markdown files in posts_dir.
    Returns list of (front, body) tuples — skips files with no front matter.

    Args:
        posts_dir:   Path to posts directory (defaults to data/posts/).
        slug_filter: If set, return only the post with this slug.
    """
    if posts_dir is None:
        posts_dir = POSTS_DIR

    md_files = sorted(Path(posts_dir).glob("*.md"))
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


def validate_all(parsed, taxonomy, ingest_cfg):
    """
    Validate all loaded (front, body) pairs.
    Returns (valid, failed) where:
      valid  — list of (front, body) that passed
      failed — list of {"slug": ..., "reason": ...} dicts
    """
    valid  = []
    failed = []
    for front, body in parsed:
        slug     = front.get("slug", "")
        warnings = []
        if not validate_post(front, taxonomy, ingest_cfg, warnings):
            print(f"  [fail] {slug}: {'; '.join(warnings)}")
            failed.append({"slug": slug, "reason": "; ".join(warnings)})
            continue
        if warnings:
            print(f"  [warn] {slug}: {'; '.join(warnings)}")
        valid.append((front, body))
    return valid, failed


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load and validate markdown posts.")
    parser.add_argument("--slug",   default="", help="Inspect a single post by slug.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any post fails validation.")
    args = parser.parse_args()

    qdrant_cfg = load_qdrant_config()
    taxonomy   = load_taxonomy()
    ingest_cfg = qdrant_cfg.get("ingestion", {})

    print(f"\n[1/2] Loading markdown posts from {POSTS_DIR}…")
    parsed = load_posts(slug_filter=args.slug or None)
    if not parsed:
        target = f"slug '{args.slug}'" if args.slug else str(POSTS_DIR)
        print(f"  No posts found in {target}")
        sys.exit(1)
    print(f"  Loaded {len(parsed)} post(s)")

    print("\n[2/2] Validating metadata…")
    valid, failed = validate_all(parsed, taxonomy, ingest_cfg)
    print(f"  Valid: {len(valid)}  |  Failed: {len(failed)}")

    if failed:
        print("\n  Failed posts:")
        for rec in failed:
            print(f"    [{rec['slug']}] {rec['reason']}")
        if args.strict:
            sys.exit(1)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
