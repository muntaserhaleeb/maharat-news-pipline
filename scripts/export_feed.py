#!/usr/bin/env python3
"""
Maharat News Pipeline – JSON Feed Exporter
===========================================
Reads normalized markdown posts from output/posts/ and writes two feeds:

  output/feed.json          JSON Feed 1.1  (jsonfeed.org standard)
  output/posts.json         Flat posts array  (CMS / API consumption)

Usage:
    python scripts/export_feed.py
    python scripts/export_feed.py --base-url https://maharat.com
    python scripts/export_feed.py --base-url https://maharat.com --no-body
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = ROOT / "output" / "posts"
OUTPUT_DIR = ROOT / "output"

# ── Defaults ───────────────────────────────────────────────────────────────
FEED_TITLE = "MCTC News"
FEED_DESCRIPTION = (
    "News, highlights, and updates from Maharat Construction Training Center (MCTC)."
)
FEED_LANGUAGE = "en"
FEED_AUTHOR = {"name": "Maharat Construction Training Center"}

# ── Markdown → HTML (lightweight, no extra deps) ───────────────────────────
_H_RE     = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_BOLD_RE  = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"\*(.+?)\*")
_LINK_RE  = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMG_RE   = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_UL_RE    = re.compile(r"^- (.+)", re.MULTILINE)


def md_to_html(md: str, base_url: str = "") -> str:
    """
    Convert a subset of Markdown to HTML.
    Handles headings, bold, italic, links, images, unordered lists, paragraphs.
    """
    if not md.strip():
        return ""

    html_lines: list[str] = []
    in_ul = False

    for line in md.splitlines():
        stripped = line.strip()

        # Heading
        hm = _H_RE.match(stripped)
        if hm:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            level = len(hm.group(1))
            html_lines.append(f"<h{level}>{_inline(hm.group(2), base_url)}</h{level}>")
            continue

        # List item
        lm = _UL_RE.match(stripped)
        if lm:
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{_inline(lm.group(1), base_url)}</li>")
            continue

        # Close list
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False

        # Blank line → paragraph break
        if not stripped:
            if html_lines and html_lines[-1] not in ("", "<br>"):
                html_lines.append("")
            continue

        html_lines.append(f"<p>{_inline(stripped, base_url)}</p>")

    if in_ul:
        html_lines.append("</ul>")

    return "\n".join(html_lines).strip()


def _inline(text: str, base_url: str) -> str:
    text = _IMG_RE.sub(
        lambda m: f'<img src="{_abs(m.group(2), base_url)}" alt="{m.group(1)}">',
        text,
    )
    text = _LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return text


def _abs(path: str, base_url: str) -> str:
    """Make a relative path absolute using base_url, if provided."""
    if not path or path.startswith("http"):
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}" if base_url else path


# ── Markdown front-matter parser ───────────────────────────────────────────
_FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)


def parse_md(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2).strip()


# ── Field cleaners ─────────────────────────────────────────────────────────
def _clean_seo_title(title: str, max_len: int = 60) -> str:
    """Trim to last word boundary within max_len."""
    if not title or len(title) <= max_len:
        return title
    chunk = title[:max_len]
    space = chunk.rfind(" ")
    return (chunk[:space] if space > 20 else chunk).rstrip("- ,")


def _iso_date(date_str: str):
    """Return RFC 3339 datetime string or None."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _null_empty(val):
    """Convert empty string / zero / empty list to None."""
    if val == "" or val == 0 or val == [] or val == {}:
        return None
    return val


# ── Post builder ───────────────────────────────────────────────────────────
def build_post_record(front: dict, body: str, base_url: str, include_body: bool) -> dict:
    slug      = front.get("slug", "")
    post_url  = _abs(f"posts/{slug}", base_url)
    date_iso  = _iso_date(front.get("date", ""))
    img_rel   = front.get("featured_image") or ""
    gallery   = front.get("gallery_images") or []

    record: dict = {
        # ── Identity
        "id":             slug,
        "internal":       front.get("internal"),
        "slug":           slug,
        "url":            post_url or None,

        # ── Content
        "title":          front.get("title") or None,
        "summary":        _null_empty(front.get("summary", "")),
        "seo_title":      _clean_seo_title(front.get("seo_title") or front.get("title") or ""),
        "seo_description":_null_empty(front.get("seo_description", "")),

        # ── Taxonomy
        "category":       _null_empty(front.get("category", "")),
        "tags":           front.get("tags") or [],

        # ── Dates
        "date":           _null_empty(front.get("date", "")),
        "date_published": date_iso,
        "year":           _null_empty(front.get("year", 0)),
        "quarter":        _null_empty(front.get("quarter", "")),

        # ── Media
        "featured_image": _abs(img_rel, base_url) if img_rel else None,
        "gallery_images": [_abs(p, base_url) for p in gallery] if gallery else [],

        # ── Relations
        "location":       _null_empty(front.get("location", "")),
        "partner":        _null_empty(front.get("partner", "")),

        # ── Source
        "source_document": front.get("source_document"),
        "source_section":  front.get("source_section"),
        "source_page":     _null_empty(front.get("source_page", 0)),
    }

    if include_body:
        record["body_markdown"] = body or None
        record["body_html"]     = md_to_html(body, base_url) if body else None

    return record


# ── JSON Feed 1.1 builder ──────────────────────────────────────────────────
def build_jsonfeed(records: list[dict], base_url: str) -> dict:
    feed_url = _abs("feed.json", base_url)
    home_url = base_url or None

    items = []
    for r in records:
        item: dict = {
            "id":             r["url"] or r["slug"],
            "url":            r["url"],
            "title":          r["title"],
            "summary":        r["summary"],
            "date_published": r["date_published"],
            "tags":           r["tags"] or None,
            "image":          r["featured_image"],
            "_maharat": {
                "internal":       r["internal"],
                "category":       r["category"],
                "quarter":        r["quarter"],
                "gallery_images": r["gallery_images"] or None,
                "seo_title":      r["seo_title"],
                "seo_description":r["seo_description"],
                "location":       r["location"],
                "partner":        r["partner"],
                "source_document":r["source_document"],
            },
        }
        if r.get("body_html"):
            item["content_html"] = r["body_html"]
        elif r.get("body_markdown"):
            item["content_text"] = r["body_markdown"]
        # Drop None values for a clean feed
        item = {k: v for k, v in item.items() if v is not None}
        item["_maharat"] = {k: v for k, v in item["_maharat"].items() if v is not None}
        items.append(item)

    feed = {
        "version":      "https://jsonfeed.org/version/1.1",
        "title":        FEED_TITLE,
        "description":  FEED_DESCRIPTION,
        "language":     FEED_LANGUAGE,
        "authors":      [FEED_AUTHOR],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_items":  len(items),
    }
    if home_url:
        feed["home_page_url"] = home_url
    if feed_url:
        feed["feed_url"] = feed_url
    feed["items"] = items
    return feed


# ── Sort ───────────────────────────────────────────────────────────────────
def sort_records(records: list[dict]) -> list[dict]:
    """Newest first; undated posts sorted to end by title."""
    def key(r):
        d = r.get("date") or ""
        return ("0" if d else "1", d if d else r.get("title", "").lower())

    return sorted(records, key=lambda r: (not r.get("date"), r.get("date", ""), 0), reverse=False) \
        if False else sorted(records, key=lambda r: ("" if r.get("date") else "z", r.get("date", "")), reverse=True)


# ── Stats ──────────────────────────────────────────────────────────────────
def print_stats(records: list[dict]):
    total       = len(records)
    with_date   = sum(1 for r in records if r.get("date"))
    with_image  = sum(1 for r in records if r.get("featured_image"))
    with_tags   = sum(1 for r in records if r.get("tags"))
    no_date     = [r["slug"] for r in records if not r.get("date")]
    no_image    = [r["slug"] for r in records if not r.get("featured_image")]

    cats: dict[str, int] = {}
    for r in records:
        c = r.get("category") or "—"
        cats[c] = cats.get(c, 0) + 1

    print(f"\n  Total posts   : {total}")
    print(f"  Dated         : {with_date}/{total}")
    print(f"  With image    : {with_image}/{total}")
    print(f"  With tags     : {with_tags}/{total}")

    print("\n  By category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {count:3d}  {cat}")

    if no_date:
        print(f"\n  Posts missing date ({len(no_date)}):")
        for s in no_date:
            print(f"    {s}")
    if no_image:
        print(f"\n  Posts missing featured_image ({len(no_image)}):")
        for s in no_image:
            print(f"    {s}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export posts as a publishable JSON feed.")
    parser.add_argument(
        "--base-url", default="",
        help="Base URL for absolute image/post links (e.g. https://maharat.com).",
    )
    parser.add_argument(
        "--no-body", action="store_true",
        help="Omit body_markdown and body_html from posts.json (smaller file).",
    )
    args = parser.parse_args()
    base_url    = args.base_url.rstrip("/")
    include_body = not args.no_body

    md_files = sorted(POSTS_DIR.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {POSTS_DIR}")
        sys.exit(1)

    print(f"\nExporting {len(md_files)} posts …")

    records: list[dict] = []
    for path in md_files:
        front, body = parse_md(path)
        if not front:
            continue
        records.append(build_post_record(front, body, base_url, include_body))

    records = sort_records(records)

    # ── Write posts.json
    posts_feed = {
        "version":      "1.0",
        "title":        FEED_TITLE,
        "description":  FEED_DESCRIPTION,
        "language":     FEED_LANGUAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total":        len(records),
        "posts":        records,
    }
    posts_path = OUTPUT_DIR / "posts.json"
    posts_path.write_text(
        json.dumps(posts_feed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Write feed.json  (JSON Feed 1.1)
    jsonfeed = build_jsonfeed(records, base_url)
    feed_path = OUTPUT_DIR / "feed.json"
    feed_path.write_text(
        json.dumps(jsonfeed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_stats(records)

    print(f"\n  → {feed_path.relative_to(ROOT)}   (JSON Feed 1.1)")
    print(f"  → {posts_path.relative_to(ROOT)}  (flat posts array)")
    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()
