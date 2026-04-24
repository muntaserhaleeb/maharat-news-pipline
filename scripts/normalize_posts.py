#!/usr/bin/env python3
"""
Maharat News Pipeline – Post Normalizer
========================================
Reads all extracted markdown posts from output/posts, then:

  1.  Assigns one category from a controlled list
  2.  Generates 3-7 tags from a controlled vocabulary
  3.  Cleans / regenerates the summary (≤ 200 chars, sentence-complete)
  4.  Collects image references per post
  5.  Renames images → output/images/{slug}-{n:02d}.{ext}
  6.  Updates image paths inside each markdown file
  7.  Writes output/manifests/publishing_manifest.json
  8.  Writes review/normalize_review.csv  (human-editable enrichment sheet)
  9.  Validates missing metadata, missing image files, duplicate slugs
  10. Prints a summary report

Usage:
    python scripts/normalize_posts.py
    python scripts/normalize_posts.py --dry-run   # no file changes
"""

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
POSTS_DIR = ROOT / "data" / "posts"
IMAGES_DIR = ROOT / "data" / "images"
MANIFESTS_DIR = ROOT / "data" / "manifests"
REVIEW_DIR = ROOT / "review"

# ── Load category rules from config/taxonomy.yaml ─────────────────────────
_taxonomy = yaml.safe_load((ROOT / "config" / "taxonomy.yaml").read_text(encoding="utf-8"))
_cat_cfg = _taxonomy["category_rules"]
DEFAULT_CATEGORY: str = _cat_cfg["default"]
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    (rule["name"], rule["keywords"])
    for rule in _cat_cfg["ordered_rules"]
]

# ── Controlled tag vocabulary ──────────────────────────────────────────────
# Each tag maps to keywords that trigger it.
TAG_RULES: list[tuple[str, list[str]]] = [
    # ── Intakes
    ("intake-11",  ["intake 11"]),
    ("intake-12",  ["intake 12"]),
    ("intake-13",  ["intake 13"]),
    ("intake-14",  ["intake 14"]),
    ("intake-15",  ["intake 15"]),
    # ── Trades / programs
    ("welding",        ["welding", "welder", "5g welding", "steelfab"]),
    ("scaffolding",    ["scaffolding", "scaffold"]),
    ("pipefitting",    ["pipefitting", "pipefitter"]),
    ("instrumentation",["instrumentation"]),
    ("construction-safety-program", ["construction safety", "csm course", "construction safety manual"]),
    ("osh",            ["osh training", "oil and gas safety", "oil & gas safety"]),
    ("wpr",            ["work permit receiver", "wpr pre-requisite", "wpr training"]),
    # ── Academic / learning
    ("english-language",  ["english conversation", "evp", "hvp", "headway program"]),
    ("examinations",      ["midterm exam", "end-of-semester exam", "final speaking test", "final examination"]),
    ("graduation",        ["graduation ceremony", "graduation project", "pre-graduation", "celebrates graduation"]),
    ("ojt",               ["ojt", "on-the-job training", "pre-ojt"]),
    ("cpd",               ["cpd", "continuing professional development"]),
    ("counseling",        ["counseling session", "counseling sessions", "counselor"]),
    # ── Safety topics
    ("fire-safety",         ["fire drill", "fire safety"]),
    ("drug-awareness",      ["drug awareness", "drug & alcohol", "alcohol-free", "substance"]),
    ("road-safety",         ["pedestrian safety", "wrong-way driving", "drive safety", "driving session"]),
    ("working-at-height",   ["working at height", "working-at-height", "height training"]),
    ("safety-campaign",     ["safety campaign", "safety awareness campaign", "safety contest", "safety slogan"]),
    ("camping-safety",      ["camping safety"]),
    # ── Events & community
    ("nariyah-spring-festival", ["nariyah spring festival"]),
    ("ramadan",               ["ramadan", "iftar"]),
    ("umrah",                 ["umrah"]),
    ("sports",                ["football tournament", "fun day"]),
    ("volunteer",             ["volunteer", "road repair initiative"]),
    ("female-trainees",       ["female trainee", "female cohort", "first cohort of female"]),
    ("religious-activities",  ["istisqa", "prayer for rain"]),
    ("career-guidance",       ["career guidance forum"]),
    ("speak-up-club",         ["speak up club"]),
    # ── Partners & bodies
    ("nesma",            ["nesma"]),
    ("saudi-aramco",     ["saudi aramco", "aramco"]),
    ("etec",             ["etec", "education and training evaluation"]),
    ("tvtc",             ["tvtc", "technical and vocational"]),
    ("hrdf",             ["hrdf", "human resources development fund"]),
    ("city-guilds",      ["city & guilds", "city guilds"]),
    ("gatehouse",        ["gatehouse awards", "gatehouse accreditation"]),
    ("samsung",          ["samsung e&a", "samsung ea"]),
    ("sicim",            ["sicim"]),
    ("nhti",             ["nesma high training", "nhti"]),
    ("jtc",              ["juaimah training center", "ju'aymah training center", "jtc"]),
    ("halumm",           ["halumm"]),
    ("tuv-sud",          ["tüv süd", "tuv sud"]),
    # ── Locations
    ("jubail",   ["jubail"]),
    ("riyadh",   ["riyadh"]),
    ("dammam",   ["dammam"]),
    ("abqaiq",   ["abqaiq"]),
    ("sharjah",  ["sharjah"]),
    # ── Digital & admin
    ("digital-transformation", ["digital transformation"]),
    ("classera",               ["classera"]),
    ("leadxera",               ["leadxera"]),
    ("accreditation",          ["accreditation", "accredited by"]),
    ("mou",                    [" mou", "memorandum of understanding"]),
    ("staff-well-being",       ["employee well-being", "staff well-being", "wellbeing initiative", "sponsored umrah"]),
    ("media-coverage",         ["published in construction week", "construction week"]),
    ("benchmarking",           ["benchmarking visit"]),
    ("leadership",             ["leadership workshop", "leadership for empowering"]),
]

TAG_MIN = 3
TAG_MAX = 7

# ── Markdown helpers ───────────────────────────────────────────────────────
_FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)


def parse_md(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2).strip()


def write_md(path: Path, front: dict, body: str, dry_run: bool = False):
    content = (
        "---\n"
        + yaml.dump(front, allow_unicode=True, sort_keys=False)
        + "---\n\n"
        + body
        + "\n"
    )
    if not dry_run:
        path.write_text(content, encoding="utf-8")


# ── Text scoring helpers ───────────────────────────────────────────────────
def _haystack(front: dict, body: str) -> str:
    """Build a single lower-case string for keyword matching."""
    return " ".join([
        front.get("title", ""),
        front.get("summary", ""),
        body,
    ]).lower()


def assign_category(front: dict, body: str) -> str:
    hay = _haystack(front, body)
    for category, keywords in CATEGORY_RULES:
        if any(kw in hay for kw in keywords):
            return category
    return DEFAULT_CATEGORY


def assign_tags(front: dict, body: str) -> list[str]:
    hay = _haystack(front, body)
    matched = [tag for tag, keywords in TAG_RULES if any(kw in hay for kw in keywords)]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for t in matched:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    # Trim to [TAG_MIN, TAG_MAX]
    if len(unique) < TAG_MIN:
        return unique          # keep what we have rather than padding with noise
    return unique[:TAG_MAX]


# ── Summary cleaner ────────────────────────────────────────────────────────
def clean_summary(front: dict, body: str, max_len: int = 200) -> str:
    """
    Return a clean ≤ max_len char summary, always extracted fresh from the body
    so we never re-use a pre-truncated stored value.
    Falls back to the stored summary only if the body yields nothing.
    Truncates at the last sentence boundary, then word boundary.
    """
    # Always prefer the full body paragraph (avoids re-truncating stored value)
    raw = ""
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("http"):
            raw = line
            break

    # Fall back to stored summary if body is empty
    if not raw:
        raw = (front.get("summary") or "").strip().rstrip("…").strip()
    if not raw:
        return ""

    if len(raw) <= max_len:
        return raw

    # Try sentence boundary first
    chunk = raw[:max_len]
    for sep in (". ", "! ", "? "):
        idx = chunk.rfind(sep)
        if idx > 60:
            return chunk[: idx + 1]
    # Trim to last word boundary
    word_end = chunk.rfind(" ")
    if word_end > 60:
        return chunk[:word_end] + "…"
    return chunk.rstrip() + "…"


# ── Image renaming ─────────────────────────────────────────────────────────
def collect_images(front: dict) -> list[str]:
    """Return list of image paths referenced in the post (featured first)."""
    imgs = []
    if front.get("featured_image"):
        imgs.append(front["featured_image"])
    imgs.extend(front.get("gallery_images") or [])
    return imgs


def rename_images(
    front: dict,
    slug: str,
    images_dir: Path,
    dry_run: bool = False,
) -> tuple[str, list[str], list[str]]:
    """
    Rename image files to {slug}-{n:02d}.{ext}.
    Returns (new_featured_image, new_gallery_images, warnings).
    """
    old_imgs = collect_images(front)
    warnings: list[str] = []
    new_paths: list[str] = []

    for i, rel_path in enumerate(old_imgs, start=1):
        old_abs = images_dir / Path(rel_path).name
        if not old_abs.exists():
            warnings.append(f"missing image file: {rel_path}")
            new_paths.append(rel_path)  # keep old ref, flag it
            continue

        ext = old_abs.suffix.lower()
        new_name = f"{slug}-{i:02d}{ext}"
        new_abs = images_dir / new_name
        new_rel = f"images/{new_name}"

        if old_abs.name != new_name:
            if not dry_run:
                shutil.move(str(old_abs), str(new_abs))
        new_paths.append(new_rel)

    featured = new_paths[0] if new_paths else ""
    gallery = new_paths[1:] if len(new_paths) > 1 else []
    return featured, gallery, warnings


# ── Validation ─────────────────────────────────────────────────────────────
REQUIRED_FIELDS = ["title", "slug", "date", "category", "tags", "summary", "featured_image"]


def validate_post(front: dict, images_dir: Path, original_images: list[str]) -> list[str]:
    issues: list[str] = []

    for field in REQUIRED_FIELDS:
        val = front.get(field)
        if not val or (isinstance(val, list) and len(val) == 0):
            issues.append(f"missing: {field}")

    if front.get("year") == 0:
        issues.append("missing: year (date not parsed)")

    tags = front.get("tags") or []
    if len(tags) < TAG_MIN:
        issues.append(f"low tag count: {len(tags)} (min {TAG_MIN})")

    # Build set of all image names that exist (both old hash-names and new slug-names)
    existing = {p.name for p in images_dir.glob("*") if p.is_file()}

    def img_exists(ref: str) -> bool:
        return Path(ref).name in existing

    feat = front.get("featured_image")
    # Only flag truly missing — if original also absent
    orig_featured = original_images[0] if original_images else ""
    if feat and not img_exists(feat) and not img_exists(orig_featured):
        issues.append(f"image file not found: {feat}")

    new_gallery = front.get("gallery_images") or []
    orig_gallery = original_images[1:] if len(original_images) > 1 else []
    for i, img in enumerate(new_gallery):
        orig = orig_gallery[i] if i < len(orig_gallery) else ""
        if not img_exists(img) and not img_exists(orig):
            issues.append(f"gallery image not found: {img}")

    return issues


# ── Output writers ─────────────────────────────────────────────────────────
MANIFEST_FIELDS = [
    "internal", "slug", "title", "date", "year", "quarter",
    "category", "tags", "featured_image", "gallery_images",
    "seo_title", "seo_description", "summary",
    "source_document", "source_section",
]

REVIEW_FIELDS = [
    "internal", "slug", "title", "date", "year", "quarter",
    "category", "tags", "location", "partner",
    "seo_title", "seo_description", "featured_image",
    "image_count", "summary", "validation",
]


def write_manifest(records: list[dict], dry_run: bool = False) -> Path:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(records),
        "posts": [
            {k: r.get(k) for k in MANIFEST_FIELDS}
            for r in records
        ],
    }
    out = MANIFESTS_DIR / "publishing_manifest.json"
    if not dry_run:
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_review_csv(records: list[dict], dry_run: bool = False) -> Path:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = REVIEW_DIR / "normalize_review.csv"
    if not dry_run:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                row = dict(r)
                row["tags"] = ", ".join(r.get("tags") or [])
                row["gallery_images"] = ", ".join(r.get("gallery_images") or [])
                writer.writerow(row)
    return out


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Normalize extracted news posts.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run all logic but write no files.",
    )
    args = parser.parse_args()
    dry = args.dry_run

    if dry:
        print("── DRY RUN – no files will be modified ──")

    md_files = sorted(POSTS_DIR.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {POSTS_DIR}")
        sys.exit(1)

    print(f"\nNormalizing {len(md_files)} posts …\n")

    records: list[dict] = []
    all_slugs: list[str] = []
    total_warnings: list[tuple[str, list[str]]] = []

    for md_path in md_files:
        front, body = parse_md(md_path)
        if not front:
            print(f"  SKIP (no front matter): {md_path.name}")
            continue

        slug = front.get("slug") or md_path.stem
        all_slugs.append(slug)

        # 1. Category — always recalculate (idempotent; human edits belong in review CSV)
        front["category"] = assign_category(front, body)

        # 2. Tags — always recalculate
        front["tags"] = assign_tags(front, body)

        # 3. Summary
        front["summary"] = clean_summary(front, body)
        seo_raw = front["summary"][:155] if front["summary"] else ""
        if len(front["summary"]) > 155:
            seo_raw = seo_raw[: seo_raw.rfind(" ")] + "…" if " " in seo_raw else seo_raw
        front["seo_description"] = seo_raw

        # 4 & 5. Rename images + update references
        original_images = collect_images(front)
        new_featured, new_gallery, img_warnings = rename_images(
            front, slug, IMAGES_DIR, dry_run=dry
        )
        front["featured_image"] = new_featured
        front["gallery_images"] = new_gallery

        # 6. Validate (pass originals so dry-run doesn't false-positive on renamed paths)
        issues = validate_post(front, IMAGES_DIR, original_images)
        if img_warnings:
            issues.extend(img_warnings)
        if issues:
            total_warnings.append((slug, issues))

        # 7. Write updated markdown
        write_md(md_path, front, body, dry_run=dry)

        image_count = (1 if new_featured else 0) + len(new_gallery)
        tag_str = ", ".join(front["tags"])
        print(f"  {slug[:55]:<55}  cat={front['category'][:20]:<20}  imgs={image_count}  tags={len(front['tags'])}")

        record = dict(front)
        record["image_count"] = image_count
        record["validation"] = "; ".join(issues) if issues else "ok"
        records.append(record)

    # ── Duplicate slug check
    seen: set[str] = set()
    dupes = []
    for s in all_slugs:
        if s in seen:
            dupes.append(s)
        seen.add(s)
    if dupes:
        print(f"\n  WARNING – duplicate slugs: {dupes}")

    # ── Write outputs
    manifest_path = write_manifest(records, dry_run=dry)
    csv_path = write_review_csv(records, dry_run=dry)

    # ── Validation report
    print(f"\n{'─' * 60}")
    ok_count = sum(1 for _, issues in total_warnings if not issues)
    print(f"Posts processed  : {len(records)}")
    print(f"With validation issues: {len(total_warnings)}")

    if total_warnings:
        print("\nValidation issues:")
        for slug, issues in total_warnings:
            for issue in issues:
                print(f"  [{slug[:45]}]  {issue}")

    if not dry:
        print(f"\nManifest → {manifest_path.relative_to(ROOT)}")
        print(f"Review   → {csv_path.relative_to(ROOT)}")

    print(f"\n{'✓' if not dry else '~'} Done.")


if __name__ == "__main__":
    main()
