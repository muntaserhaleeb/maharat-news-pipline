"""
Weekly Highlights Refresh Pipeline
====================================
Stages:
  1. identify   — locate existing posts whose source_document matches the .docx files in --source
  2. backup     — copy matched .md files and their images to a timestamped backup dir
  3. delete     — remove matched .md files, image files, and Qdrant points
  4. extract    — re-extract all posts from the new source .docx files
  5. normalize  — assign category / tags / summary, rename images
  6. export     — regenerate feed.json + posts.json for the full corpus
  7. ingest     — upsert newly extracted posts to Qdrant
  8. liferay    — write a Liferay-ready JSON + CSV manifest
  9. report     — write a full validation report; return it as a dict

The pipeline is idempotent: running it twice with the same source files
produces the same final state and never creates duplicate records.
"""

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── script-level imports (module-level side effects are config-loading only) ─
import extract_posts as _extract
import normalize_posts as _normalize
import export_feed as _export

# ── service imports ──────────────────────────────────────────────────────────
from qdrant_client.models import PointStruct, SparseVector

from services.chunk_service import make_chunks, parse_markdown
from services.config_service import (
    get_taxonomy_rules,
    load_chunking_config,
    load_qdrant_config,
    load_taxonomy,
)
from services.embedding_service import EmbeddingService
from services.entity_service import EntityService
from services.qdrant_service import QdrantService, build_payload, chunk_id_to_uuid

# ── path constants ───────────────────────────────────────────────────────────
POSTS_DIR     = ROOT / "data" / "posts"
IMAGES_DIR    = ROOT / "data" / "images"
FEEDS_DIR     = ROOT / "data"
MANIFESTS_DIR = ROOT / "data" / "manifests"
BACKUPS_DIR   = ROOT / "data" / "backups"
LOGS_DIR      = ROOT / "logs"
REVIEW_DIR    = ROOT / "review"


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_front(md_path: Path) -> dict:
    """Parse front matter only; return {} on failure."""
    front, _ = _normalize.parse_md(md_path)
    return front or {}


def _collect_image_paths(front: dict) -> List[Path]:
    """Resolve all image file paths referenced in a post's front matter."""
    refs = []
    if front.get("featured_image"):
        refs.append(front["featured_image"])
    refs.extend(front.get("gallery_images") or [])
    return [IMAGES_DIR / Path(r).name for r in refs if r]


def _slug_from_path(md_path: Path) -> str:
    return md_path.stem


# ── pipeline class ───────────────────────────────────────────────────────────

class RefreshPipeline:
    """
    Orchestrates the full Weekly Highlights refresh workflow.
    Construct via RefreshPipeline.from_config(source_dir).
    """

    def __init__(
        self,
        source_dir: Path,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        entity_service: EntityService,
        chunk_cfg: dict,
        taxonomy: dict,
        ingest_cfg: dict,
    ):
        self.source_dir        = source_dir
        self.qdrant_service    = qdrant_service
        self.embedding_service = embedding_service
        self.entity_service    = entity_service
        self.chunk_cfg         = chunk_cfg
        self.taxonomy          = taxonomy
        self.taxonomy_rules    = get_taxonomy_rules(taxonomy)
        self.ingest_cfg        = ingest_cfg

    @classmethod
    def from_config(cls, source_dir: Path) -> "RefreshPipeline":
        qdrant_cfg = load_qdrant_config()
        col_cfg    = qdrant_cfg["collections"]["primary"]
        taxonomy   = load_taxonomy()
        return cls(
            source_dir=source_dir,
            qdrant_service=QdrantService.from_config(qdrant_cfg),
            embedding_service=EmbeddingService.from_config(col_cfg),
            entity_service=EntityService.from_config(),
            chunk_cfg=load_chunking_config(),
            taxonomy=taxonomy,
            ingest_cfg=qdrant_cfg.get("ingestion", {}),
        )

    # ── stage 1: identify ───────────────────────────────────────────────────

    def _find_source_docs(self) -> List[Path]:
        """Return .docx files in source_dir."""
        return sorted(self.source_dir.glob("*.docx"))

    def _find_existing_posts(
        self,
        source_doc_names: List[str],
    ) -> Tuple[List[Path], List[str], List[Path]]:
        """
        Scan data/posts/ for .md files whose source_document matches any
        of the provided filenames.
        Returns (md_paths, slugs, image_paths).
        """
        if not POSTS_DIR.exists():
            return [], [], []
        md_paths:    List[Path] = []
        slugs:       List[str]  = []
        image_paths: List[Path] = []
        name_set = {n.lower() for n in source_doc_names}

        for md_path in sorted(POSTS_DIR.glob("*.md")):
            front = _read_front(md_path)
            if not front:
                continue
            src_doc = (front.get("source_document") or "").lower()
            if src_doc in name_set:
                md_paths.append(md_path)
                slug = front.get("slug") or md_path.stem
                slugs.append(slug)
                image_paths.extend(_collect_image_paths(front))

        # Deduplicate image paths
        seen: set = set()
        unique_imgs = []
        for p in image_paths:
            if str(p) not in seen:
                seen.add(str(p))
                unique_imgs.append(p)

        return md_paths, slugs, unique_imgs

    # ── stage 2: backup ─────────────────────────────────────────────────────

    def _backup(
        self,
        md_paths: List[Path],
        image_paths: List[Path],
        timestamp: str,
    ) -> Optional[Path]:
        """Copy .md files and images to a timestamped backup directory."""
        if not md_paths and not image_paths:
            return None

        backup_root = BACKUPS_DIR / timestamp
        posts_bak   = backup_root / "posts"
        images_bak  = backup_root / "images"
        posts_bak.mkdir(parents=True, exist_ok=True)
        images_bak.mkdir(parents=True, exist_ok=True)

        backed_posts  = 0
        backed_images = 0

        for md_path in md_paths:
            if md_path.exists():
                shutil.copy2(md_path, posts_bak / md_path.name)
                backed_posts += 1

        for img_path in image_paths:
            if img_path.exists():
                shutil.copy2(img_path, images_bak / img_path.name)
                backed_images += 1

        manifest = {
            "created_at":   timestamp,
            "posts_backed": backed_posts,
            "images_backed": backed_images,
            "posts":  [p.name for p in md_paths if p.exists()],
            "images": [p.name for p in image_paths if p.exists()],
        }
        (backup_root / "backup_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Backup → {backup_root.relative_to(ROOT)}  "
              f"({backed_posts} posts, {backed_images} images)")
        return backup_root

    # ── stage 3: delete ─────────────────────────────────────────────────────

    def _delete_from_disk(
        self,
        md_paths: List[Path],
        image_paths: List[Path],
        dry_run: bool,
    ) -> Tuple[int, int]:
        posts_del = images_del = 0
        for p in md_paths:
            if p.exists():
                if not dry_run:
                    p.unlink()
                posts_del += 1
        for p in image_paths:
            if p.exists():
                if not dry_run:
                    p.unlink()
                images_del += 1
        return posts_del, images_del

    def _delete_from_qdrant(self, slugs: List[str], dry_run: bool) -> int:
        if dry_run or not slugs:
            return 0
        count_before = self.qdrant_service.count_for_slugs(slugs)
        deleted      = self.qdrant_service.delete_by_slugs(slugs)
        print(f"  Qdrant: removed points for {deleted} slug(s) "
              f"(~{count_before} chunks deleted)")
        return count_before

    # ── stage 4: extract ────────────────────────────────────────────────────

    def _extract_docs(
        self,
        docx_paths: List[Path],
        dry_run: bool,
    ) -> Tuple[List[Path], List[dict], List[dict]]:
        """
        Run DOCX extraction for each source file.
        Returns (new_md_paths, post_dicts, failures).
        """
        if dry_run:
            return [], [], []

        POSTS_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        all_posts: List[dict] = []
        all_md_paths: List[Path] = []
        failures: List[dict] = []

        for docx_path in docx_paths:
            try:
                print(f"  Extracting: {docx_path.name}")
                posts = _extract.process_docx(docx_path)
                for post in posts:
                    md_path = POSTS_DIR / f"{post['slug']}.md"
                    if md_path.exists():
                        all_md_paths.append(md_path)
                all_posts.extend(posts)
            except Exception as e:
                msg = f"{docx_path.name}: {e}"
                print(f"  [fail] {msg}")
                failures.append({"file": str(docx_path), "error": str(e)})

        print(f"  Extracted {len(all_posts)} posts from {len(docx_paths)} file(s)")
        return all_md_paths, all_posts, failures

    # ── stage 5: normalize ──────────────────────────────────────────────────

    def _normalize_posts(
        self,
        md_paths: List[Path],
        dry_run: bool,
    ) -> Tuple[List[dict], List[Tuple[str, List[str]]]]:
        """
        Run category/tags/summary assignment and image renaming for each post.
        Returns (records, issues_list).
        """
        records: List[dict] = []
        issues:  List[Tuple[str, List[str]]] = []

        for md_path in md_paths:
            if not md_path.exists():
                continue
            front, body = _normalize.parse_md(md_path)
            if not front:
                continue

            slug = front.get("slug") or md_path.stem

            front["category"]    = _normalize.assign_category(front, body)
            front["tags"]        = _normalize.assign_tags(front, body)
            front["summary"]     = _normalize.clean_summary(front, body)
            seo_raw = front["summary"][:155] if front["summary"] else ""
            if len(front["summary"]) > 155:
                seo_raw = (seo_raw[: seo_raw.rfind(" ")] + "…") if " " in seo_raw else seo_raw
            front["seo_description"] = seo_raw

            orig_imgs = _normalize.collect_images(front)
            new_feat, new_gal, img_warns = _normalize.rename_images(
                front, slug, IMAGES_DIR, dry_run=dry_run
            )
            front["featured_image"] = new_feat
            front["gallery_images"] = new_gal

            post_issues = _normalize.validate_post(front, IMAGES_DIR, orig_imgs)
            post_issues.extend(img_warns)
            if post_issues:
                issues.append((slug, post_issues))

            _normalize.write_md(md_path, front, body, dry_run=dry_run)

            record = dict(front)
            record["image_count"] = (1 if new_feat else 0) + len(new_gal)
            record["validation"]  = "; ".join(post_issues) if post_issues else "ok"
            records.append(record)

        print(f"  Normalized {len(records)} post(s)  "
              f"({len(issues)} with validation issues)")
        return records, issues

    # ── stage 6: export ─────────────────────────────────────────────────────

    def _export_feeds(self, base_url: str, dry_run: bool) -> None:
        """Regenerate feed.json and posts.json for the full corpus."""
        if dry_run:
            return
        if not POSTS_DIR.exists():
            return

        records = []
        for path in sorted(POSTS_DIR.glob("*.md")):
            front, body = _export.parse_md(path)
            if front:
                records.append(_export.build_post_record(front, body, base_url, True))

        records = _export.sort_records(records)

        FEEDS_DIR.mkdir(parents=True, exist_ok=True)

        posts_feed = {
            "version":      "1.0",
            "title":        _export.FEED_TITLE,
            "description":  _export.FEED_DESCRIPTION,
            "language":     _export.FEED_LANGUAGE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total":        len(records),
            "posts":        records,
        }
        (FEEDS_DIR / "posts.json").write_text(
            json.dumps(posts_feed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        jsonfeed = _export.build_jsonfeed(records, base_url)
        (FEEDS_DIR / "feed.json").write_text(
            json.dumps(jsonfeed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Feeds updated: {len(records)} total posts")

    # ── stage 7: ingest ─────────────────────────────────────────────────────

    def _ingest_posts(
        self,
        md_paths: List[Path],
        dry_run: bool,
    ) -> int:
        """Embed and upsert newly extracted posts to Qdrant."""
        max_tokens     = self.chunk_cfg.get("chunking", {}).get("max_tokens", 700)
        overlap_tokens = self.chunk_cfg.get("chunking", {}).get("overlap_tokens", 100)
        batch_size     = self.ingest_cfg.get("upsert_batch_size", 64)

        if not dry_run:
            self.qdrant_service.setup_collection(recreate=False)
            self.qdrant_service.setup_payload_indexes()
            self.qdrant_service.setup_alias()

        pending: List[dict] = []
        for md_path in md_paths:
            if not md_path.exists():
                continue
            front, body = parse_markdown(md_path)
            if not front:
                continue
            entities = self.entity_service.extract_from_article(front, body)
            for chunk in make_chunks(front, body,
                                     max_tokens=max_tokens,
                                     overlap_tokens=overlap_tokens):
                pending.append({
                    "id":      chunk_id_to_uuid(chunk["chunk_id"]),
                    "text":    chunk["chunk_text"],
                    "payload": build_payload(
                        front, chunk, self.ingest_cfg, self.taxonomy_rules,
                        entities=entities,
                    ),
                })

        print(f"  {len(pending)} chunks to embed")

        if not pending:
            return 0

        points: List[PointStruct] = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start: start + batch_size]
            texts = [item["text"] for item in batch]
            dense_vecs, sparse_vecs = self.embedding_service.embed_documents(texts)
            for item, dv, sv in zip(batch, dense_vecs, sparse_vecs):
                points.append(PointStruct(
                    id=item["id"],
                    vector={
                        "dense": dv.tolist(),
                        "sparse": SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                    },
                    payload=item["payload"],
                ))
            print(f"  Embedded {min(start + batch_size, len(pending))}/{len(pending)}")

        if dry_run:
            print(f"  [dry-run] would upsert {len(points)} points")
            return 0

        return self.qdrant_service.upsert_points(points, batch_size=batch_size)

    # ── stage 8: liferay manifest ────────────────────────────────────────────

    def _create_liferay_manifest(
        self,
        records: List[dict],
        source_doc_names: List[str],
        dry_run: bool,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Write liferay_manifest.json and liferay_articles.csv."""
        if dry_run or not records:
            return None, None

        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)

        articles = []
        documents = []

        for i, r in enumerate(records, start=1):
            slug    = r.get("slug", "")
            title   = r.get("title", "")
            ext_ref = f"WH-{slug}"
            body    = ""
            # Reconstruct body from .md file if available
            md_path = POSTS_DIR / f"{slug}.md"
            if md_path.exists():
                _, raw_body = _normalize.parse_md(md_path)
                body = _export.md_to_html(raw_body) if raw_body else ""

            article = {
                "externalReferenceCode": ext_ref,
                "contentStructureId":    "MCTC_NEWS_ARTICLE",
                "friendlyUrlPath":       f"weekly-highlights/{slug}",
                "defaultLanguageId":     "en_US",
                "title":                 {"en_US": title},
                "description":           {"en_US": r.get("summary", "") or ""},
                "datePublished":         r.get("date") or None,
                "year":                  r.get("year") or None,
                "quarter":               r.get("quarter") or None,
                "status":                "approved",
                "taxonomyCategories":    [
                    {
                        "taxonomyVocabularyName": "Category",
                        "name": r.get("category", ""),
                    }
                ] if r.get("category") else [],
                "keywords": r.get("tags") or [],
                "contentFields": {
                    "Summary":         r.get("summary", ""),
                    "Body":            body,
                    "FeaturedImage":   r.get("featured_image", "") or "",
                    "GalleryImages":   ", ".join(r.get("gallery_images") or []),
                    "SourceDocument":  r.get("source_document", ""),
                    "SEOTitle":        r.get("seo_title", "") or "",
                    "SEODescription":  r.get("seo_description", "") or "",
                    "Location":        r.get("location", "") or "",
                    "Partner":         r.get("partner", "") or "",
                },
            }
            articles.append(article)

            # Image entries
            feat = r.get("featured_image", "")
            gal  = r.get("gallery_images") or []
            if feat:
                documents.append({
                    "externalReferenceCode": f"IMG-{slug}-01",
                    "fileName":              Path(feat).name,
                    "title":                 {"en_US": f"{title} — Featured Image"},
                    "articleReference":      ext_ref,
                    "sequence":              1,
                    "mediaType":             "featured",
                    "folder":                "weekly-highlights",
                })
            for j, img in enumerate(gal, start=2):
                documents.append({
                    "externalReferenceCode": f"IMG-{slug}-{j:02d}",
                    "fileName":              Path(img).name,
                    "title":                 {"en_US": f"{title} — Image {j}"},
                    "articleReference":      ext_ref,
                    "sequence":              j,
                    "mediaType":             "gallery",
                    "folder":                "weekly-highlights",
                })

        manifest = {
            "format":          "liferay-headless-v1",
            "defaultLanguageId": "en_US",
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "source_documents": source_doc_names,
            "summary": {
                "total_articles":  len(articles),
                "total_documents": len(documents),
            },
            "articles":  articles,
            "documents": documents,
        }

        json_path = MANIFESTS_DIR / "liferay_manifest.json"
        json_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # CSV (flat, spreadsheet-friendly)
        csv_path = REVIEW_DIR / "liferay_articles.csv"
        csv_fields = [
            "externalReferenceCode", "slug", "title", "datePublished",
            "year", "quarter", "category", "tags", "featured_image",
            "gallery_count", "source_document", "seo_title",
            "seo_description", "summary", "status", "friendlyUrlPath",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            for art, rec in zip(articles, records):
                w.writerow({
                    "externalReferenceCode": art["externalReferenceCode"],
                    "slug":           rec.get("slug", ""),
                    "title":          rec.get("title", ""),
                    "datePublished":  rec.get("date", ""),
                    "year":           rec.get("year", ""),
                    "quarter":        rec.get("quarter", ""),
                    "category":       rec.get("category", ""),
                    "tags":           ", ".join(rec.get("tags") or []),
                    "featured_image": rec.get("featured_image", "") or "",
                    "gallery_count":  len(rec.get("gallery_images") or []),
                    "source_document": rec.get("source_document", ""),
                    "seo_title":      rec.get("seo_title", "") or "",
                    "seo_description": rec.get("seo_description", "") or "",
                    "summary":        (rec.get("summary", "") or "")[:200],
                    "status":         "approved",
                    "friendlyUrlPath": f"weekly-highlights/{rec.get('slug', '')}",
                })

        print(f"  Liferay manifest → {json_path.relative_to(ROOT)}")
        print(f"  Liferay CSV      → {csv_path.relative_to(ROOT)}")
        return json_path, csv_path

    # ── stage 9: report ──────────────────────────────────────────────────────

    def _find_orphan_images(self, known_slugs: List[str]) -> List[str]:
        """Return image filenames in data/images/ not referenced by any known post."""
        if not IMAGES_DIR.exists():
            return []

        referenced: set = set()
        for md_path in POSTS_DIR.glob("*.md"):
            front = _read_front(md_path)
            if front.get("featured_image"):
                referenced.add(Path(front["featured_image"]).name)
            for img in (front.get("gallery_images") or []):
                referenced.add(Path(img).name)

        orphans = []
        for img_file in sorted(IMAGES_DIR.iterdir()):
            if img_file.is_file() and img_file.name not in referenced:
                orphans.append(img_file.name)
        return orphans

    def _write_report(self, report: dict, dry_run: bool) -> Path:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp    = report["run_at"].replace(":", "-").replace(".", "-")[:19]
        out_path = LOGS_DIR / f"refresh_report_{stamp}.json"
        if not dry_run:
            out_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return out_path

    def _write_manifest_json(self, records: List[dict], dry_run: bool) -> None:
        """Write / refresh publishing_manifest.json and normalize_review.csv."""
        if dry_run:
            return
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        _normalize.write_manifest(records, dry_run=False)
        _normalize.write_review_csv(records, dry_run=False)

    # ── main entry point ────────────────────────────────────────────────────

    def run(
        self,
        dry_run: bool = False,
        backup: bool = True,
        delete_existing: bool = True,
        reinsert: bool = True,
        regenerate_image_metadata: bool = True,
        create_liferay_manifest: bool = True,
        base_url: str = "",
    ) -> dict:
        """
        Execute the full refresh pipeline.
        Returns a validation report dict.
        """
        ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        now = datetime.now(timezone.utc).isoformat()

        print(f"\n{'═' * 70}")
        print(f"  Weekly Highlights Refresh Pipeline")
        print(f"  Source   : {self.source_dir}")
        print(f"  Dry-run  : {dry_run}")
        print(f"{'═' * 70}")

        report: dict = {
            "run_at":       now,
            "source_dir":   str(self.source_dir),
            "dry_run":      dry_run,
            "source_docs":  [],
            "backup_path":  None,
            "before":       {},
            "after":        {},
            "issues":       {
                "unmatched_images":  [],
                "missing_metadata":  [],
                "duplicate_slugs":   [],
                "failed_files":      [],
            },
        }

        # ── [1/9] identify source docs ──────────────────────────────────────
        print("\n[1/9] Identifying source documents…")
        if not self.source_dir.exists():
            print(f"  ERROR: source directory not found: {self.source_dir}")
            report["issues"]["failed_files"].append({
                "file": str(self.source_dir), "error": "directory not found"
            })
            return report

        source_docs = self._find_source_docs()
        if not source_docs:
            print(f"  No .docx files found in {self.source_dir}")
            return report

        source_doc_names = [d.name for d in source_docs]
        report["source_docs"] = source_doc_names
        print(f"  Found {len(source_docs)} source file(s): {source_doc_names}")

        # ── [2/9] identify existing posts ───────────────────────────────────
        print("\n[2/9] Identifying existing posts matching source documents…")
        existing_md, existing_slugs, existing_imgs = self._find_existing_posts(
            source_doc_names
        )
        qdrant_count = self.qdrant_service.count_for_slugs(existing_slugs)

        report["before"] = {
            "posts_found":        len(existing_md),
            "images_found":       len(existing_imgs),
            "qdrant_points_found": qdrant_count,
            "post_slugs":         existing_slugs,
            "image_files":        [p.name for p in existing_imgs],
        }
        print(f"  Existing posts   : {len(existing_md)}")
        print(f"  Existing images  : {len(existing_imgs)}")
        print(f"  Qdrant points    : {qdrant_count}")

        # ── [3/9] backup ────────────────────────────────────────────────────
        print("\n[3/9] Backup…")
        backup_path = None
        if backup and existing_md:
            backup_path = self._backup(existing_md, existing_imgs, ts)
            report["backup_path"] = str(backup_path) if backup_path else None
        else:
            reason = "skipped (no backup flag)" if not backup else "nothing to backup"
            print(f"  {reason}")

        # ── [4/9] delete existing ────────────────────────────────────────────
        print("\n[4/9] Deleting existing posts…")
        posts_del = images_del = qdrant_del = 0
        if delete_existing and existing_md:
            posts_del, images_del = self._delete_from_disk(
                existing_md, existing_imgs, dry_run
            )
            qdrant_del = self._delete_from_qdrant(existing_slugs, dry_run)
            prefix = "[dry-run] would delete" if dry_run else "Deleted"
            print(f"  {prefix} {posts_del} post(s), {images_del} image(s), "
                  f"~{qdrant_del} Qdrant chunk(s)")
        elif not delete_existing:
            print("  Skipped (--delete-existing false)")
        else:
            print("  Nothing to delete")

        # ── [5/9] extract ────────────────────────────────────────────────────
        print("\n[5/9] Extracting posts from source documents…")
        new_md_paths, new_posts, extract_failures = self._extract_docs(
            source_docs, dry_run
        )
        report["issues"]["failed_files"].extend(extract_failures)

        # ── [6/9] normalize ──────────────────────────────────────────────────
        print("\n[6/9] Normalizing posts…")
        norm_records: List[dict] = []
        norm_issues: List[Tuple[str, List[str]]] = []
        if regenerate_image_metadata and new_md_paths:
            norm_records, norm_issues = self._normalize_posts(new_md_paths, dry_run)
            # Update new_md_paths to use normalized front matter paths
        elif new_md_paths:
            print("  Skipped (--regenerate-image-metadata false)")
            # Load records directly from .md files without normalizing
            for md_path in new_md_paths:
                front, _ = _normalize.parse_md(md_path)
                if front:
                    norm_records.append(front)

        # Check for duplicate slugs across newly extracted posts
        seen_slugs: dict = {}
        for r in norm_records:
            s = r.get("slug", "")
            seen_slugs[s] = seen_slugs.get(s, 0) + 1
        dupes = [s for s, count in seen_slugs.items() if count > 1]
        if dupes:
            report["issues"]["duplicate_slugs"] = dupes
            print(f"  WARNING: duplicate slugs found: {dupes}")

        # Collect metadata issues
        for slug, issues in norm_issues:
            report["issues"]["missing_metadata"].append({
                "slug": slug, "issues": issues
            })

        # ── [7/9] export feeds ───────────────────────────────────────────────
        print("\n[7/9] Exporting feeds…")
        self._export_feeds(base_url, dry_run)

        # Write publishing_manifest.json + normalize_review.csv
        if norm_records:
            self._write_manifest_json(norm_records, dry_run)

        # ── [8/9] ingest to Qdrant ───────────────────────────────────────────
        print("\n[8/9] Ingesting new posts to Qdrant…")
        points_upserted = 0
        if reinsert and new_md_paths:
            points_upserted = self._ingest_posts(new_md_paths, dry_run)
        elif not reinsert:
            print("  Skipped (--reinsert false)")
        else:
            print("  Nothing to ingest")

        # ── [9/9] liferay manifest ───────────────────────────────────────────
        print("\n[9/9] Generating Liferay manifest…")
        liferay_json = liferay_csv = None
        if create_liferay_manifest and norm_records:
            liferay_json, liferay_csv = self._create_liferay_manifest(
                norm_records, source_doc_names, dry_run
            )
        elif not create_liferay_manifest:
            print("  Skipped (--create-liferay-manifest false)")
        else:
            print("  No records to export")

        # ── compute after snapshot ───────────────────────────────────────────
        new_slugs = [r.get("slug", "") for r in norm_records]
        new_qdrant_count = (
            self.qdrant_service.count_for_slugs(new_slugs)
            if new_slugs and not dry_run
            else points_upserted
        )

        report["after"] = {
            "posts_extracted":  len(new_posts),
            "posts_normalized": len(norm_records),
            "chunks_produced":  points_upserted,  # approximation from ingest step
            "points_upserted":  points_upserted if not dry_run else 0,
            "qdrant_points":    new_qdrant_count,
            "post_slugs":       new_slugs,
        }

        # ── orphan images ───────────────────────────────────────────────────
        orphans = self._find_orphan_images(new_slugs)
        report["issues"]["unmatched_images"] = orphans
        if orphans:
            print(f"\n  WARNING: {len(orphans)} orphan image(s) not referenced by any post")

        # ── write report ────────────────────────────────────────────────────
        report_path = self._write_report(report, dry_run)

        # ── print summary ────────────────────────────────────────────────────
        print(f"\n{'─' * 70}")
        print("  VALIDATION REPORT")
        print(f"{'─' * 70}")
        print(f"  Posts deleted         : {posts_del}")
        print(f"  Images deleted        : {images_del}")
        print(f"  Qdrant chunks deleted : ~{qdrant_del}")
        print(f"  Posts extracted       : {len(new_posts)}")
        print(f"  Posts normalized      : {len(norm_records)}")
        print(f"  Qdrant chunks upserted: {points_upserted}")
        print(f"  Orphan images         : {len(orphans)}")
        print(f"  Missing metadata      : {len(norm_issues)}")
        print(f"  Duplicate slugs       : {len(dupes)}")
        print(f"  Failed files          : {len(extract_failures)}")
        if not dry_run:
            print(f"\n  Report → {report_path.relative_to(ROOT)}")
            if backup_path:
                print(f"  Backup → {backup_path.relative_to(ROOT)}")
        print(f"\n{'✓' if not dry_run else '~'} Refresh {'complete' if not dry_run else 'dry-run complete'}.")
        return report
