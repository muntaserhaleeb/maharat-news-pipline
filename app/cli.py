#!/usr/bin/env python3
"""
Maharat News Pipeline CLI
==========================
Single entry point — calls pipelines only.

Commands:
    ingest          Validate, chunk, embed and upsert data/posts/ into Qdrant
    rebuild-index   Drop collection, recreate schema, re-ingest everything
    search          Hybrid search over indexed content
    draft           Generate a grounded article draft via RAG
    evaluate        Run retrieval eval cases from tests/retrieval_eval.csv

Examples:
    python app/cli.py ingest
    python app/cli.py ingest --slug mctc-hosts-fire-drill --dry-run
    python app/cli.py rebuild-index

    python app/cli.py search "female graduation"
    python app/cli.py search "SCA agreement" --limit 5 --category "Partnerships & Agreements"
    python app/cli.py search "safety drill" --year 2026 --json

    python app/cli.py draft --topic "Maharat collaboration with Samsung E&A" --mode website_news --article-type partnership_announcement
    python app/cli.py draft --topic "graduation ceremony" --mode website_news --article-type graduation_story
    python app/cli.py draft --topic "Samsung partnership announcement" --mode linkedin_post
    python app/cli.py draft --topic "MCTC workforce development" --mode magazine_article
    python app/cli.py draft --topic "safety drill" --dry-run
    python app/cli.py draft --topic "OJT monitoring" --year 2026 --no-stream --mode event_announcement

    python app/cli.py evaluate
    python app/cli.py evaluate --verbose
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── command handlers ───────────────────────────────────────────────────────

def cmd_ingest(args):
    from pipelines.ingest_pipeline import IngestPipeline
    summary = IngestPipeline.from_config().run(
        slug_filter=args.slug or None,
        dry_run=args.dry_run,
        recreate=False,
    )
    print(f"\nSummary: {summary}")


def cmd_rebuild_index(args):
    from pipelines.ingest_pipeline import IngestPipeline
    print("Rebuilding index: dropping collection and re-ingesting all posts…")
    summary = IngestPipeline.from_config().run(recreate=True)
    print(f"\nSummary: {summary}")


def cmd_search(args):
    from pipelines.retrieval_pipeline import RetrievalPipeline
    results = RetrievalPipeline.from_config().retrieve(
        query=args.query,
        limit=args.limit or None,
        category=args.category or None,
        year=args.year or None,
        quarter=args.quarter or None,
        score_threshold=args.score_threshold or None,
    )

    if args.json:
        print(json.dumps(
            [{"id": r.id, "score": r.score, "payload": r.payload} for r in results],
            indent=2, ensure_ascii=False,
        ))
        return

    print(f"\nQuery: \"{args.query}\"  Results: {len(results)}\n" + "\u2500" * 72)
    for i, r in enumerate(results, 1):
        p     = r.payload or {}
        title = p.get("title", "\u2014")
        score = r.score
        print(f"{i:2d}. {title}")
        print(f"     Score: {score:.4f}")
        print(f"     slug: {p.get('slug', '')}  |  "
              f"category: {p.get('category', '')}  |  "
              f"date: {(p.get('date') or '')[:10]}")
        if p.get("tags"):
            print(f"     tags: {', '.join(p['tags'])}")
        excerpt = (p.get("chunk_text") or "")[:200]
        if excerpt:
            print(f"     Chunk: {excerpt}\u2026")
        print()


def cmd_draft(args):
    from pipelines.drafting_pipeline import DraftingPipeline
    pipeline = DraftingPipeline.from_config(model_override=args.model or None)
    result   = pipeline.draft(
        topic=args.topic,
        generation_mode=args.mode or None,
        article_type=args.article_type or None,
        category=args.category or None,
        year=args.year or None,
        limit=args.limit or None,
        score_threshold=args.score_threshold or None,
        dry_run=args.dry_run,
        stream=not args.no_stream,
    )

    if result and args.output:
        out_path = Path(args.output)
        out_path.write_text(result.article_text, encoding="utf-8")
        print(f"\nArticle text also written to {out_path}", file=sys.stderr)


def cmd_evaluate(args):
    from tests.test_retrieval import run_evaluation
    ok = run_evaluation(verbose=args.verbose)
    sys.exit(0 if ok else 1)


# ── argument parser ────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="maharat",
        description="Maharat News Pipeline \u2014 ingest / rebuild-index / search / draft / evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ──────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest",
        help="Validate, chunk, embed and upsert data/posts/ into Qdrant")
    p_ingest.add_argument("--dry-run", action="store_true",
        help="Parse and chunk without writing to Qdrant")
    p_ingest.add_argument("--slug", default="",
        help="Ingest only the post with this slug")

    # ── rebuild-index ────────────────────────────────────────────────────────
    sub.add_parser("rebuild-index",
        help="Drop Qdrant collection, recreate schema, re-ingest everything")

    # ── search ───────────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="Hybrid search over indexed content")
    p_search.add_argument("query",         help="Search query text")
    p_search.add_argument("--limit",       type=int, default=0,
        help="Max results (default: generation.yaml max_context_chunks)")
    p_search.add_argument("--category",    default="", help="Filter by category")
    p_search.add_argument("--year",        type=int, default=0, help="Filter by year")
    p_search.add_argument("--quarter",     default="",
        help="Filter by quarter (Q1 / Q2 / Q3 / Q4)")
    p_search.add_argument("--score-threshold", type=float, default=0.0,
        dest="score_threshold")
    p_search.add_argument("--json",        action="store_true",
        help="Output raw JSON")

    # ── draft ────────────────────────────────────────────────────────────────
    p_draft = sub.add_parser("draft",
        help="Generate a grounded article draft via RAG")
    p_draft.add_argument("--topic", required=True,
        help="Article topic or theme (required)")
    p_draft.add_argument("--mode", default="", dest="mode",
        metavar="MODE",
        help=(
            "Generation mode (from generation.yaml): "
            "website_news, linkedin_post, magazine_article, "
            "event_announcement, partner_highlight"
        ))
    p_draft.add_argument("--article-type", default="", dest="article_type",
        metavar="TYPE",
        help=(
            "Editorial article type (from editorial_style.yaml): "
            "partnership_announcement, graduation_story, training_program_update, "
            "event_story, recognition_story, leadership_news"
        ))
    p_draft.add_argument("--category", default="",
        help="Filter retrieved chunks to this category")
    p_draft.add_argument("--year",     type=int, default=0,
        help="Filter retrieved chunks to this year")
    p_draft.add_argument("--limit",    type=int, default=0,
        help="Number of chunks to retrieve")
    p_draft.add_argument("--score-threshold", type=float, default=0.0,
        dest="score_threshold")
    p_draft.add_argument("--model",    default="",
        help="Override Claude model (e.g. claude-opus-4-6)")
    p_draft.add_argument("--output",   default="",
        help="Also write article text to this file path")
    p_draft.add_argument("--dry-run",  action="store_true",
        help="Show retrieved chunks without calling Claude")
    p_draft.add_argument("--no-stream", action="store_true", dest="no_stream",
        help="Collect full response before printing")

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate",
        help="Run retrieval eval cases from tests/retrieval_eval.csv")
    p_eval.add_argument("--verbose", action="store_true",
        help="Print result for every case, not just failures")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    {
        "ingest":         cmd_ingest,
        "rebuild-index":  cmd_rebuild_index,
        "search":         cmd_search,
        "draft":          cmd_draft,
        "evaluate":       cmd_evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
