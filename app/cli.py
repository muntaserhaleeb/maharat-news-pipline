#!/usr/bin/env python3
"""
Maharat News Pipeline CLI
==========================
Single entry point — calls pipelines only.

Commands:
    ingest                 Validate, chunk, embed and upsert data/posts/ into Qdrant
    rebuild-index          Drop collection, recreate schema, re-ingest everything
    search                 Hybrid search over indexed content
    draft                  Generate a grounded article draft via RAG
    evaluate               Run retrieval eval cases from tests/retrieval_eval.csv
    ingest-knowledge       Ingest data/knowledge/**/*.md into the knowledge collection
    search-knowledge       Hybrid search over the knowledge collection
    route-query            Show which memory layer(s) a query routes to
    evaluate-knowledge     Run knowledge retrieval eval cases
    evaluate-dual          Run routing + dual-retrieval eval cases
    refresh-weekly-highlights  Safe refresh of Weekly Highlights DOCX content

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
    python app/cli.py draft --topic "Maharat collaboration with Sinopec" --mode website_news --use-knowledge

    python app/cli.py route-query "Write an article about Maharat and Sinopec"
    python app/cli.py route-query "What is Maharat?" --intent knowledge

    python app/cli.py evaluate-knowledge
    python app/cli.py evaluate-knowledge --verbose

    python app/cli.py evaluate-dual
    python app/cli.py evaluate-dual --verbose

    python app/cli.py evaluate
    python app/cli.py evaluate --verbose

    python app/cli.py ingest-knowledge
    python app/cli.py ingest-knowledge --dry-run
    python app/cli.py ingest-knowledge --recreate

    python app/cli.py search-knowledge "What is Maharat?"
    python app/cli.py search-knowledge "training methodology" --knowledge-type training_methodology
    python app/cli.py search-knowledge "strategic partnerships" --limit 5

    python app/cli.py refresh-weekly-highlights --source input/weekly-highlights
    python app/cli.py refresh-weekly-highlights --source input/weekly-highlights --dry-run
    python app/cli.py refresh-weekly-highlights --source input/weekly-highlights \\
        --backup true --delete-existing true --reinsert true \\
        --regenerate-image-metadata true --create-liferay-manifest true
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
        use_knowledge=args.use_knowledge,
    )

    if result and args.output:
        out_path = Path(args.output)
        out_path.write_text(result.article_text, encoding="utf-8")
        print(f"\nArticle text also written to {out_path}", file=sys.stderr)


def cmd_evaluate(args):
    from tests.test_retrieval import run_evaluation
    ok = run_evaluation(verbose=args.verbose)
    sys.exit(0 if ok else 1)


def cmd_route_query(args):
    from services.memory_router import MemoryRouter
    _ROUTE_COLLECTIONS = {
        "news":      ["maharat_content_live"],
        "knowledge": ["maharat_knowledge_live"],
        "both":      ["maharat_content_live", "maharat_knowledge_live"],
    }
    router = MemoryRouter.from_config()
    result = router.route_query(args.query, intent=args.intent or None)
    entities = router._detect_graph_entities(args.query)

    print(f"\nQuery       : {args.query}")
    print(f"Route       : {result.route}")
    print(f"Reason      : {result.reasoning}")
    collections = _ROUTE_COLLECTIONS.get(result.route, [])
    if collections:
        print(f"Collections : {', '.join(collections)}")
    if entities:
        enames = [e.get("name", e.get("id", "?")) for e in entities]
        print(f"Graph entities: {', '.join(enames)}")


def cmd_evaluate_knowledge(args):
    from tests.test_knowledge_retrieval import run_knowledge_evaluation
    ok = run_knowledge_evaluation(verbose=args.verbose)
    sys.exit(0 if ok else 1)


def cmd_evaluate_dual(args):
    from tests.test_dual_retrieval import run_dual_evaluation
    ok = run_dual_evaluation(verbose=args.verbose)
    sys.exit(0 if ok else 1)


def cmd_ingest_knowledge(args):
    from pipelines.knowledge_ingest_pipeline import KnowledgeIngestPipeline
    summary = KnowledgeIngestPipeline.from_config().run(
        dry_run=args.dry_run,
        recreate=args.recreate,
    )
    print(f"\nSummary: {summary}")


def cmd_search_knowledge(args):
    from services.retrieval_service import RetrievalService
    from services.config_service import load_qdrant_config

    qdrant_cfg = load_qdrant_config()
    service    = RetrievalService.from_config(qdrant_cfg, collection_key="knowledge")

    query_filter = service.build_filter(
        knowledge_type=args.knowledge_type or None,
        language=args.language or None,
        priority=args.priority or None,
        published=True,
        status="approved",
    )
    results = service.search(
        query_text=args.query,
        limit=args.limit or 8,
        query_filter=query_filter,
    )

    if args.json:
        print(json.dumps(
            [{"id": r.id, "score": r.score, "payload": r.payload} for r in results],
            indent=2, ensure_ascii=False,
        ))
        return

    print(f"\nQuery: \"{args.query}\"  Results: {len(results)}\n" + "─" * 72)
    for i, r in enumerate(results, 1):
        p    = r.payload or {}
        print(f"{i:2d}. {p.get('title', '—')}")
        print(f"     Score        : {r.score:.4f}")
        print(f"     slug         : {p.get('slug', '')}")
        print(f"     knowledge_type: {p.get('knowledge_type', '')}")
        print(f"     section      : {p.get('section', '')}")
        excerpt = (p.get("chunk_text") or "")[:220]
        if excerpt:
            print(f"     Chunk: {excerpt}…")
        print()


def _str_to_bool(v: str) -> bool:
    """Accept 'true'/'false' strings as boolean values."""
    return str(v).strip().lower() in ("true", "yes", "1")


def cmd_refresh_weekly_highlights(args):
    from pathlib import Path
    from pipelines.refresh_pipeline import RefreshPipeline

    source_dir = Path(args.source)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir

    pipeline = RefreshPipeline.from_config(source_dir=source_dir)
    report = pipeline.run(
        dry_run=args.dry_run,
        backup=args.backup,
        delete_existing=args.delete_existing,
        reinsert=args.reinsert,
        regenerate_image_metadata=args.regenerate_image_metadata,
        create_liferay_manifest=args.create_liferay_manifest,
        base_url=args.base_url or "",
    )

    failed = report.get("issues", {}).get("failed_files", [])
    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for f in failed:
            print(f"  {f['file']}: {f['error']}")

    sys.exit(1 if failed else 0)


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
    p_draft.add_argument("--use-knowledge", action="store_true", dest="use_knowledge",
        help="Also retrieve from the knowledge collection (dual retrieval)")

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate",
        help="Run retrieval eval cases from tests/retrieval_eval.csv")
    p_eval.add_argument("--verbose", action="store_true",
        help="Print result for every case, not just failures")

    # ── route-query ───────────────────────────────────────────────────────────
    p_rq = sub.add_parser("route-query",
        help="Show which memory layer(s) a query would route to")
    p_rq.add_argument("query", help="Query text to classify")
    p_rq.add_argument("--intent", default="",
        help="Explicit intent override: news | knowledge | both")

    # ── evaluate-knowledge ────────────────────────────────────────────────────
    p_ek = sub.add_parser("evaluate-knowledge",
        help="Run retrieval eval cases against the knowledge collection")
    p_ek.add_argument("--verbose", action="store_true",
        help="Print result for every case, not just failures")

    # ── evaluate-dual ─────────────────────────────────────────────────────────
    p_ed = sub.add_parser("evaluate-dual",
        help="Run routing and dual-retrieval eval cases")
    p_ed.add_argument("--verbose", action="store_true",
        help="Print result for every case, not just failures")

    # ── ingest-knowledge ─────────────────────────────────────────────────────
    p_ik = sub.add_parser("ingest-knowledge",
        help="Ingest data/knowledge/**/*.md into the knowledge Qdrant collection")
    p_ik.add_argument("--dry-run", action="store_true",
        help="Parse and chunk without writing to Qdrant")
    p_ik.add_argument("--recreate", action="store_true",
        help="Drop and recreate the knowledge collection before ingesting")

    # ── search-knowledge ─────────────────────────────────────────────────────
    p_sk = sub.add_parser("search-knowledge",
        help="Hybrid search over the knowledge collection")
    p_sk.add_argument("query",              help="Search query text")
    p_sk.add_argument("--limit",            type=int, default=0,
        help="Max results (default: 8)")
    p_sk.add_argument("--knowledge-type",   default="", dest="knowledge_type",
        help="Filter by knowledge_type (e.g. institutional_profile)")
    p_sk.add_argument("--language",         default="",
        help="Filter by language code (default: no filter)")
    p_sk.add_argument("--priority",         default="",
        help="Filter by priority (high / medium / low)")
    p_sk.add_argument("--json",             action="store_true",
        help="Output raw JSON")

    # ── refresh-weekly-highlights ─────────────────────────────────────────
    p_rwh = sub.add_parser(
        "refresh-weekly-highlights",
        help="Safe, idempotent refresh of Weekly Highlights DOCX content",
    )
    p_rwh.add_argument(
        "--source", required=True,
        metavar="DIR",
        help="Directory containing the revised Weekly Highlights .docx files "
             "(e.g. input/weekly-highlights)",
    )
    p_rwh.add_argument("--dry-run", action="store_true",
        help="Show what would change without writing any files")
    p_rwh.add_argument("--backup", type=_str_to_bool, default=True,
        metavar="BOOL",
        help="Backup existing posts and images before deletion (default: true)")
    p_rwh.add_argument("--delete-existing", type=_str_to_bool, default=True,
        dest="delete_existing", metavar="BOOL",
        help="Delete existing matching posts before reinserting (default: true)")
    p_rwh.add_argument("--reinsert", type=_str_to_bool, default=True,
        metavar="BOOL",
        help="Upsert newly extracted posts to Qdrant (default: true)")
    p_rwh.add_argument("--regenerate-image-metadata", type=_str_to_bool, default=True,
        dest="regenerate_image_metadata", metavar="BOOL",
        help="Run category/tags/summary assignment and image renaming (default: true)")
    p_rwh.add_argument("--create-liferay-manifest", type=_str_to_bool, default=True,
        dest="create_liferay_manifest", metavar="BOOL",
        help="Generate Liferay-ready JSON + CSV manifest (default: true)")
    p_rwh.add_argument("--base-url", default="", dest="base_url",
        help="Base URL for absolute image/post URLs in feeds")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    {
        "ingest":                      cmd_ingest,
        "rebuild-index":               cmd_rebuild_index,
        "search":                      cmd_search,
        "draft":                       cmd_draft,
        "evaluate":                    cmd_evaluate,
        "ingest-knowledge":            cmd_ingest_knowledge,
        "search-knowledge":            cmd_search_knowledge,
        "route-query":                 cmd_route_query,
        "evaluate-knowledge":          cmd_evaluate_knowledge,
        "evaluate-dual":               cmd_evaluate_dual,
        "refresh-weekly-highlights":   cmd_refresh_weekly_highlights,
    }[args.command](args)


if __name__ == "__main__":
    main()
