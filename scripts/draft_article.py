#!/usr/bin/env python3
"""
Maharat RAG Article Drafter
============================
Retrieves the most relevant content chunks from Qdrant, then uses Claude
to draft a professional Maharat / MCTC news article grounded in those chunks.

Pipeline:
    User topic
        ↓
    Hybrid search  (search_qdrant.py)
        ↓
    Top-K chunks as grounding context
        ↓
    Claude Opus 4.6  (streaming, adaptive thinking)
        ↓
    Drafted article  (stdout  or  --output file)

Usage:
    python scripts/draft_article.py "partnership with Samsung"
    python scripts/draft_article.py "fire safety drill" --limit 10
    python scripts/draft_article.py "OJT monitoring" --category "On-the-Job Training"
    python scripts/draft_article.py "graduation ceremony" --output article.md
    python scripts/draft_article.py "Samsung agreement" --dry-run   # show context only
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_qdrant_config, make_client
from embed_chunks import Embedder
from search_qdrant import build_filter, search


# ── prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional news writer for Maharat Construction Training Center (MCTC),
a vocational training institution in Saudi Arabia that prepares trainees for careers
in construction, welding, pipefitting, scaffolding, instrumentation, and related trades.

Your task is to draft a polished, factual news article for the MCTC website or
newsletter. The article must:

  • Be grounded exclusively in the SOURCE CHUNKS provided — do not invent facts,
    names, dates, or statistics not present in the source material.
  • Open with a strong headline (# Heading) and a compelling lead paragraph.
  • Follow a clear news-article structure: headline → lead → body → closing quote
    or forward-looking statement.
  • Use a formal yet accessible tone — professional, positive, and institutional.
  • Be 250–450 words (excluding the headline).
  • If the source chunks mention specific names, dates, locations, or partner
    organisations, include them accurately.
  • If the source chunks are insufficient to write a full article, say so clearly
    and summarise what IS available.

Do NOT add a disclaimer, meta-commentary, or notes about the drafting process.
Output the article directly."""


def _format_chunks_as_context(chunks):
    """Format retrieved ScoredPoint objects into a readable context block."""
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


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Draft an MCTC news article using RAG + Claude."
    )
    parser.add_argument("topic",
        help="Topic or theme for the article (e.g. 'partnership with Samsung')")
    parser.add_argument("--limit", type=int, default=8,
        help="Number of chunks to retrieve from Qdrant (default 8)")
    parser.add_argument("--candidates", type=int, default=0,
        help="Candidate pool size for hybrid search (default: from config)")
    parser.add_argument("--category", default="",
        help="Filter search to a specific category")
    parser.add_argument("--score-threshold", type=float, default=0.0,
        help="Minimum relevance score for retrieved chunks")
    parser.add_argument("--collection", default="",
        help="Override Qdrant collection / alias name")
    parser.add_argument("--model", default="claude-opus-4-6",
        help="Claude model to use (default: claude-opus-4-6)")
    parser.add_argument("--output", default="",
        help="Write the drafted article to this file instead of stdout")
    parser.add_argument("--dry-run", action="store_true",
        help="Show retrieved chunks without calling Claude")
    args = parser.parse_args()

    # ── load config ──────────────────────────────────────────────────────
    qdrant_cfg = load_qdrant_config()
    qcfg       = qdrant_cfg["qdrant"]
    col_cfg    = qdrant_cfg["collections"]["primary"]
    retrieval  = qdrant_cfg.get("retrieval", {})

    collection_name = args.collection or col_cfg.get("live_alias") or col_cfg["name"]
    candidate_limit = args.candidates or retrieval.get("candidate_limit", 24)
    score_threshold = args.score_threshold or 0.0
    dense_model     = col_cfg["vectors"]["dense"].get("model", "BAAI/bge-small-en-v1.5")

    # ── STEP 1: retrieve relevant chunks ─────────────────────────────────
    print(f'\nSearching for: "{args.topic}"', file=sys.stderr)
    print(f"Collection   : {collection_name}", file=sys.stderr)

    client   = make_client(qdrant_cfg)
    embedder = Embedder(dense_model=dense_model)

    query_filter = build_filter(
        category=args.category or None,
    )

    chunks = search(
        query_text=args.topic,
        client=client,
        collection_name=collection_name,
        embedder=embedder,
        limit=args.limit,
        candidate_limit=candidate_limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

    print(f"Retrieved    : {len(chunks)} chunk(s)", file=sys.stderr)

    if not chunks:
        print(
            "\nNo relevant chunks found. Try a broader topic or lower --score-threshold.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── STEP 2: format context ───────────────────────────────────────────
    context_block = _format_chunks_as_context(chunks)

    if args.dry_run:
        print("\n" + "─" * 72)
        print("SOURCE CHUNKS (dry-run — Claude not called)")
        print("─" * 72)
        print(context_block)
        return

    # ── STEP 3: call Claude with streaming ───────────────────────────────
    user_message = (
        f"Draft a professional MCTC news article about: {args.topic}\n\n"
        f"SOURCE CHUNKS:\n\n{context_block}"
    )

    print(f"\nDrafting article with {args.model}…\n", file=sys.stderr)
    print("─" * 72, file=sys.stderr)

    anthropic_client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    article_parts = []

    with anthropic_client.messages.stream(
        model=args.model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            article_parts.append(text)

    article = "".join(article_parts)
    final   = stream.get_final_message()

    print(f"\n\n{'─' * 72}", file=sys.stderr)
    print(
        f"Tokens — input: {final.usage.input_tokens}  "
        f"output: {final.usage.output_tokens}",
        file=sys.stderr,
    )

    # ── STEP 4: optionally write to file ─────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(article, encoding="utf-8")
        print(f"\nArticle saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
