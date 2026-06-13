# Maharat News Pipeline

A Python system for managing Maharat/MCTC news content and knowledge:

1. **DOCX Pipeline** — Extract Word documents → Markdown posts → JSON feed
2. **RAG Ingestion** — Load posts and knowledge docs into Qdrant for hybrid search (dense + sparse)
3. **Article Drafting** — Claude-powered generation with dual-collection retrieval and knowledge graph context

All RAG commands route through the canonical entry point:

```bash
python3 app/cli.py <command>
```

For detailed architecture, config reference, and debugging notes, see [CLAUDE.md](CLAUDE.md).

---

## Quick Start

### Setup

```bash
pip3 install -r requirements.txt
```

Set your Anthropic API key before running draft commands:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

**Dependencies:** `python-docx`, `PyYAML`, `qdrant-client`, `fastembed`, `anthropic`

Embedding models (`BAAI/bge-small-en-v1.5`, BM25) are downloaded automatically on first use.

### Ingest & Search

```bash
# Index news posts from data/posts/
python3 app/cli.py rebuild-index

# Index knowledge base from data/knowledge/
python3 app/cli.py ingest-knowledge --recreate

# Hybrid search
python3 app/cli.py search "graduation ceremony" --year 2026
python3 app/cli.py search-knowledge "strategic partnerships" --limit 5
```

### Draft an Article

```bash
# News retrieval only
python3 app/cli.py draft --topic "Maharat training programs" --mode website_news

# Dual retrieval (news + knowledge + graph context)
python3 app/cli.py draft \
  --topic "Maharat strategy and partnerships" \
  --mode website_news \
  --use-knowledge

# Preview retrieved chunks without calling Claude
python3 app/cli.py draft --topic "safety drill" --dry-run --use-knowledge
```

### Evaluate Retrieval Quality

```bash
python3 app/cli.py evaluate              # news retrieval (10 cases)
python3 app/cli.py evaluate-knowledge      # knowledge retrieval (10 cases)
python3 app/cli.py evaluate-dual           # routing + dual retrieval (17 cases)
```

---

## RAG System

### Architecture

```
data/posts/*.md          ──► IngestPipeline      ──► maharat_content_live
data/knowledge/**/*.md   ──► KnowledgeIngest     ──► maharat_knowledge_live
                                                          │
User query ──► MemoryRouter ──► Hybrid search (dense + BM25, RRF fusion)
                    │                    │
                    └── Knowledge graph ─┘
                              │
                              ▼
                    DraftingPipeline ──► Claude API ──► outputs/drafts/
```

### Storage

Qdrant runs in **embedded local mode** at `./qdrant_storage` (configured in `config/qdrant.yaml`). Only one process can access the store at a time.

| Collection | Alias | Source |
|------------|-------|--------|
| `maharat_content_chunks_v1` | `maharat_content_live` | News posts (`data/posts/`) |
| `maharat_knowledge_memory_v1` | `maharat_knowledge_live` | Knowledge docs (`data/knowledge/`) |

**Vectors:** BAAI/bge-small-en-v1.5 (384-dim, cosine) + Qdrant BM25 (sparse), fused with Reciprocal Rank Fusion.

### CLI Commands

| Command | Description |
|---------|-------------|
| `ingest` | Validate, chunk, embed, and upsert `data/posts/` into Qdrant |
| `rebuild-index` | Drop collection, recreate schema, re-ingest all posts |
| `ingest-knowledge` | Ingest `data/knowledge/**/*.md` into the knowledge collection |
| `search` | Hybrid search over news content |
| `search-knowledge` | Hybrid search over knowledge docs |
| `draft` | Generate a grounded article via RAG |
| `route-query` | Show which memory layer(s) a query routes to |
| `evaluate` | Run news retrieval eval cases |
| `evaluate-knowledge` | Run knowledge retrieval eval cases |
| `evaluate-dual` | Run routing + dual-retrieval eval cases |
| `refresh-weekly-highlights` | Safe refresh of Weekly Highlights DOCX content |

### Draft Options

```bash
python3 app/cli.py draft --topic "..." \
  --mode website_news \           # website_news | linkedin_post | magazine_article |
                                  # event_announcement | partner_highlight
  --article-type graduation_story \  # partnership_announcement | graduation_story |
                                     # training_program_update | event_story |
                                     # recognition_story | leadership_news
  --year 2026 \
  --use-knowledge \
  --dry-run \
  --no-stream
```

Draft output is saved to `outputs/drafts/<topic-slug>/` as `draft.md`, `sources.json`, and `retrieval_debug.json`.

### Knowledge Base

25 Markdown files across 8 categories in `data/knowledge/`:

| Category | Examples |
|----------|----------|
| institutional | Corporate profile, strategy, governance |
| programs | Training overview, methodology, progression |
| partnerships | Strategic partners, MOUs |
| accreditation | Certifications, quality framework |
| editorial | Style rules, brand language |
| media | Quotes, story frameworks |
| faq | FAQ (skipped on ingest — no front matter) |
| Facility | Campus and facilities |

---

## DOCX Pipeline

Extracts news highlights from Word documents into structured Markdown and a publishable JSON feed.

```
input/*.docx  →  extract  →  normalize  →  export  →  output/feed.json
```

Run all three stages:

```bash
./run_pipeline.sh
./run_pipeline.sh --base-url https://maharat.com
./run_pipeline.sh --clean   # wipe output before running
```

The **Weekly Highlights refresh** command wraps extraction, normalization, feed export, and Qdrant re-ingestion in one idempotent flow:

```bash
python3 app/cli.py refresh-weekly-highlights --source input/weekly-highlights
python3 app/cli.py refresh-weekly-highlights --source input/weekly-highlights --dry-run
```

### Stage 1 — `extract_posts.py`

Reads every `.docx` in `input/`, splits at heading boundaries, extracts images, and writes:

| Output | Description |
|--------|-------------|
| `output/posts/<slug>.md` | One Markdown file per highlight with YAML front matter |
| `output/images/<hash>.<ext>` | Embedded images, named by content hash |
| `output/manifests/<doc>_manifest.json` | Per-document summary |
| `review/<doc>_review.csv` | Flat sheet for manual enrichment |

```bash
python3 scripts/extract_posts.py
python3 scripts/extract_posts.py --input "input/MyFile.docx"
python3 scripts/extract_posts.py --split-level 2
```

### Stage 2 — `normalize_posts.py`

Assigns category and tags from controlled vocabularies, cleans summaries, renames images, and validates metadata.

```bash
python3 scripts/normalize_posts.py
python3 scripts/normalize_posts.py --dry-run
```

**Categories:** Partnerships & Agreements, Accreditation & Compliance, Safety Campaigns & Drills, Competitions & Awards, Events & Ceremonies, Industry Visits & Site Tours, Staff Development, Academic & Examinations, On-the-Job Training, Trainee Programs, Campus & Facilities, Media & Publications.

### Stage 3 — `export_feed.py`

Writes JSON Feed 1.1 and a flat posts array:

| File | Format | Use |
|------|--------|-----|
| `output/feed.json` | [JSON Feed 1.1](https://jsonfeed.org/version/1.1/) | RSS readers, aggregators, headless CMS |
| `output/posts.json` | Flat posts array | REST API, static site generators |

```bash
python3 scripts/export_feed.py
python3 scripts/export_feed.py --base-url https://maharat.com
python3 scripts/export_feed.py --base-url https://maharat.com --no-body
```

### Post Schema

Fields are defined in `schema/news_schema.yaml`:

```
title, internal, slug, date, year, quarter
summary, body_markdown, category, tags
location, partner, featured_image, gallery_images
seo_title, seo_description
source_document, source_section, source_page
```

---

## Directory Layout

```
maharat-news-pipline/
├── app/
│   └── cli.py                  # Canonical entry point (all RAG commands)
├── pipelines/                  # Orchestration: ingest, retrieval, drafting, refresh
├── services/                   # Business logic: chunking, embedding, search, generation
├── config/                     # YAML configs (qdrant, chunking, taxonomy, generation, …)
├── data/
│   ├── posts/                  # News markdown posts (gitignored)
│   ├── images/                 # Post images (gitignored)
│   ├── knowledge/              # Institutional knowledge docs (25 files)
│   ├── graph/                  # Entity graph (entities.yaml, relationships.yaml)
│   ├── feed.json               # JSON feed (gitignored)
│   └── posts.json              # Flat post array (gitignored)
├── input/                      # Source .docx files (gitignored)
├── output/                     # Legacy DOCX pipeline output (gitignored)
├── outputs/                    # RAG draft output (draft.md, sources.json, debug)
├── qdrant_storage/             # Embedded Qdrant store (gitignored)
├── scripts/                    # Legacy DOCX + early RAG scripts
├── tests/                      # Retrieval eval runners + CSV fixtures
├── schema/
│   └── news_schema.yaml
├── run_pipeline.sh             # DOCX pipeline wrapper
├── CLAUDE.md                   # Full RAG system documentation
└── requirements.txt
```

---

## Configuration

All behavior is driven by YAML files in `config/`:

| File | Purpose |
|------|---------|
| `qdrant.yaml` | Qdrant client, collections, aliases, payload indexes |
| `chunking.yaml` | News chunking (max_tokens=700, overlap=100) |
| `knowledge_chunking.yaml` | Knowledge chunking (max_words=450, overlap=50) |
| `taxonomy.yaml` | 12 categories, 60-tag vocabulary |
| `generation.yaml` | Claude model, generation modes, article types |
| `editorial_style.yaml` | Headline patterns, article templates |
| `entities.yaml` | Entity regex patterns for extraction |

---

## Legacy Scripts

One-off scripts in `scripts/` exist for backwards compatibility. Prefer `app/cli.py` for all new workflows:

| Legacy script | Modern equivalent |
|---------------|-------------------|
| `ingest_markdown.py`, `embed_chunks.py`, `upsert_qdrant.py` | `app/cli.py ingest` |
| `search_qdrant.py` | `app/cli.py search` |
| `draft_article.py` | `app/cli.py draft` |
| `extract_posts.py`, `normalize_posts.py`, `export_feed.py` | `run_pipeline.sh` or `refresh-weekly-highlights` |
