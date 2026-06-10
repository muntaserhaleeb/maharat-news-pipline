# Maharat News Pipeline — RAG System

## Overview

This is a three-tier Python/Anthropic system for managing Maharat's content and knowledge:

1. **DOCX Pipeline** — Extract Word documents → Markdown posts → JSON feed + images
2. **RAG Ingestion** — Load posts & knowledge docs into Qdrant for hybrid search (dense + sparse)
3. **Article Drafting** — Claude-powered generation with dual-collection retrieval + knowledge graph context

All commands route through the canonical entry point: `python3 app/cli.py <command>`.

---

## Quick Start

### Ingest
```bash
# News posts (fresh)
python3 app/cli.py rebuild-index

# Knowledge base
python3 app/cli.py ingest-knowledge --recreate
```

### Search
```bash
python3 app/cli.py search "graduation ceremony" --year 2026
python3 app/cli.py search-knowledge "strategic partnerships" --limit 5
```

### Draft
```bash
# News only
python3 app/cli.py draft --topic "Maharat training programs" --mode website_news

# Dual retrieval (news + knowledge)
python3 app/cli.py draft --topic "Maharat strategy and partnerships" --mode website_news --use-knowledge
```

### Route & Evaluate
```bash
python3 app/cli.py route-query "Write about Maharat and Sinopec"
python3 app/cli.py evaluate-dual --verbose
```

---

## Storage: Qdrant Embedded Local Store

**Location:** `./qdrant_storage` (configured in `config/qdrant.yaml`)  
**NOT** the HTTP server on `localhost:6333` — that's a separate test instance.

### Collections
| Name | Alias | Points | Source |
|---|---|---|---|
| `maharat_content_chunks_v1` | `maharat_content_live` | ~1,162 | News posts (`data/posts/`) |
| `maharat_knowledge_memory_v1` | `maharat_knowledge_live` | ~388 | Knowledge docs (`data/knowledge/`) |

### Vector Config
- **Dense:** BAAI/bge-small-en-v1.5 (384-dim, cosine distance)
- **Sparse:** Qdrant BM25 (via fastembed)
- **Hybrid search:** Reciprocal Rank Fusion (RRF) over dense + sparse results

---

## Directory Structure

```
maharat-news-pipline/
├── app/
│   └── cli.py                 # Canonical entry point (all commands)
├── pipelines/
│   ├── ingest_pipeline.py     # News posts → qdrant_content_chunks_v1
│   ├── knowledge_ingest_pipeline.py  # Knowledge docs → qdrant_knowledge_memory_v1
│   ├── retrieval_pipeline.py  # Search over content collection
│   ├── drafting_pipeline.py   # Full RAG: route → retrieve → generate
│   └── refresh_pipeline.py    # Safe refresh of Weekly Highlights DOCX
├── services/
│   ├── config_service.py      # Load configs; make_client() for Qdrant
│   ├── chunk_service.py       # Parse markdown, chunk, validate posts
│   ├── embedding_service.py   # Dense + sparse embeddings (fastembed)
│   ├── qdrant_service.py      # Collection setup, upsert, alias
│   ├── entity_service.py      # Entity extraction (config/entities.yaml)
│   ├── retrieval_service.py   # Hybrid search (Prefetch + RRF)
│   ├── generation_service.py  # Claude API calls + output save
│   ├── memory_router.py       # Route query → news / knowledge / both
│   ├── knowledge_graph_service.py # Entity graph (YAML-backed)
│   ├── citation_service.py    # Build citations from search results
│   ├── prompt_service.py      # Prompt building & formatting
│   └── ...
├── scripts/                   # Legacy one-off scripts (prefer app/cli.py)
├── config/
│   ├── qdrant.yaml            # Client, collections, aliases, indexes
│   ├── chunking.yaml          # News chunking: max_tokens=700, overlap=100
│   ├── knowledge_chunking.yaml# Knowledge: max_words=450, overlap=50
│   ├── taxonomy.yaml          # 12 categories, 60-tag vocab, category rules
│   ├── generation.yaml        # Claude model, generation modes, article types
│   ├── editorial_style.yaml   # Headlines, article templates
│   ├── entities.yaml          # Entity regex patterns
│   └── ...
├── data/
│   ├── posts/                 # Extracted markdown posts (gitignored)
│   ├── images/                # Post images (gitignored)
│   ├── knowledge/             # 8 categories, 25 markdown files
│   │   ├── institutional/     # Corporate profile, strategy, governance
│   │   ├── programs/          # Training overview, methodology, progression
│   │   ├── partnerships/      # Strategic partners, MOUs, collaboration
│   │   ├── accreditation/     # Certifications, quality framework
│   │   ├── editorial/         # Style rules, brand language, headlines
│   │   ├── media/             # Quotes, angles, magazine articles
│   │   ├── faq/               # FAQ (no front matter — skipped on ingest)
│   │   └── Facility/          # Campus & facilities
│   ├── graph/
│   │   ├── entities.yaml      # Entity definitions
│   │   ├── relationships.yaml # Entity relationships
│   │   └── knowledge_graph.json # Serialized graph (derived)
│   ├── feed.json              # JSON Feed 1.1 of all posts (gitignored)
│   └── posts.json             # Flat post array (gitignored)
├── tests/
│   ├── retrieval_eval.csv     # Retrieval eval cases (news)
│   ├── knowledge_eval.csv     # Knowledge retrieval eval cases
│   ├── dual_eval.csv          # Routing + dual-retrieval eval
│   └── test_*.py              # Eval runners
├── logs/                      # Failed records (jsonl)
├── run_pipeline.sh            # Bash wrapper: news ingest (3 stages)
├── CLAUDE.md                  # This file
└── README.md                  # Original project README

```

---

## Services Layer

**All services import from `config_service.py` for config loading; no global state.**

### config_service.py
- `load_qdrant_config()` — Parse `config/qdrant.yaml`
- `load_chunking_config()` — `chunking.yaml`
- `load_knowledge_chunking_config()` — `knowledge_chunking.yaml`
- `load_taxonomy()`, `load_generation_config()`, etc.
- **`make_client(qdrant_cfg=None)`** — Returns a `QdrantClient` (embedded local or HTTP, per config)

### chunk_service.py
- `parse_markdown(path)` → `(front, body)` — Parse YAML front matter + markdown
- `make_chunks(front, body, max_tokens, overlap_tokens)` — Split on headings, respect overlap
- `load_posts(posts_dir, slug_filter)` → `[(front, body), ...]`
- `validate_all(parsed, taxonomy, config)` → `(valid, failed)` — Metadata checks

### embedding_service.py
- `EmbeddingService.from_config(collection_cfg)` — Load dense + sparse models (one-time)
- `.embed_documents(texts)` → `(dense_vectors, sparse_vectors)` — Batch embed
- `.embed_query(text)` → `(dense, sparse)` — Single query embed (uses query prefix if model supports)

### qdrant_service.py
- `QdrantService.from_config(qdrant_cfg, collection_key="primary" | "knowledge")`
- `.setup_collection(recreate=False)` — Create or skip
- `.setup_payload_indexes()` — Define indexes (no-op in embedded mode)
- `.setup_alias()` — Create collection → alias mapping
- `.upsert_points(points, batch_size)` — Batch upsert
- `.get_collection_info()` — Counts, vector config

### entity_service.py
- `EntityService.from_config()` — Load compiled regex patterns from `config/entities.yaml`
- `.extract_from_article(front, body)` → `{organizations, programs, locations, credentials, people}`
- Canonical, deduplicated entity names per type

### retrieval_service.py
- `RetrievalService.from_config(qdrant_cfg, collection_key="primary" | "knowledge")`
- `.build_filter(category, year, status, language, ...)` — Build Qdrant filter struct
- `.search(query_text, limit, query_filter)` → `[ScoredPoint, ...]`
- Hybrid search: dense + sparse via `Prefetch` + RRF `FusionQuery`

### memory_router.py
- `MemoryRouter.from_config()` — Set up dual-collection retrieval
- `.route_query(query, intent=None)` → `RouteResult` (route: "news" | "knowledge" | "both", reasoning)
- `.retrieve(query)` → `RetrievalResult` — Full dual retrieval: news + knowledge + editorial + graph entities

### generation_service.py
- `GenerationService.from_config(model_override=None)`
- `.draft(topic, generation_mode, article_type, retrieval_result, ...)` → `DraftResult`
- Calls Claude API, structures response as JSON (article_text, sources, metadata)
- Saves `output/{topic}-draft.md`, `sources.json`, `retrieval_debug.json`

### knowledge_graph_service.py
- `KnowledgeGraphService.from_config()` — Load `data/graph/entities.yaml` + `relationships.yaml`
- `.get_entity_profile(name)` → `{id, type, aliases, ...}`
- `.get_related_entities(entity_id)` → List of related entities
- `.build_context_block(entity_id)` → Prompt-ready context string

---

## Pipelines

### ingest_pipeline.py : `IngestPipeline`
**News posts → `maharat_content_chunks_v1`**

```bash
python3 app/cli.py ingest [--slug SLUG] [--dry-run]
python3 app/cli.py rebuild-index  # drop + recreate
```

6-stage: discover → load → validate → chunk → embed → upsert → verify  
Source: `data/posts/` (markdown, YAML front matter)  
Output: `logs/failed_records.jsonl` (if validation fails)

### knowledge_ingest_pipeline.py : `KnowledgeIngestPipeline`
**Knowledge docs → `maharat_knowledge_memory_v1`**

```bash
python3 app/cli.py ingest-knowledge [--dry-run] [--recreate]
```

6-stage: discover → parse → validate → chunk → embed → upsert → verify  
Source: `data/knowledge/**/*.md` (25 files, 8 categories)  
Output: `logs/failed_knowledge.jsonl` (skipped files)  
**Note:** 6 files fail validation due to missing/non-standard front matter (faq.md, accreditation_entities.md, media/*.md) — gracefully skipped, not fatal.

### retrieval_pipeline.py : `RetrievalPipeline`
**Hybrid search over content collection**

```bash
python3 app/cli.py search "query text" [--limit 10] [--category CAT] [--year YEAR] [--json]
```

Filters: category, year, quarter, status, published  
Returns: `[ScoredPoint, ...]` (id, score, payload with chunk_text, metadata)

### drafting_pipeline.py : `DraftingPipeline`
**Full RAG: route query → retrieve from 1–2 collections → call Claude → save draft**

```bash
python3 app/cli.py draft --topic "..." [--mode MODE] [--article-type TYPE] [--year YEAR] [--use-knowledge] [--dry-run] [--no-stream]
```

Flow:
1. Route query via `MemoryRouter` (news / knowledge / both)
2. Retrieve from appropriate collections + knowledge graph
3. Call Claude with grounded prompt (generation mode template + article type)
4. Stream response + save to `output/{topic}-draft.md`, `sources.json`, `retrieval_debug.json`

Generation modes: `website_news`, `linkedin_post`, `magazine_article`, `event_announcement`, `partner_highlight`  
Article types: `partnership_announcement`, `graduation_story`, `training_program_update`, `event_story`, `recognition_story`, `leadership_news`

### refresh_pipeline.py : `RefreshPipeline`
**Safe, idempotent refresh of Weekly Highlights DOCX content**

```bash
python3 app/cli.py refresh-weekly-highlights --source input/weekly-highlights [--backup true|false] [--delete-existing true|false] [--reinsert true|false] ...
```

---

## Config Files

### qdrant.yaml
```yaml
qdrant:
  path: "qdrant_storage"    # Embedded local; relative to project root
  url: null                 # Or HTTP URL if using server
  api_key: null

collections:
  primary:                  # News
    name: "maharat_content_chunks_v1"
    live_alias: "maharat_content_live"
    vector_size: 384
    distance: "cosine"
    ...
  knowledge:                # Knowledge docs
    name: "maharat_knowledge_memory_v1"
    live_alias: "maharat_knowledge_live"
    ...

payload_indexes:            # Index definitions (no-op in embedded mode)
  - {field_name: "category", field_schema: "keyword"}
  - {field_name: "status", field_schema: "keyword"}
  - ...
```

### chunking.yaml & knowledge_chunking.yaml
News: `max_tokens=700, overlap_tokens=100, strategy=markdown_headings`  
Knowledge: `max_words=450, overlap_words=50, strategy=markdown_headings`

### taxonomy.yaml
- 12 ordered categories (specifics first, broad last) — first match wins
- 60-entry tag vocabulary
- Category rules: keyword matching + phrase patterns

### generation.yaml
- Claude model (default: `claude-opus-4-8`)
- Generation modes: prompts + formatting rules
- Article types: templates + guidelines

### editorial_style.yaml
- Headline patterns (category-specific)
- Article type rules (length, tone, structure)

### entities.yaml
Entity regex patterns for extraction:
- Organizations (company names, acronyms)
- Programs (course/training names)
- Locations (cities, regions, facilities)
- Credentials (certifications, standards)
- People (roles, titles)

---

## Knowledge Base Structure

**Location:** `data/knowledge/`  
**Files:** 25 Markdown files across 8 categories

### Required Front Matter
```yaml
---
title: "..."
slug: "..."
knowledge_type: "institutional_profile" | "program_catalog" | ...
status: "approved" | "draft"
published: true
language: "en"
priority: "high" | "medium" | "low"
memory_layer: "knowledge"
---
```

### Categories
| Category | Docs | knowledge_type examples |
|---|---|---|
| **institutional** | 5 | institutional_profile, strategy, governance |
| **programs** | 4 | program_catalog, delivery_model, learner_progression |
| **partnerships** | 4 | partner_registry, agreement_registry, institutional_ecosystem |
| **accreditation** | 2 | institutional_accreditation, quality_assurance |
| **editorial** | 3 | editorial_guidelines |
| **media** | 3 | editorial_quote_library, editorial_story_framework (skipped — bad front matter) |
| **faq** | 1 | (no front matter — skipped) |
| **Facility** | 1 | institutional_infrastructure |

---

## CLI Commands Reference

```bash
# Ingest / Rebuild
python3 app/cli.py ingest [--slug SLUG] [--dry-run]
python3 app/cli.py rebuild-index
python3 app/cli.py ingest-knowledge [--dry-run] [--recreate]

# Search
python3 app/cli.py search "query" [--limit 10] [--category CAT] [--year YEAR] [--quarter Q1-Q4] [--score-threshold SCORE] [--json]
python3 app/cli.py search-knowledge "query" [--limit 8] [--knowledge-type TYPE] [--language en] [--priority high|medium|low] [--json]

# Draft
python3 app/cli.py draft --topic "..." [--mode website_news|linkedin_post|magazine_article|event_announcement|partner_highlight] [--article-type TYPE] [--category CAT] [--year YEAR] [--limit N] [--score-threshold SCORE] [--model MODEL] [--output FILE] [--dry-run] [--no-stream] [--use-knowledge]

# Route & Graph
python3 app/cli.py route-query "query" [--intent news|knowledge|both]

# Evaluate
python3 app/cli.py evaluate [--verbose]                  # News retrieval
python3 app/cli.py evaluate-knowledge [--verbose]        # Knowledge retrieval
python3 app/cli.py evaluate-dual [--verbose]             # Routing + dual retrieval

# Refresh
python3 app/cli.py refresh-weekly-highlights --source input/weekly-highlights [--dry-run] [--backup true|false] [--delete-existing true|false] [--reinsert true|false] [--regenerate-image-metadata true|false] [--create-liferay-manifest true|false] [--base-url URL]
```

---

## Python / Environment

- **Python:** 3.9.6 (no `X | Y` union syntax — use `Optional[X]`)
- **Tools:** `python3` and `pip3` (not `python`/`pip`)
- **GitHub CLI:** `/opt/homebrew/bin/gh`

### Key Dependencies
```
qdrant-client>=1.7.0           # Embedded + HTTP Qdrant client
fastembed>=0.3.0               # Dense + sparse embeddings
anthropic>=0.40.0              # Claude API
python-docx>=1.1.0             # DOCX extraction
PyYAML>=6.0                    # Config parsing
```

---

## Common Patterns

### Single-File Ingest (for testing)
```bash
# News post
python3 app/cli.py ingest --slug my-post-slug

# Knowledge doc — no CLI option; manually edit data/knowledge and run:
python3 app/cli.py ingest-knowledge --recreate
```

### Dual-Collection Retrieval
```bash
python3 app/cli.py draft --topic "Maharat and Samsung partnership" --mode website_news --use-knowledge
```
This retrieves from both `maharat_content_live` and `maharat_knowledge_live`, merges results, and adds knowledge graph context.

### Dry-Run Before Commit
```bash
python3 app/cli.py ingest --dry-run               # Parse + validate, no upsert
python3 app/cli.py draft --topic "..." --dry-run  # Show retrieved chunks, no Claude call
```

### Filter by Metadata
```bash
# Year
python3 app/cli.py search "safety drill" --year 2026

# Category
python3 app/cli.py search "graduation" --category "Training & Certification" --limit 5

# Knowledge type
python3 app/cli.py search-knowledge "partnerships" --knowledge-type partner_registry

# Score threshold (returns only high-confidence matches)
python3 app/cli.py search "female workforce" --score-threshold 0.7
```

---

## Debugging

### Check Ingestion Failures
```bash
cat logs/failed_records.jsonl     # News posts
cat logs/failed_knowledge.jsonl   # Knowledge docs
```

### Inspect Qdrant State
```bash
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from config import make_client
c = make_client()
for coll in [x.name for x in c.get_collections().collections]:
    cnt = c.count(coll).count
    print(f'{coll}: {cnt} points')
for alias in c.get_aliases().aliases:
    print(f'{alias.alias_name} -> {alias.collection_name}')
"
```

### Verify Search Quality
```bash
# Check top result for a query
python3 app/cli.py search "fire safety drill" --limit 3 --json | python3 -m json.tool
```

### Test Draft Flow (dry-run)
```bash
python3 app/cli.py draft --topic "Maharat female employees" --mode website_news --dry-run --use-knowledge
```
Shows retrieved chunks without calling Claude.

---

## Gotchas & Notes

1. **Qdrant Store:** Embedded local (`./qdrant_storage`) — not the HTTP server. To use a real Qdrant server, change `config/qdrant.yaml` to set `url:` instead of `path:`.

2. **Payload Indexes:** No-op in embedded mode (logged warning during ingest). Filtering still works; indexes only accelerate server mode.

3. **Knowledge Files:** 6 files fail front-matter validation and are skipped (faq.md, accreditation_entities.md, media/*.md). Intentional; these files have non-standard formats. No action needed — ingest logs them and continues.

4. **Category Mismatch:** News posts use pipeline-derived categories (e.g., "Safety Campaigns & Drills"); knowledge docs use config-defined `knowledge_type` (e.g., "institutional_profile"). Routes are independent; no alignment issue.

5. **Git & Files:** `data/posts/`, `data/images/`, `data/feed.json`, etc. are gitignored. Only `scripts/`, `config/`, `services/`, `app/`, `pipelines/`, `tests/`, `CLAUDE.md`, `README.md` are tracked.

6. **Anthropic API:** Uses `ANTHROPIC_API_KEY` env var. Set before running `draft` commands.

---

## Legacy Scripts (`scripts/`)

One-off scripts exist for backwards compatibility but are not the canonical path:
- `extract_posts.py`, `normalize_posts.py`, `export_feed.py` — news extraction pipeline (use `run_pipeline.sh` wrapper)
- `ingest_markdown.py`, `chunk_markdown.py`, `embed_chunks.py`, `upsert_qdrant.py`, `search_qdrant.py` — early RAG (prefer `app/cli.py`)

**Use `app/cli.py` for all new workflows.**

---

**Last updated:** 2026-06-10  
**Python:** 3.9.6  
**Key models:** BAAI/bge-small-en-v1.5 (dense), Qdrant/bm25 (sparse), claude-opus-4-8 (generation)
