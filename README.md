# Maharat News Pipeline

A three-stage Python pipeline that turns a Word document of MCTC news highlights into structured Markdown posts, normalised metadata, and a publishable JSON feed.

```
input/*.docx  →  extract  →  normalize  →  export  →  output/feed.json
```

### Results on MCTC Highlights 2026

| Metric | Count |
|--------|------:|
| Posts extracted | 76 |
| Posts with date | 75 / 76 |
| Posts with featured image | 44 / 76 |
| Posts with tags | 75 / 76 |
| Images extracted & renamed | 94 |
| Feed size (feed.json) | 146 KB |
| Feed size (posts.json) | 225 KB |

**By category**

| Category | Posts |
|----------|------:|
| Trainee Programs | 12 |
| Safety Campaigns & Drills | 10 |
| Competitions & Awards | 9 |
| Events & Ceremonies | 8 |
| Staff Development | 8 |
| Partnerships & Agreements | 7 |
| Media & Publications | 5 |
| Accreditation & Compliance | 5 |
| Industry Visits & Site Tours | 3 |
| On-the-Job Training | 3 |
| Campus & Facilities | 2 |
| Academic & Examinations | 1 |
| General | 3 |

---

## Pipeline stages

### 1. `extract_posts.py` — DOCX → Markdown

Reads every `.docx` in `input/`, splits the document into sections at heading boundaries, extracts embedded images, and writes:

| Output | Description |
|--------|-------------|
| `output/posts/<slug>.md` | One Markdown file per highlight with YAML front matter |
| `output/images/<hash>.<ext>` | All embedded images, named by content hash |
| `output/manifests/<doc>_manifest.json` | Per-document summary |
| `review/<doc>_review.csv` | Flat sheet for manual enrichment |

```bash
python scripts/extract_posts.py
python scripts/extract_posts.py --input "input/MyFile.docx"
python scripts/extract_posts.py --split-level 2   # if stories start at Heading 2
```

---

### 2. `normalize_posts.py` — Metadata normalisation

Reads all posts from `output/posts/`, then:

- Assigns one **category** from a controlled list
- Generates **3–7 tags** from a controlled vocabulary
- Cleans **summaries** to a sentence boundary (≤ 200 chars)
- Renames images to `{slug}-01.jpg`, `{slug}-02.jpg`, … and updates references
- Validates missing dates, missing images, low tag counts, and duplicate slugs
- Writes `output/manifests/publishing_manifest.json` and `review/normalize_review.csv`

```bash
python scripts/normalize_posts.py
python scripts/normalize_posts.py --dry-run   # preview without writing
```

**Categories**

| Category | Examples |
|----------|---------|
| Partnerships & Agreements | MoUs, training agreements, accreditations |
| Accreditation & Compliance | ETEC, TVTC, HRDF, Saudi Aramco surveys |
| Safety Campaigns & Drills | Fire drills, safety awareness campaigns |
| Competitions & Awards | Welding competitions, football tournaments |
| Events & Ceremonies | Graduations, Iftar gatherings, festivals |
| Industry Visits & Site Tours | Site visits, benchmarking trips |
| Staff Development | CPD sessions, Classera training, LeadXera |
| Academic & Examinations | Midterms, end-of-semester exams |
| On-the-Job Training | OJT deployment, pre-OJT visits |
| Trainee Programs | WPR training, CSM course, 5G welding |
| Campus & Facilities | Renovations, digital transformation |
| Media & Publications | Press coverage |

---

### 3. `export_feed.py` — JSON feed export

Reads all normalised posts and writes two feed files:

| File | Format | Use |
|------|--------|-----|
| `output/feed.json` | [JSON Feed 1.1](https://jsonfeed.org/version/1.1/) | RSS readers, news aggregators, headless CMS |
| `output/posts.json` | Flat posts array | REST API, static site generators, search index |

```bash
python scripts/export_feed.py
python scripts/export_feed.py --base-url https://maharat.com   # absolute image/post URLs
python scripts/export_feed.py --base-url https://maharat.com --no-body  # omit HTML body
```

Each post in `posts.json` includes:

```json
{
  "slug": "mctc-hosts-fire-drill",
  "title": "MCTC Hosts Fire Drill Conducted by its Training Provider",
  "date": "2026-01-06",
  "date_published": "2026-01-06T00:00:00+00:00",
  "category": "Safety Campaigns & Drills",
  "tags": ["fire-safety", "nesma", "nhti"],
  "summary": "On January 6, 2026, Nesma High Training Institute conducted a fire drill…",
  "featured_image": "https://maharat.com/images/mctc-hosts-fire-drill-01.jpg",
  "gallery_images": ["…-02.jpg", "…-03.jpg"],
  "body_markdown": "…",
  "body_html": "…"
}
```

---

## Schema

Post fields are defined in `schema/news_schema.yaml`:

```
title, internal, slug, date, year, quarter
summary, body_markdown, category, tags
location, partner, featured_image, gallery_images
seo_title, seo_description
source_document, source_section, source_page
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `python-docx`, `PyYAML` — no other packages required.

---

## Directory layout

```
maharat-news-pipline/
├── input/                  # Source .docx files (gitignored)
├── output/
│   ├── posts/              # Extracted Markdown posts (gitignored)
│   ├── images/             # Extracted images (gitignored)
│   ├── manifests/          # JSON manifests (gitignored)
│   ├── feed.json           # JSON Feed 1.1 (gitignored)
│   └── posts.json          # Flat posts array (gitignored)
├── review/                 # CSV review sheets (gitignored)
├── schema/
│   └── news_schema.yaml
├── scripts/
│   ├── extract_posts.py
│   ├── normalize_posts.py
│   └── export_feed.py
└── requirements.txt
```
