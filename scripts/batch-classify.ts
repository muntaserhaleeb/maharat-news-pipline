#!/usr/bin/env tsx
/**
 * Classifies every markdown post under data/posts/ and writes individual
 * metadata JSON files to output/classification/.
 *
 * Usage:
 *   tsx scripts/batch-classify.ts [--posts-dir <dir>]
 */

import * as fs from 'fs';
import * as path from 'path';

import { loadEntityGraph, loadTaxonomy, loadClassificationRules } from '../src/lib/classification/loadTaxonomy';
import { detectEntities } from '../src/lib/classification/detectEntities';
import { classifyContent } from '../src/lib/classification/classifyContent';
import { ContentMetadata, ClassificationSource } from '../src/lib/classification/types';
import matter from 'gray-matter';

// ---------------------------------------------------------------------------
// Inline helpers (avoid re-importing generateContentMetadata so we load
// configs only once for the whole batch)
// ---------------------------------------------------------------------------

const PROJECT_ROOT = path.resolve(__dirname, '..');

function stripMarkdown(text: string): string {
  return text
    .replace(/#{1,6}\s+/g, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`[^`]+`/g, '')
    .replace(/^[-*+]\s+/gm, '')
    .replace(/^\d+\.\s+/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function extractSummary(body: string, maxWords = 60): string {
  const plain = stripMarkdown(body);
  const sentences = plain.split(/(?<=[.!?])\s+/);
  let summary = '';
  let wordCount = 0;
  for (const sentence of sentences) {
    const words = sentence.trim().split(/\s+/);
    if (wordCount + words.length > maxWords && summary.length > 0) break;
    summary += (summary ? ' ' : '') + sentence.trim();
    wordCount += words.length;
    if (wordCount >= 30) break;
  }
  return summary || plain.slice(0, 250).trimEnd();
}

// ---------------------------------------------------------------------------
// Batch run
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const postsDirArg = args.indexOf('--posts-dir');
const postsDir = postsDirArg >= 0
  ? path.resolve(args[postsDirArg + 1])
  : path.join(PROJECT_ROOT, 'data', 'posts');

if (!fs.existsSync(postsDir)) {
  console.error(`Posts directory not found: ${postsDir}`);
  process.exit(1);
}

const outputDir = path.join(PROJECT_ROOT, 'output', 'classification');
fs.mkdirSync(outputDir, { recursive: true });

console.log(`\nLoading configs…`);
const entityGraph = loadEntityGraph();
const taxonomy = loadTaxonomy();
const rules = loadClassificationRules();
console.log(`  ✓ entity graph   (${
  Object.values(entityGraph.entities).reduce((n, a) => n + a.length, 0)
} entries)`);
console.log(`  ✓ taxonomy       (${taxonomy.primary_categories.length} categories)`);
console.log(`  ✓ rules          (${rules.classification_rules.keyword_fallback_rules.length} keyword rules)\n`);

const posts = fs
  .readdirSync(postsDir)
  .filter(f => f.endsWith('.md'))
  .sort();

console.log(`Found ${posts.length} posts in ${postsDir}\n`);

// Results accumulator for summary table
type Row = {
  slug: string;
  category: string;
  subcategory: string;
  source: ClassificationSource;
  catScore: number;
  entities: number;
  ok: boolean;
  error?: string;
};

const rows: Row[] = [];
const categoryCounts: Record<string, number> = {};
const sourceCounts: Record<string, number> = {};
let saved = 0;
let failed = 0;

const padEnd = (s: string, n: number) => s.slice(0, n).padEnd(n);

// Header
console.log(
  `${'#'.padStart(3)}  ${padEnd('Slug', 55)}  ${padEnd('Category', 28)}  ${'Src'.padEnd(14)}  Conf`,
);
console.log('─'.repeat(115));

for (let i = 0; i < posts.length; i++) {
  const filename = posts[i];
  const filePath = path.join(postsDir, filename);
  const num = String(i + 1).padStart(3);

  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    const { data, content } = matter(raw);

    const title = String(data['title'] ?? path.basename(filename, '.md'));
    const slug = String(data['slug'] ?? path.basename(filename, '.md'));

    let date: string | null = null;
    const rawDate = data['date'];
    if (rawDate) {
      date = rawDate instanceof Date
        ? rawDate.toISOString().split('T')[0]
        : String(rawDate);
    }
    const image = data['image'] ? String(data['image']) : null;
    const language = String(data['language'] ?? 'en');
    const status = String(data['status'] ?? 'approved');
    const visibility = String(data['visibility'] ?? 'public');

    const fullText = `${title}\n\n${content.trim()}`;
    const entities = detectEntities(fullText, entityGraph);
    const result = classifyContent(entities, fullText, taxonomy, rules);

    const summary = extractSummary(content.trim());
    const entitiesFlat = [
      ...entities.organizations, ...entities.programs,
      ...entities.locations, ...entities.credentials,
      ...entities.people, ...entities.activities,
    ];

    const metadata: ContentMetadata = {
      title, slug, language, status, visibility,
      summary,
      body: content.trim(),
      publish_date: date,
      featured_image: image,
      source_file: filePath,
      primary_category: result.category,
      subcategory: result.subcategory,
      tags: result.tags,
      entities: result.entities,
      confidence: result.confidence,
      classification_source: result.classification_source,
      qdrant_payload: {
        content_type: 'news_post',
        primary_category: result.category,
        subcategory: result.subcategory,
        tags: result.tags,
        entities_flat: entitiesFlat,
        language,
        date,
      },
    };

    const outPath = path.join(outputDir, `${slug}-metadata.json`);
    fs.writeFileSync(outPath, JSON.stringify(metadata, null, 2), 'utf8');

    // Accumulate stats
    categoryCounts[result.category] = (categoryCounts[result.category] ?? 0) + 1;
    sourceCounts[result.classification_source] = (sourceCounts[result.classification_source] ?? 0) + 1;
    saved++;

    const detectedCount = (Object.values(entities) as string[][]).filter(a => a.length > 0).length;
    rows.push({
      slug,
      category: result.category,
      subcategory: result.subcategory,
      source: result.classification_source,
      catScore: result.confidence.category_score,
      entities: detectedCount,
      ok: true,
    });

    console.log(
      `${num}  ${padEnd(slug, 55)}  ${padEnd(result.category, 28)}  ${result.classification_source.padEnd(14)}  ${String(result.confidence.category_score).padStart(3)}`,
    );
  } catch (err) {
    failed++;
    const errMsg = err instanceof Error ? err.message : String(err);
    rows.push({ slug: filename, category: '', subcategory: '', source: 'default', catScore: 0, entities: 0, ok: false, error: errMsg });
    console.error(`${num}  ${padEnd(filename, 55)}  ERROR: ${errMsg}`);
  }
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log('\n' + '═'.repeat(115));
console.log(`\nBatch complete: ${saved} saved  |  ${failed} errors\n`);

console.log('Category distribution:');
const sortedCats = Object.entries(categoryCounts).sort((a, b) => b[1] - a[1]);
for (const [cat, count] of sortedCats) {
  const bar = '█'.repeat(Math.round((count / posts.length) * 30));
  console.log(`  ${padEnd(cat, 30)}  ${bar.padEnd(30)}  ${count}`);
}

console.log('\nClassification source:');
for (const [src, count] of Object.entries(sourceCounts).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${src.padEnd(18)}  ${count}`);
}

console.log(`\nOutput: ${outputDir}\n`);

// Write a single consolidated index file
const indexPath = path.join(outputDir, '_index.json');
const index = rows.map(r => ({
  slug: r.slug,
  category: r.category,
  subcategory: r.subcategory,
  classification_source: r.source,
  category_score: r.catScore,
  detected_entity_types: r.entities,
  ok: r.ok,
  ...(r.error ? { error: r.error } : {}),
}));
fs.writeFileSync(indexPath, JSON.stringify(index, null, 2), 'utf8');
console.log(`Index written → output/classification/_index.json\n`);
