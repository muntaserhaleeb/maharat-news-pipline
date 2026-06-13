import * as fs from 'fs';
import * as path from 'path';
import matter from 'gray-matter';

import { loadEntityGraph, loadTaxonomy, loadClassificationRules } from './loadTaxonomy';
import { detectEntities } from './detectEntities';
import { classifyContent } from './classifyContent';
import { ContentMetadata, ParsedContent } from './types';

const PROJECT_ROOT = path.resolve(__dirname, '../../../');

// ---------------------------------------------------------------------------
// Content parsing
// ---------------------------------------------------------------------------

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
  // Split on sentence boundaries
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

function parseContent(filePath: string): ParsedContent {
  const raw = fs.readFileSync(filePath, 'utf8');
  const { data, content } = matter(raw);

  const title = String(data['title'] ?? path.basename(filePath, '.md'));
  const slug = String(data['slug'] ?? path.basename(filePath, '.md'));

  // gray-matter/js-yaml may parse ISO date strings as Date objects
  let date: string | null = null;
  const rawDate = data['date'];
  if (rawDate) {
    date = rawDate instanceof Date
      ? rawDate.toISOString().split('T')[0]
      : String(rawDate);
  }

  const image = data['image'] ? String(data['image']) : null;

  return {
    title,
    slug,
    date,
    image,
    body: content.trim(),
    frontmatter: data as Record<string, unknown>,
  };
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export function generateContentMetadata(filePath: string): ContentMetadata {
  const absPath = path.resolve(filePath);

  const entityGraph = loadEntityGraph();
  const taxonomy = loadTaxonomy();
  const rules = loadClassificationRules();

  const parsed = parseContent(absPath);

  // Combine title and body for entity detection; title carries high signal weight
  const fullText = `${parsed.title}\n\n${parsed.body}`;

  const entities = detectEntities(fullText, entityGraph);
  const result = classifyContent(entities, fullText, taxonomy, rules);

  const summary = extractSummary(parsed.body);
  const { frontmatter: fm } = parsed;

  const language = String(fm['language'] ?? 'en');
  const status = String(fm['status'] ?? 'approved');
  const visibility = String(fm['visibility'] ?? 'public');

  const entitiesFlat = [
    ...entities.organizations,
    ...entities.programs,
    ...entities.locations,
    ...entities.credentials,
    ...entities.people,
    ...entities.activities,
  ];

  const metadata: ContentMetadata = {
    title: parsed.title,
    slug: parsed.slug,
    language,
    status,
    visibility,
    summary,
    body: parsed.body,
    publish_date: parsed.date,
    featured_image: parsed.image,
    source_file: absPath,
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
      date: parsed.date,
    },
  };

  const outputDir = path.join(PROJECT_ROOT, 'output', 'classification');
  fs.mkdirSync(outputDir, { recursive: true });
  const outputPath = path.join(outputDir, `${parsed.slug}-metadata.json`);
  fs.writeFileSync(outputPath, JSON.stringify(metadata, null, 2), 'utf8');

  return metadata;
}
