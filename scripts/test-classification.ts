#!/usr/bin/env tsx
/**
 * Usage:
 *   tsx scripts/test-classification.ts <path-to-markdown-file>
 *   npm run classify -- data/posts/some-post.md
 */

import * as path from 'path';
import { generateContentMetadata } from '../src/lib/classification/generateContentMetadata';
import { DetectedEntities } from '../src/lib/classification/types';

const filePath = process.argv[2];

if (!filePath) {
  console.error(
    'Usage: tsx scripts/test-classification.ts <path-to-markdown-file>',
  );
  process.exit(1);
}

const absPath = path.resolve(filePath);
console.log(`\nClassifying: ${absPath}\n`);

try {
  const md = generateContentMetadata(absPath);

  const bar = (n: number, max = 100) => {
    const filled = Math.round((n / max) * 20);
    return '[' + '█'.repeat(filled) + '░'.repeat(20 - filled) + ']';
  };

  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║              CLASSIFICATION RESULT                      ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  console.log(`Title:          ${md.title}`);
  console.log(`Slug:           ${md.slug}`);
  console.log(`Date:           ${md.publish_date ?? '(none)'}`);
  console.log(`Image:          ${md.featured_image ?? '(none)'}`);
  console.log('');
  console.log(`Category:       ${md.primary_category}`);
  console.log(`Subcategory:    ${md.subcategory}`);
  console.log(`Source:         ${md.classification_source}`);
  console.log('');
  console.log('Confidence:');
  console.log(`  Category      ${bar(md.confidence.category_score)} ${md.confidence.category_score}`);
  console.log(`  Subcategory   ${bar(md.confidence.subcategory_score)} ${md.confidence.subcategory_score}`);
  console.log(`  Entity cov.   ${bar(md.confidence.entity_extraction_score)} ${md.confidence.entity_extraction_score}`);

  console.log('\nEntities Detected:');
  const { entities } = md;
  const e = entities as DetectedEntities;
  const rows: [string, string[]][] = [
    ['Organizations', e.organizations],
    ['Programs', e.programs],
    ['Credentials', e.credentials],
    ['Locations', e.locations],
    ['Activities', e.activities],
    ['People', e.people],
  ];
  for (const [label, items] of rows) {
    if (items.length > 0) {
      console.log(`  ${label.padEnd(14)} ${items.join(', ')}`);
    }
  }

  console.log(`\nTags (${md.tags.length}/${9}):`);
  md.tags.forEach((t, i) => console.log(`  ${(i + 1).toString().padStart(2)}. ${t}`));

  console.log(`\nSummary:\n  ${md.summary}`);

  console.log('\n── Qdrant Payload ──────────────────────────────────────────');
  console.log(JSON.stringify(md.qdrant_payload, null, 2));

  console.log(
    `\nMetadata saved → output/classification/${md.slug}-metadata.json\n`,
  );
} catch (err) {
  console.error('\nClassification failed:', err);
  process.exit(1);
}
