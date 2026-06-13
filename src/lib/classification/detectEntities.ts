import { EntityGraph, EntityEntry, DetectedEntities } from './types';

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Uses lookahead/lookbehind instead of \b so entity names ending in non-word
// characters (e.g. "Co., LTD." or "E&A") match correctly at word boundaries.
function buildEntityRegex(entry: EntityEntry): RegExp {
  const terms = [entry.canonical, ...(entry.aliases ?? [])];
  // Sort longest first so the most specific alias wins in alternation
  terms.sort((a, b) => b.length - a.length);
  const pattern = terms.map(escapeRegex).join('|');
  return new RegExp(`(?<!\\w)(?:${pattern})(?!\\w)`, 'i');
}

function detectFromEntries(text: string, entries: EntityEntry[]): string[] {
  const found = new Set<string>();
  for (const entry of entries) {
    if (buildEntityRegex(entry).test(text)) {
      found.add(entry.canonical);
    }
  }
  return [...found];
}

export function detectEntities(text: string, graph: EntityGraph): DetectedEntities {
  const { entities } = graph;
  return {
    organizations: detectFromEntries(text, entities.organizations),
    programs: detectFromEntries(text, entities.programs),
    locations: detectFromEntries(text, entities.locations),
    credentials: detectFromEntries(text, entities.credentials),
    people: detectFromEntries(text, entities.people),
    activities: detectFromEntries(text, entities.activities),
  };
}
