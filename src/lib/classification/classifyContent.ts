import {
  DetectedEntities,
  TaxonomyCategory,
  Taxonomy,
  ClassificationRules,
  ClassificationRule,
  ClassificationResult,
  ClassificationSource,
} from './types';

type ScoreMap = Record<string, number>;

// ---------------------------------------------------------------------------
// Scoring helpers
// ---------------------------------------------------------------------------

// Resolves whether a compound pattern element matches.
// Elements are either a specific canonical entity name or an entity type keyword
// ("organization", "program", "location", "credential", "person", "activity").
function matchesEntityPattern(
  pattern: string[],
  entities: DetectedEntities,
): boolean {
  const TYPE_MAP: Record<string, keyof DetectedEntities> = {
    organization: 'organizations',
    organizations: 'organizations',
    program: 'programs',
    programs: 'programs',
    location: 'locations',
    locations: 'locations',
    credential: 'credentials',
    credentials: 'credentials',
    person: 'people',
    people: 'people',
    activity: 'activities',
    activities: 'activities',
  };

  for (const element of pattern) {
    const key = TYPE_MAP[element.toLowerCase()];
    if (key !== undefined) {
      if (entities[key].length === 0) return false;
    } else {
      const all = (Object.values(entities) as string[][]).flat();
      if (!all.includes(element)) return false;
    }
  }
  return true;
}

function scoreCategory(
  cat: TaxonomyCategory,
  entities: DetectedEntities,
  scoring: ClassificationRule['scoring'],
): number {
  let score = 0;
  const s = cat.signals;

  if (s.activities) {
    for (const act of s.activities) {
      if (entities.activities.includes(act)) score += scoring.activity_match;
    }
  }

  if (s.entity_types) {
    for (const type of s.entity_types) {
      switch (type) {
        case 'programs':
          score += entities.programs.length * scoring.program_match;
          break;
        case 'credentials':
          score += entities.credentials.length * scoring.credential_match;
          break;
        case 'locations':
          score += entities.locations.length * scoring.location_match;
          break;
        case 'organizations':
          score += entities.organizations.length * scoring.organization_match;
          break;
        case 'people':
          // Treat people with same weight as organizations for category scoring
          score += entities.people.length * scoring.organization_match;
          break;
      }
    }
  }

  if (s.programs) {
    for (const prog of s.programs) {
      if (entities.programs.includes(prog)) score += scoring.program_match;
    }
  }

  if (s.organizations) {
    for (const org of s.organizations) {
      if (entities.organizations.includes(org)) score += scoring.organization_match;
    }
  }

  if (s.credentials) {
    for (const cred of s.credentials) {
      if (entities.credentials.includes(cred)) score += scoring.credential_match;
    }
  }

  if (s.entity_patterns) {
    for (const pattern of s.entity_patterns) {
      if (matchesEntityPattern(pattern, entities)) {
        score += scoring.entity_pattern_match;
      }
    }
  }

  return score;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function scoreKeywordFallback(
  text: string,
  rules: ClassificationRule,
): ScoreMap {
  const scores: ScoreMap = {};
  for (const rule of rules.keyword_fallback_rules) {
    let s = 0;
    for (const kw of rule.keywords) {
      // Word-boundary matching prevents partial hits like "press" ⊂ "pressure"
      const regex = new RegExp(`(?<!\\w)${escapeRegex(kw)}(?!\\w)`, 'i');
      if (regex.test(text)) s += rules.scoring.keyword_fallback_match;
    }
    if (s > 0) scores[rule.category] = s;
  }
  return scores;
}

// ---------------------------------------------------------------------------
// Subcategory resolution
// ---------------------------------------------------------------------------

function resolveSubcategory(
  category: string,
  entities: DetectedEntities,
  taxonomy: Taxonomy,
): { subcategory: string; matched: boolean } {
  for (const rule of taxonomy.subcategory_resolution.rules) {
    if (rule.when.category !== category) continue;

    let ok = true;

    if (rule.when.has_any_organization) {
      if (!rule.when.has_any_organization.some(o => entities.organizations.includes(o)))
        ok = false;
    }
    if (ok && rule.when.has_any_location) {
      if (!rule.when.has_any_location.some(l => entities.locations.includes(l)))
        ok = false;
    }
    if (ok && rule.when.has_any_program) {
      if (!rule.when.has_any_program.some(p => entities.programs.includes(p)))
        ok = false;
    }
    if (ok && rule.when.has_any_credential) {
      if (!rule.when.has_any_credential.some(c => entities.credentials.includes(c)))
        ok = false;
    }
    if (ok && rule.when.has_activity) {
      if (!entities.activities.includes(rule.when.has_activity)) ok = false;
    }

    if (ok) return { subcategory: rule.subcategory, matched: true };
  }

  return {
    subcategory: taxonomy.subcategory_resolution.fallback_subcategory,
    matched: false,
  };
}

// ---------------------------------------------------------------------------
// Tag generation
// ---------------------------------------------------------------------------

const INTL_ORGS = new Set([
  'Sinopec Nanjing Engineering Middle East Co., LTD.',
  'Nanjing University of Posts and Telecommunications',
  'TÜV SÜD',
  'Vermaat Technics B.V.',
  'Hyundai Engineering & Construction',
  'ESAB Middle East',
]);

const INTL_LOCATIONS = new Set(['China', 'Nanjing', 'Sharjah']);

const TECH_PROGRAMS = new Set([
  'Welding Associate Diploma',
  'Automated Welding Diploma',
  'Pipefitting Diploma',
  'Electrical Diploma',
  'Instrumentation Diploma',
  'Scaffolding Diploma',
]);

function selectThematicTags(
  category: string,
  entities: DetectedEntities,
  text: string,
  allThematic: string[],
): string[] {
  const lower = text.toLowerCase();
  const selected: string[] = [];

  const add = (tag: string) => {
    if (allThematic.includes(tag) && !selected.includes(tag)) selected.push(tag);
  };

  if (category === 'Industry Engagement' || category === 'Training Programs') {
    add('Workforce Development');
    add('Skills Development');
  }
  if (entities.activities.includes('On-the-Job Training') || category === 'Industry Engagement') {
    add('Employer Engagement');
    add('Practical Training');
  }
  if (entities.activities.includes('Graduation') || category === 'Events & Ceremonies') {
    add('Career Readiness');
  }
  if (lower.includes('female') || lower.includes('women') || lower.includes(' girl')) {
    add('Women in Construction');
  }
  if (entities.programs.some(p => TECH_PROGRAMS.has(p))) {
    add('Construction Technology');
  }
  if (category === 'Digital Transformation') {
    add('Digital Transformation');
    add('Innovation');
  }
  if (lower.includes('vision 2030') || lower.includes('vision2030')) {
    add('Saudi Vision 2030');
  }
  if (category === 'Safety & HSE') {
    add('Site Safety');
  }
  if (category === 'Accreditation & Compliance') {
    add('Quality Assurance');
  }
  if (
    entities.organizations.some(o => INTL_ORGS.has(o)) ||
    entities.locations.some(l => INTL_LOCATIONS.has(l))
  ) {
    add('International Collaboration');
  }
  if (category === 'Community Outreach') {
    add('Community Impact');
  }

  return selected;
}

function generateTags(
  entities: DetectedEntities,
  taxonomy: Taxonomy,
  rules: ClassificationRules,
  category: string,
  text: string,
): string[] {
  const behavior = taxonomy.entity_type_to_tag_behavior;
  const tagRules = rules.classification_rules.tag_rules;
  const disallowed = new Set(tagRules.disallowed_free_tags.map(t => t.toLowerCase()));

  const tags: string[] = [];

  const addFromType = (
    items: string[],
    behaviorKey: string,
  ) => {
    const b = behavior[behaviorKey];
    const limit = b?.max_per_article ?? items.length;
    for (const item of items.slice(0, limit)) {
      if (!disallowed.has(item.toLowerCase()) && !tags.includes(item)) {
        tags.push(item);
      }
    }
  };

  addFromType(entities.organizations, 'organizations');
  addFromType(entities.programs, 'programs');
  addFromType(entities.credentials, 'credentials');
  addFromType(entities.locations, 'locations');
  addFromType(entities.activities, 'activities');
  addFromType(entities.people, 'people');

  for (const tag of selectThematicTags(category, entities, text, tagRules.thematic_tags)) {
    if (!tags.includes(tag)) tags.push(tag);
  }

  const deduped = tagRules.deduplicate ? [...new Set(tags)] : tags;
  return deduped.slice(0, tagRules.max_tags);
}

// ---------------------------------------------------------------------------
// Semantic tie-breakers
// Implements the YAML tie_breakers rules by boosting activity-driven categories
// when their defining activity is present, preventing entity-type inflation from
// overriding dominant event/agreement/accreditation signals.
// ---------------------------------------------------------------------------

function applyTieBreakers(
  scores: ScoreMap,
  entities: DetectedEntities,
  minThreshold: number,
): ScoreMap {
  const adj = { ...scores };

  const bump = (label: string) => {
    const current = adj[label] ?? 0;
    const max = Math.max(...Object.values(adj), 0);
    // Ensure this category clears threshold and beats the current leader
    adj[label] = Math.max(current, Math.max(max, minThreshold) + 1);
  };

  // Rule: prefer Events & Ceremonies when Graduation is present (and not
  // primarily a Media Coverage story — Media Coverage would already outscore it)
  if (
    entities.activities.includes('Graduation') &&
    !entities.activities.includes('Media Coverage')
  ) {
    bump('Events & Ceremonies');
  }

  // Rule: prefer Partnerships & Agreements when 2+ orgs co-occur with
  // Agreement or MoU activity
  if (
    entities.organizations.length >= 2 &&
    (entities.activities.includes('Agreement') ||
      entities.activities.includes('Memorandum of Understanding'))
  ) {
    bump('Partnerships & Agreements');
  }

  // Rule: prefer Accreditation & Compliance when a credential entity and an
  // accreditation-type activity co-occur
  if (
    entities.credentials.length > 0 &&
    (entities.activities.includes('Accreditation') ||
      entities.activities.includes('Certification') ||
      entities.activities.includes('Endorsement'))
  ) {
    bump('Accreditation & Compliance');
  }

  return adj;
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export function classifyContent(
  entities: DetectedEntities,
  text: string,
  taxonomy: Taxonomy,
  rules: ClassificationRules,
): ClassificationResult {
  const r = rules.classification_rules;

  // Score all categories via entity signals
  const rawScores: ScoreMap = {};
  for (const cat of taxonomy.primary_categories) {
    rawScores[cat.label] = scoreCategory(cat, entities, r.scoring);
  }

  // Apply semantic tie-breaker boosts before category selection
  const entityScores = applyTieBreakers(
    rawScores,
    entities,
    r.scoring.minimum_confidence_for_auto_category,
  );

  // Score keyword fallback per category
  const kwScores = scoreKeywordFallback(text, r);

  // Pick best category: highest entity score (priority_order as tiebreaker).
  // We track the best-above-threshold AND the overall best-any-score separately
  // so we can prefer entity signals over keyword fallback per the entity-first
  // classification strategy.
  let bestCategory = r.default_category;
  let bestEntityScore = 0;       // best score that clears the threshold
  let bestAnyCategory = r.default_category;
  let bestAnyScore = 0;          // best score regardless of threshold

  for (const label of r.priority_order) {
    const s = entityScores[label] ?? 0;
    if (s > bestAnyScore) {
      bestAnyScore = s;
      bestAnyCategory = label;
    }
    if (s > bestEntityScore && s >= r.scoring.minimum_confidence_for_auto_category) {
      bestEntityScore = s;
      bestCategory = label;
    }
  }

  // Keyword fallback applies only when there are ZERO entity signals.
  // When entity signals are present but below threshold, prefer the entity-
  // driven category (lower confidence) over keyword fallback — this avoids
  // generic keywords like "awareness" overriding a detected activity entity.
  let usedFallback = false;
  if (bestAnyScore === 0) {
    for (const label of r.priority_order) {
      if ((kwScores[label] ?? 0) > 0) {
        bestCategory = label;
        usedFallback = true;
        break;
      }
    }
  } else if (bestEntityScore === 0) {
    // Entity signals exist but none cleared the threshold — use best entity match
    bestCategory = bestAnyCategory;
  }

  // Determine classification source
  const finalEntityScore = entityScores[bestCategory] ?? 0;
  const finalKwScore = kwScores[bestCategory] ?? 0;
  let source: ClassificationSource;

  if (finalEntityScore >= r.scoring.minimum_confidence_for_auto_category) {
    source = finalKwScore > 0 ? 'hybrid' : 'entity_match';
  } else if (usedFallback) {
    source = 'keyword_fallback';
  } else if (finalEntityScore > 0) {
    source = 'entity_match';
  } else {
    source = 'default';
  }

  // Subcategory
  const { subcategory, matched } = resolveSubcategory(bestCategory, entities, taxonomy);

  // Confidence scores (capped at 100)
  const categoryScore = Math.min(100, finalEntityScore + (usedFallback ? finalKwScore : 0));
  const subcategoryScore = matched
    ? Math.min(100, Math.round(categoryScore * 0.85))
    : Math.min(40, categoryScore);
  const detectedTypes = (Object.values(entities) as string[][]).filter(a => a.length > 0).length;
  const entityExtractionScore = Math.round((detectedTypes / 6) * 100);

  return {
    category: bestCategory,
    subcategory,
    tags: generateTags(entities, taxonomy, rules, bestCategory, text),
    entities,
    confidence: {
      category_score: categoryScore,
      subcategory_score: subcategoryScore,
      entity_extraction_score: entityExtractionScore,
    },
    classification_source: source,
  };
}
