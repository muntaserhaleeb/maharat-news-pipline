import { z } from 'zod';

// ---------------------------------------------------------------------------
// Entity Graph (entities_2_entity_graph.yaml)
// ---------------------------------------------------------------------------

export const EntityEntrySchema = z.object({
  canonical: z.string(),
  aliases: z.array(z.string()).default([]),
  title: z.string().optional(),
  affiliation: z.string().optional(),
});

export const EntityGraphSchema = z.object({
  entities: z.object({
    organizations: z.array(EntityEntrySchema),
    programs: z.array(EntityEntrySchema),
    locations: z.array(EntityEntrySchema),
    credentials: z.array(EntityEntrySchema),
    people: z.array(EntityEntrySchema),
    activities: z.array(EntityEntrySchema),
  }),
});

export type EntityEntry = z.infer<typeof EntityEntrySchema>;
export type EntityGraph = z.infer<typeof EntityGraphSchema>;

// ---------------------------------------------------------------------------
// Taxonomy (entity_driven_taxonomy.yaml)
// ---------------------------------------------------------------------------

const TaxonomySignalsSchema = z.object({
  activities: z.array(z.string()).optional(),
  // type names like 'programs', 'credentials', 'locations', 'people', 'organizations'
  entity_types: z.array(z.string()).optional(),
  // each inner array is a compound pattern, e.g. ["MCTC", "organization", "Agreement"]
  entity_patterns: z.array(z.array(z.string())).optional(),
  // specific org/program/credential names as category signals
  organizations: z.array(z.string()).optional(),
  programs: z.array(z.string()).optional(),
  credentials: z.array(z.string()).optional(),
  keywords_fallback: z.array(z.string()).optional(),
});

export const TaxonomyCategorySchema = z.object({
  id: z.string(),
  label: z.string(),
  description: z.string(),
  subcategories: z.array(z.string()),
  signals: TaxonomySignalsSchema,
});

const SubcategoryWhenSchema = z.object({
  category: z.string(),
  has_any_organization: z.array(z.string()).optional(),
  has_any_location: z.array(z.string()).optional(),
  has_any_program: z.array(z.string()).optional(),
  has_any_credential: z.array(z.string()).optional(),
  has_activity: z.string().optional(),
});

const TagBehaviorSchema = z.object({
  tag_group: z.string(),
  max_per_article: z.number(),
  use_canonical_name: z.boolean(),
});

export const TaxonomySchema = z.object({
  taxonomy: z.object({
    version: z.string(),
    locale: z.string(),
    source: z.string().optional(),
    classification_strategy: z.string().optional(),
  }),
  primary_categories: z.array(TaxonomyCategorySchema),
  entity_type_to_tag_behavior: z.record(TagBehaviorSchema),
  subcategory_resolution: z.object({
    rules: z.array(
      z.object({
        when: SubcategoryWhenSchema,
        subcategory: z.string(),
      }),
    ),
    fallback_subcategory: z.string(),
  }),
});

export type TaxonomyCategory = z.infer<typeof TaxonomyCategorySchema>;
export type Taxonomy = z.infer<typeof TaxonomySchema>;

// ---------------------------------------------------------------------------
// Classification Rules (classification_rules_entity_first.yaml)
// ---------------------------------------------------------------------------

const ScoringSchema = z.object({
  activity_match: z.number(),
  entity_pattern_match: z.number(),
  credential_match: z.number(),
  program_match: z.number(),
  organization_match: z.number(),
  location_match: z.number(),
  keyword_fallback_match: z.number(),
  minimum_confidence_for_auto_category: z.number(),
  minimum_confidence_for_auto_subcategory: z.number(),
});

export const ClassificationRulesSchema = z.object({
  classification_rules: z.object({
    version: z.string(),
    default_category: z.string(),
    single_primary_category: z.boolean(),
    priority_order: z.array(z.string()),
    scoring: ScoringSchema,
    tie_breakers: z.array(z.string()),
    keyword_fallback_rules: z.array(
      z.object({ category: z.string(), keywords: z.array(z.string()) }),
    ),
    tag_rules: z.object({
      min_tags: z.number(),
      max_tags: z.number(),
      deduplicate: z.boolean(),
      normalize_to_canonical_entities: z.boolean(),
      disallowed_free_tags: z.array(z.string()),
      thematic_tags: z.array(z.string()),
    }),
  }),
});

export type ClassificationRules = z.infer<typeof ClassificationRulesSchema>;
export type ClassificationRule = ClassificationRules['classification_rules'];

// ---------------------------------------------------------------------------
// Runtime types (not Zod-validated — produced internally)
// ---------------------------------------------------------------------------

export interface DetectedEntities {
  organizations: string[];
  programs: string[];
  locations: string[];
  credentials: string[];
  people: string[];
  activities: string[];
}

export interface ParsedContent {
  title: string;
  slug: string;
  date: string | null;
  image: string | null;
  body: string;
  frontmatter: Record<string, unknown>;
}

export type ClassificationSource =
  | 'entity_match'
  | 'keyword_fallback'
  | 'hybrid'
  | 'default';

export interface ClassificationResult {
  category: string;
  subcategory: string;
  tags: string[];
  entities: DetectedEntities;
  confidence: {
    category_score: number;
    subcategory_score: number;
    entity_extraction_score: number;
  };
  classification_source: ClassificationSource;
}

export interface ContentMetadata {
  title: string;
  slug: string;
  language: string;
  status: string;
  visibility: string;
  summary: string;
  body: string;
  publish_date: string | null;
  featured_image: string | null;
  source_file: string;
  primary_category: string;
  subcategory: string;
  tags: string[];
  entities: DetectedEntities;
  confidence: {
    category_score: number;
    subcategory_score: number;
    entity_extraction_score: number;
  };
  classification_source: ClassificationSource;
  qdrant_payload: {
    content_type: string;
    primary_category: string;
    subcategory: string;
    tags: string[];
    entities_flat: string[];
    language: string;
    date: string | null;
  };
}
