import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

import {
  EntityGraph,
  EntityGraphSchema,
  Taxonomy,
  TaxonomySchema,
  ClassificationRules,
  ClassificationRulesSchema,
} from './types';

const CONFIG_DIR = path.resolve(__dirname, '../../../config');

function readYaml(filename: string): unknown {
  const filepath = path.join(CONFIG_DIR, filename);
  if (!fs.existsSync(filepath)) {
    throw new Error(`Config file not found: ${filepath}`);
  }
  return yaml.load(fs.readFileSync(filepath, 'utf8'));
}

export function loadEntityGraph(): EntityGraph {
  return EntityGraphSchema.parse(readYaml('entities_2_entity_graph.yaml'));
}

export function loadTaxonomy(): Taxonomy {
  return TaxonomySchema.parse(readYaml('entity_driven_taxonomy.yaml'));
}

export function loadClassificationRules(): ClassificationRules {
  return ClassificationRulesSchema.parse(
    readYaml('classification_rules_entity_first.yaml'),
  );
}
