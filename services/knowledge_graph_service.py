"""
Knowledge Graph Service — lightweight YAML/JSON-backed entity graph.

Loads data/graph/entities.yaml + data/graph/relationships.yaml at startup,
builds an in-memory indexed graph, and answers entity context queries:

    svc = KnowledgeGraphService.from_config()
    svc.get_entity_profile("Sinopec")
    svc.get_related_entities("Maharat")
    svc.get_relationships("Maharat", "Sinopec")
    svc.build_context_block("Sinopec")         # prompt-ready string
    svc.export_json("data/graph/knowledge_graph.json")
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT       = Path(__file__).resolve().parent.parent
GRAPH_DIR  = ROOT / "data" / "graph"


# ── loading helpers ────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── graph builder ──────────────────────────────────────────────────────────

def _build_graph(entities: List[dict], relationships: List[dict]) -> dict:
    """
    Build three indexes from the raw YAML data:
      - entity_index:   {entity_id: entity_dict}
      - adjacency:      {entity_id: [edge_dict, ...]}
      - name_index:     {lowercase_alias: entity_id}
    """
    entity_index: Dict[str, dict] = {}
    adjacency:    Dict[str, list] = {}
    name_index:   Dict[str, str]  = {}

    # Index entities and build name_index
    for ent in entities:
        eid  = ent["id"]
        entity_index[eid] = ent
        adjacency[eid]    = []

        # Register canonical name
        for raw in [ent.get("name"), ent.get("short_name")] + (ent.get("aliases") or []):
            if raw:
                name_index[raw.lower()] = eid

    # Index relationships into adjacency lists
    for rel in relationships:
        src = rel["source"]
        tgt = rel["target"]

        # Ensure entries exist for nodes that might be missing from entities.yaml
        for node in (src, tgt):
            if node not in adjacency:
                adjacency[node] = []
            if node not in entity_index:
                entity_index[node] = {"id": node, "name": node, "type": "unknown"}

        # Outgoing edge on source
        adjacency[src].append({
            "rel_id":    rel.get("id", ""),
            "direction": "outgoing",
            "type":      rel["type"],
            "target":    tgt,
            "label":     rel.get("label", ""),
            "attributes": rel.get("attributes", {}),
        })

        # Incoming edge on target
        adjacency[tgt].append({
            "rel_id":    rel.get("id", ""),
            "direction": "incoming",
            "type":      rel["type"],
            "source":    src,
            "label":     rel.get("label", ""),
            "attributes": rel.get("attributes", {}),
        })

    return {
        "entity_index": entity_index,
        "adjacency":    adjacency,
        "name_index":   name_index,
    }


# ── main service class ─────────────────────────────────────────────────────

class KnowledgeGraphService:
    """
    Lightweight knowledge graph over Maharat entities and relationships.
    Source data: data/graph/entities.yaml + data/graph/relationships.yaml
    """

    def __init__(
        self,
        entities_path: Path,
        relationships_path: Path,
    ):
        raw_entities      = _load_yaml(entities_path).get("entities", [])
        raw_relationships = _load_yaml(relationships_path).get("relationships", [])

        self._raw_entities      = raw_entities
        self._raw_relationships = raw_relationships
        self._graph             = _build_graph(raw_entities, raw_relationships)

    @classmethod
    def from_config(
        cls,
        graph_dir: Optional[Path] = None,
    ) -> "KnowledgeGraphService":
        """Load from data/graph/ using project-relative paths."""
        base = Path(graph_dir) if graph_dir else GRAPH_DIR
        return cls(
            entities_path=base / "entities.yaml",
            relationships_path=base / "relationships.yaml",
        )

    # ── entity lookup ──────────────────────────────────────────────────────

    def find_entity(self, name: str) -> Optional[dict]:
        """
        Case-insensitive lookup by name, short_name, or any alias.
        Returns the entity dict or None.
        """
        key = name.strip().lower()
        entity_id = self._graph["name_index"].get(key)

        if entity_id is None:
            # Partial match fallback: find the entity whose name contains the query
            for alias, eid in self._graph["name_index"].items():
                if key in alias or alias in key:
                    entity_id = eid
                    break

        if entity_id is None:
            return None
        return self._graph["entity_index"].get(entity_id)

    # ── core query methods ─────────────────────────────────────────────────

    def get_entity_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Return full entity data plus all its direct relationships.

        Returns:
            {
              "entity": {...},
              "relationships": [
                {"direction": "outgoing", "type": "...", "target_id": "...",
                 "target_name": "...", "label": "...", "attributes": {...}},
                ...
              ]
            }
        """
        entity = self.find_entity(name)
        if entity is None:
            return None

        eid   = entity["id"]
        edges = self._graph["adjacency"].get(eid, [])

        enriched_edges = []
        for edge in edges:
            enriched = dict(edge)
            other_id = edge.get("target") or edge.get("source")
            other    = self._graph["entity_index"].get(other_id, {})
            enriched["other_id"]   = other_id
            enriched["other_name"] = other.get("name", other_id)
            enriched["other_type"] = other.get("type", "unknown")
            enriched_edges.append(enriched)

        return {"entity": entity, "relationships": enriched_edges}

    def get_related_entities(
        self,
        name: str,
        rel_type: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all entities directly connected to the given entity.

        rel_type  — filter by relationship type (e.g. "partnered_with")
        direction — "outgoing" | "incoming" | None (both)

        Returns list of:
            {"entity": {...}, "via_rel_type": "...", "label": "...", "direction": "..."}
        """
        entity = self.find_entity(name)
        if entity is None:
            return []

        eid    = entity["id"]
        edges  = self._graph["adjacency"].get(eid, [])
        result = []

        for edge in edges:
            if rel_type and edge["type"] != rel_type:
                continue
            if direction and edge["direction"] != direction:
                continue

            other_id   = edge.get("target") or edge.get("source")
            other_ent  = self._graph["entity_index"].get(other_id, {})
            result.append({
                "entity":       other_ent,
                "via_rel_type": edge["type"],
                "direction":    edge["direction"],
                "label":        edge.get("label", ""),
                "attributes":   edge.get("attributes", {}),
            })

        return result

    def get_relationships(
        self,
        entity1: str,
        entity2: str,
    ) -> List[Dict[str, Any]]:
        """
        Return all relationships between two entities (either direction).

        Returns list of:
            {"rel_id": "...", "direction": "...", "type": "...", "label": "...", "attributes": {...}}
        """
        ent1 = self.find_entity(entity1)
        ent2 = self.find_entity(entity2)

        if ent1 is None or ent2 is None:
            return []

        id1    = ent1["id"]
        id2    = ent2["id"]
        edges  = self._graph["adjacency"].get(id1, [])
        result = []

        for edge in edges:
            other_id = edge.get("target") or edge.get("source")
            if other_id == id2:
                result.append({
                    "rel_id":     edge.get("rel_id", ""),
                    "direction":  edge["direction"],
                    "type":       edge["type"],
                    "label":      edge.get("label", ""),
                    "attributes": edge.get("attributes", {}),
                })

        return result

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
    ) -> List[dict]:
        """
        Search entities by partial name match; optionally filter by type.
        Returns list of matching entity dicts.
        """
        q       = query.lower()
        results = []

        for eid, ent in self._graph["entity_index"].items():
            if entity_type and ent.get("type") != entity_type:
                continue
            # Match against name, short_name, description, aliases
            searchable = " ".join(filter(None, [
                ent.get("name", ""),
                ent.get("short_name", ""),
                ent.get("description", ""),
                " ".join(ent.get("aliases") or []),
            ])).lower()
            if q in searchable:
                results.append(ent)

        return results

    def get_entities_by_type(self, entity_type: str) -> List[dict]:
        """Return all entities of a given type."""
        return [
            ent for ent in self._graph["entity_index"].values()
            if ent.get("type") == entity_type
        ]

    # ── prompt context builder ─────────────────────────────────────────────

    def build_context_block(self, name: str, max_rels: int = 12) -> str:
        """
        Build a compact, prompt-ready context string for a named entity.
        Suitable for inclusion in RAG prompts to ground generation.
        """
        profile = self.get_entity_profile(name)
        if profile is None:
            return f"[Knowledge Graph: no entity found for '{name}']"

        ent   = profile["entity"]
        rels  = profile["relationships"][:max_rels]

        lines = [
            f"Knowledge Graph — {ent.get('name', name)} ({ent.get('type', 'unknown')})",
        ]
        if ent.get("description"):
            lines.append(f"  Description: {ent['description'].strip()}")

        attrs = ent.get("attributes", {})
        if attrs:
            for k, v in attrs.items():
                if v not in (None, "", [], {}):
                    lines.append(f"  {k}: {v}")

        if rels:
            lines.append("  Relationships:")
            for r in rels:
                other  = r.get("other_name", r.get("other_id", "?"))
                r_type = r.get("type", r.get("via_rel_type", "related_to")).replace("_", " ")
                direct = r["direction"]
                if direct == "outgoing":
                    lines.append(f"    -> {r_type} -> {other}")
                else:
                    lines.append(f"    <- {r_type} <- {other}")

        return "\n".join(lines)

    # ── graph stats ────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return basic graph statistics."""
        by_type: Dict[str, int] = {}
        for ent in self._graph["entity_index"].values():
            t = ent.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "entity_count":       len(self._graph["entity_index"]),
            "relationship_count": len(self._raw_relationships),
            "name_aliases":       len(self._graph["name_index"]),
            "by_type":            by_type,
        }

    # ── JSON export ────────────────────────────────────────────────────────

    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        Compile and save knowledge_graph.json.
        Returns the output path as a string.
        """
        out = Path(output_path) if output_path else GRAPH_DIR / "knowledge_graph.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        graph_doc = {
            "meta": {
                "version":            "1.0",
                "generated_at":       _today_iso(),
                "entity_count":       len(self._raw_entities),
                "relationship_count": len(self._raw_relationships),
                "source_entities":    "data/graph/entities.yaml",
                "source_relations":   "data/graph/relationships.yaml",
            },
            "entities":   {
                eid: ent
                for eid, ent in self._graph["entity_index"].items()
            },
            "relationships": self._raw_relationships,
            "indexes": {
                "by_name": self._graph["name_index"],
                "by_type": {
                    t: [e["id"] for e in self._graph["entity_index"].values()
                        if e.get("type") == t]
                    for t in {
                        e.get("type", "unknown")
                        for e in self._graph["entity_index"].values()
                    }
                },
                "adjacency": self._graph["adjacency"],
            },
        }

        with open(out, "w", encoding="utf-8") as fh:
            json.dump(graph_doc, fh, indent=2, ensure_ascii=False, default=str)

        return str(out)


def _today_iso() -> str:
    from datetime import date
    return date.today().isoformat()
