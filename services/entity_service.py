"""
Entity service — deterministic dictionary-based entity extraction.
Uses compiled regex patterns built from config/entities.yaml.
Returns canonical, deduplicated entity names per type.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT       = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

_ENTITY_TYPES = (
    "organizations",
    "programs",
    "locations",
    "credentials",
    "people",
)


def _compile(term: str) -> re.Pattern:
    """
    Build a match pattern for one alias or canonical name.
    Pure-word terms (no special chars) use word boundaries.
    Terms with punctuation / whitespace use case-insensitive substring match.
    """
    escaped = re.escape(term)
    if re.search(r"\W", term):
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


class EntityService:
    """
    Pre-compiles all entity patterns at init time.
    Thread-safe after construction (read-only at extraction time).
    """

    def __init__(self, entities_cfg: dict):
        # _patterns: {entity_type: [(compiled_pattern, canonical_name), ...]}
        self._patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        # _metadata: {entity_type: {canonical_name: {field: value, ...}}}
        self._metadata: Dict[str, Dict[str, dict]] = {}

        raw = entities_cfg.get("entities", {})
        for etype in _ENTITY_TYPES:
            patterns: List[Tuple[re.Pattern, str]] = []
            meta: Dict[str, dict] = {}
            for entry in raw.get(etype, []):
                canonical = entry["canonical"]
                terms = [canonical] + list(entry.get("aliases", []))
                # Longest terms first — avoids shadowing by short aliases
                terms.sort(key=len, reverse=True)
                for term in terms:
                    if term.strip():
                        patterns.append((_compile(term), canonical))
                # Store extra metadata fields (title, affiliation, etc.)
                extra = {k: v for k, v in entry.items()
                         if k not in ("canonical", "aliases") and v}
                if extra:
                    meta[canonical] = extra
            self._patterns[etype] = patterns
            self._metadata[etype] = meta

    @classmethod
    def from_config(cls) -> "EntityService":
        path = CONFIG_DIR / "entities.yaml"
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(cfg)

    # ── core extraction ────────────────────────────────────────────────────

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from any text string.
        Returns {entity_type: [canonical_name, ...]} — sorted, deduplicated.
        """
        result: Dict[str, List[str]] = {}
        for etype, patterns in self._patterns.items():
            found: set = set()
            for pattern, canonical in patterns:
                if pattern.search(text):
                    found.add(canonical)
            result[etype] = sorted(found)
        return result

    def extract_from_article(self, front: dict, body: str) -> Dict[str, List[str]]:
        """
        Extract from the combined article text: title + summary + body.
        Call this once per article; attach results to all its chunks.
        """
        parts = [
            front.get("title", "") or "",
            front.get("summary", "") or "",
            body or "",
        ]
        full_text = "\n".join(p for p in parts if p.strip())
        return self.extract(full_text)

    def get_metadata(self, canonical: str, etype: str = "people") -> dict:
        """
        Return extra metadata (title, affiliation, …) for a canonical name.
        Returns an empty dict if no metadata is stored.
        """
        return self._metadata.get(etype, {}).get(canonical, {})

    def get_people_metadata(self, canonicals: List[str]) -> List[dict]:
        """
        Return a list of metadata dicts for a list of canonical people names.
        Each dict always includes the 'canonical' key.
        """
        result = []
        for name in canonicals:
            meta = {"canonical": name}
            meta.update(self.get_metadata(name, "people"))
            result.append(meta)
        return result

    # ── diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return number of patterns compiled per entity type."""
        return {etype: len(pats) for etype, pats in self._patterns.items()}
