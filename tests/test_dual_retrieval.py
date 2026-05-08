#!/usr/bin/env python3
"""
Dual-retrieval evaluation.
Tests MemoryRouter routing decisions and combined news + knowledge retrieval.

Usage:
    python tests/test_dual_retrieval.py
    python tests/test_dual_retrieval.py --verbose
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.config_service import load_qdrant_config
from services.memory_router import MemoryRouter
from services.retrieval_service import RetrievalService


# ── routing test cases ────────────────────────────────────────────────────

ROUTING_CASES = [
    # (query, expected_route, optional intent override)
    ("What is Maharat?",                                    "knowledge", None),
    ("Maharat mission and vision",                          "knowledge", None),
    ("Maharat accreditation certifications",                "knowledge", None),
    ("What short courses does Maharat offer?",              "knowledge", None),
    ("Maharat campus and facilities",                       "knowledge", None),
    ("fire safety training drill",                          "news",      None),
    ("graduation ceremony female trainees",                 "news",      None),
    ("Samsung partnership agreement",                       "news",      None),
    ("Write an article about Maharat and Sinopec",          "both",      None),
    ("Draft LinkedIn post about female graduation",         "both",      None),
    ("Generate article about Maharat training methodology", "both",      None),
    ("What is Maharat? Draft article about accreditation",  "both",      None),
    # explicit overrides
    ("graduation ceremony",                                 "knowledge", "knowledge"),
    ("Maharat overview",                                    "news",      "news"),
]

# ── dual retrieval test cases ─────────────────────────────────────────────

DUAL_RETRIEVAL_CASES = [
    {
        "query":              "Write article about Maharat collaboration with Sinopec",
        "expect_news_min":    1,
        "expect_know_min":    1,
    },
    {
        "query":              "Draft LinkedIn post about female graduation",
        "expect_news_min":    1,
        "expect_know_min":    0,
    },
    {
        "query":              "Maharat accreditation and strategic partnerships",
        "expect_news_min":    0,
        "expect_know_min":    1,
    },
]


def _ok(label: str, detail: str = "") -> None:
    msg = f"  [OK]   {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


def _fail(label: str, detail: str = "") -> None:
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


def run_routing_eval(
    cases: List[tuple],
    router: MemoryRouter,
    verbose: bool = False,
) -> Tuple[int, int]:
    passed = failed = 0

    for query, expected_route, intent in cases:
        result = router.route_query(query, intent=intent)
        if result.route == expected_route:
            if verbose:
                _ok(f"'{query}'", f"route={result.route}  {result.reasoning}")
            passed += 1
        else:
            _fail(
                f"'{query}'",
                f"expected route={expected_route}, got route={result.route}  "
                f"({result.reasoning})",
            )
            failed += 1

    return passed, failed


def run_dual_retrieval_eval(
    cases: List[dict],
    news_service: RetrievalService,
    knowledge_service: RetrievalService,
    verbose: bool = False,
) -> Tuple[int, int]:
    passed = failed = 0

    for case in cases:
        query = case["query"]
        news_results  = news_service.search(query_text=query, limit=5)
        know_results  = knowledge_service.search(query_text=query, limit=5)

        ok = True
        detail_parts = []

        if len(news_results) < case["expect_news_min"]:
            _fail(
                f"'{query}' [news]",
                f"got {len(news_results)}, expected >= {case['expect_news_min']}",
            )
            ok = False
        else:
            detail_parts.append(f"news={len(news_results)}")

        if len(know_results) < case["expect_know_min"]:
            _fail(
                f"'{query}' [knowledge]",
                f"got {len(know_results)}, expected >= {case['expect_know_min']}",
            )
            ok = False
        else:
            detail_parts.append(f"knowledge={len(know_results)}")

        # Check for duplicate chunks across collections (same chunk_id)
        news_ids  = {(r.payload or {}).get("chunk_id") for r in news_results}
        know_ids  = {(r.payload or {}).get("chunk_id") for r in know_results}
        overlap   = news_ids & know_ids - {None}
        if overlap:
            detail_parts.append(f"⚠ {len(overlap)} chunk_id overlap(s)")

        if ok:
            if verbose:
                _ok(f"'{query}'", "  ".join(detail_parts))
            passed += 1
        else:
            failed += 1

    return passed, failed


def run_dual_evaluation(verbose: bool = False) -> bool:
    """Callable entry point for app/cli.py evaluate-dual."""
    router     = MemoryRouter()
    qdrant_cfg = load_qdrant_config()

    # Routing evaluation (no Qdrant needed)
    print(f"\nRouting evaluation — {len(ROUTING_CASES)} cases…")
    r_pass, r_fail = run_routing_eval(ROUTING_CASES, router, verbose=verbose)
    print(f"  Routing: passed={r_pass}  failed={r_fail}")

    # Dual retrieval evaluation — share one Qdrant client to avoid embedded storage lock
    print(f"\nDual retrieval evaluation — {len(DUAL_RETRIEVAL_CASES)} cases…")
    try:
        from services.config_service import make_client
        from services.embedding_service import EmbeddingService

        shared_client = make_client(qdrant_cfg)

        primary_cfg = qdrant_cfg["collections"]["primary"]
        knowledge_cfg = qdrant_cfg["collections"]["knowledge"]

        news_service = RetrievalService(
            client=shared_client,
            collection_name=primary_cfg.get("live_alias") or primary_cfg["name"],
            embedding_service=EmbeddingService.from_config(primary_cfg),
        )
        know_service = RetrievalService(
            client=shared_client,
            collection_name=knowledge_cfg.get("live_alias") or knowledge_cfg["name"],
            embedding_service=EmbeddingService.from_config(knowledge_cfg),
        )
    except Exception as exc:
        print(f"Error connecting to Qdrant: {exc}")
        print("Run: python app/cli.py ingest  and  python app/cli.py ingest-knowledge")
        return False

    d_pass, d_fail = run_dual_retrieval_eval(
        DUAL_RETRIEVAL_CASES, news_service, know_service, verbose=verbose
    )
    print(f"  Dual retrieval: passed={d_pass}  failed={d_fail}")

    total_pass = r_pass + d_pass
    total_fail = r_fail + d_fail

    print(f"\n{'─' * 40}")
    print(f"Passed: {total_pass}  |  Failed: {total_fail}  |  Total: {total_pass + total_fail}")
    if total_fail == 0:
        print("✓ All dual eval cases passed.")
    else:
        print(f"✗ {total_fail} case(s) failed.")
    return total_fail == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run dual-retrieval and routing eval cases."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ok = run_dual_evaluation(verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
