#!/usr/bin/env python3
"""
Generation smoke tests.
Runs a dry-run drafting pipeline call (no Claude API hit) always.
Runs a live Claude API call when --live flag is provided and ANTHROPIC_API_KEY is set.

Usage:
    python tests/test_generation.py            # dry-run only
    python tests/test_generation.py --live     # also calls Claude API
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from config import load_generation_config, load_qdrant_config
from pipelines.drafting_pipeline import DraftingPipeline


def test_dry_run():
    qdrant_cfg = load_qdrant_config()
    gen_cfg    = load_generation_config().get("generation", {})
    pipeline   = DraftingPipeline.from_config(qdrant_cfg=qdrant_cfg, gen_cfg=gen_cfg)
    result     = pipeline.draft(topic="fire safety training drill", dry_run=True)
    assert result is None, "dry_run should return None"
    print("  [OK]   dry_run smoke test passed")


def test_live_generation(topic: str = "welding competition award"):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [SKIP] ANTHROPIC_API_KEY not set \u2014 skipping live generation test")
        return

    qdrant_cfg = load_qdrant_config()
    gen_cfg    = load_generation_config().get("generation", {})
    pipeline   = DraftingPipeline.from_config(
        qdrant_cfg=qdrant_cfg, gen_cfg=gen_cfg, api_key=api_key
    )
    result = pipeline.draft(topic=topic, stream=False)
    assert result is not None,             "result should not be None"
    assert len(result.article_text) > 100, "article_text should be non-trivial"
    assert len(result.sources_used) > 0,   "sources_used should be populated"
    assert result.draft_id,                "draft_id should be set"
    print(
        f"  [OK]   live generation test passed"
        f" \u2014 {result.input_tokens} input / {result.output_tokens} output tokens"
    )
    print(f"         draft_id: {result.draft_id}")
    print(f"         sources : {[s['slug'] for s in result.sources_used]}")


def main():
    parser = argparse.ArgumentParser(description="Generation smoke tests.")
    parser.add_argument("--live", action="store_true", help="Also run live Claude API test")
    args = parser.parse_args()

    print("\nRunning generation smoke tests…")
    test_dry_run()
    if args.live:
        test_live_generation()
    print("\nDone.")


if __name__ == "__main__":
    main()
