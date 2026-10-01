#!/usr/bin/env python3
"""Run one paper-aligned BRIDGE online ablation.

This entrypoint deliberately exposes only the online variants with existing,
audited experiment settings. The primary method remains ``run_bridge.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from online.bridge_runtime import add_common_arguments, execute


VARIANTS = ("main", "no_integration", "no_graph", "realtime_retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--variant", required=True, choices=VARIANTS,
        help=(
            "main preserves the primary algorithm for the with-BIRD-evidence comparison; "
            "no_integration disables only COLUMN/literal prune_combine; "
            "no_graph disables only graph COLUMN retrieval and gives the detector "
            "the full schema; realtime_retrieval replaces graph COLUMN lookup with "
            "one full-schema LLM retrieval request per parsed SQL column anchor."
        ),
    )
    parser.add_argument(
        "--bird-evidence", action="store_true",
        help="Supply BIRD evidence to all existing online prompt contexts. Default: omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    execute(args, variant=args.variant, include_bird_evidence=args.bird_evidence)
