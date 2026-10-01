#!/usr/bin/env python3
"""Build the reusable text-value MinHash-LSH artifacts for one database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
if str(OFFLINE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFLINE_ROOT))

from auto_construct_graph.value_lsh import build_db_lsh  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, required=True, help="Database directory containing {db_id}.sqlite.")
    parser.add_argument("--signature-size", type=int, default=100)
    parser.add_argument("--n-gram", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.signature_size <= 0 or args.n_gram <= 0 or not 0.0 <= args.threshold <= 1.0:
        raise ValueError("signature size and n-gram must be positive; threshold must be in [0, 1]")
    manifest = build_db_lsh(
        args.db_dir,
        signature_size=args.signature_size,
        n_gram=args.n_gram,
        threshold=args.threshold,
        verbose=not args.quiet,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
