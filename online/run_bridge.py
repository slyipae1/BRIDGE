#!/usr/bin/env python3
"""Run the frozen public BRIDGE online repair route."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from online.bridge_runtime import add_common_arguments, execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
