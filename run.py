#!/usr/bin/env python3
"""Thin local orchestration entry point; all real work is in package modules.

DO NOT couple config creation, calculation, result querying, or plotting here.
This file only passes CLI arguments into the isolated run adapter.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ofc.run_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
