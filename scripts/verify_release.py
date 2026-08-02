#!/usr/bin/env python3
"""Verify a release manifest using the installed CLI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cineos.cli.main import main

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
args = parser.parse_args()
raise SystemExit(main(["release", "verify", "--manifest", str(args.manifest)]))
