#!/usr/bin/env python3
"""Run the deterministic Alpha benchmark suite."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cineos.benchmarks import BenchmarkRunner, alpha_suite

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/reports/smoke"))
parser.add_argument("--mandatory-only", action="store_true")
args = parser.parse_args()
report = BenchmarkRunner().run(
    alpha_suite(), args.output_dir, mandatory_only=args.mandatory_only
)
raise SystemExit(0 if report.passed else 1)
