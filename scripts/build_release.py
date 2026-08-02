#!/usr/bin/env python3
"""Build wheel and source artifacts without bundling external media or weights."""

import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, "-m", "build", "--no-isolation"]))
