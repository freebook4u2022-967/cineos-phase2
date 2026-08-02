"""Clean-environment installation command construction."""

import sys
from pathlib import Path


def install_command(artifact: Path) -> tuple[str, ...]:
    return (sys.executable, "-m", "pip", "install", "--no-deps", str(artifact))
