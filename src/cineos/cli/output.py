"""Human-readable and structured CLI output."""

import json
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass(slots=True)
class Output:
    """Write one consistent output format for every command."""

    json_mode: bool = False
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)

    def success(self, message: str, **details: Any) -> None:
        if self.json_mode:
            print(
                json.dumps({"ok": True, "message": message, **details}, sort_keys=True),
                file=self.stdout,
            )
        else:
            print(message, file=self.stdout)

    def error(self, message: str, *, code: int, hint: str | None = None) -> None:
        if self.json_mode:
            payload = {"ok": False, "error": message, "exit_code": code}
            if hint:
                payload["hint"] = hint
            print(json.dumps(payload, sort_keys=True), file=self.stderr)
        else:
            print(f"error: {message}", file=self.stderr)
            if hint:
                print(f"hint: {hint}", file=self.stderr)
