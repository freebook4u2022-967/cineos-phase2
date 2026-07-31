"""CLI-specific error types and stable process exit codes."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Public exit status contract for automation."""

    SUCCESS = 0
    USAGE = 2
    INPUT = 3
    VALIDATION = 4
    EXECUTION = 5


class CLIError(Exception):
    """An actionable failure suitable for presentation to a CLI user."""

    def __init__(
        self, message: str, *, code: ExitCode, hint: str | None = None
    ) -> None:
        self.code = code
        self.hint = hint
        super().__init__(message)
