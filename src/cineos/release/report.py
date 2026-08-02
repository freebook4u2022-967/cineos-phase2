"""Release gate report."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReleaseReport:
    test_summary: dict[str, object]
    benchmark_summary: dict[str, object]
    regression_summary: dict[str, object]
    hardware_profile: dict[str, object]
    renderer_profile: str
    known_limitations: tuple[str, ...]
    failed_checks: tuple[str, ...] = ()
    approval_status: str = "not-approved"

    @property
    def approved(self) -> bool:
        return self.approval_status == "approved" and not self.failed_checks
