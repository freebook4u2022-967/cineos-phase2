"""Release gates for controlled Alpha candidates."""

from dataclasses import dataclass

from cineos.benchmarks.regression import Regression, Severity

from .manifest import ReleaseManifest

GATES = (
    "mandatory_suite",
    "deterministic_fixtures",
    "cli_smoke",
    "studio_smoke",
    "preview_pipeline",
    "documentation",
    "package_integrity",
)


@dataclass(frozen=True, slots=True)
class GateResult:
    approved: bool
    failed_checks: tuple[str, ...]


def evaluate_gates(
    manifest: ReleaseManifest,
    checks: dict[str, bool],
    regressions: tuple[Regression, ...] = (),
) -> GateResult:
    failed = [name for name in GATES if not checks.get(name, False)]
    if manifest.validate():
        failed.append("version_manifest")
    if any(item.severity is Severity.BLOCKING for item in regressions):
        failed.append("blocking_regressions")
    return GateResult(not failed, tuple(failed))
