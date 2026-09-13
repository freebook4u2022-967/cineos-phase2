from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/gpu-connected-production.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_production_dependency_gate_runs_before_foundation_acquisition() -> None:
    workflow = _workflow_text()

    gate = workflow.index(
        "Fail closed on unavailable production dependencies before model acquisition"
    )
    selection = workflow.index(
        "Select strongest safe production foundation on live CUDA runner"
    )
    acquisition = workflow.index(
        "Prefetch and verify selected immutable foundation and QC snapshots"
    )

    assert gate < selection < acquisition
    preselection = workflow[gate:selection]
    assert "quality_profile_required=False" in preselection
    assert "quality_profile_required=True" not in preselection
    assert "production-dependency-readiness.json" in workflow


def test_production_dependency_gate_binds_only_requested_approved_references() -> None:
    workflow = _workflow_text()

    assert "for request in requests" in workflow
    assert "for reference_id in request.approved_reference_ids" in workflow
    assert "missing requested ids" in workflow
    assert "IdentityAssetRequirement(*manifest_entries[reference_id])" in workflow
    assert "collect_cuda_capacity()" in workflow
    assert 'shutil.which("ffmpeg") is not None' in workflow
    assert 'shutil.which("ffprobe") is not None' in workflow
    assert "if not report.ready:" in workflow
    assert "production dependency gate blocked model acquisition" in workflow
