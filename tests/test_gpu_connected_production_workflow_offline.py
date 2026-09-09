"""Regression guard for immutable-cache-only production GPU execution."""

from pathlib import Path


WORKFLOW = Path(".github/workflows/gpu-connected-production.yml")


def test_production_gpu_benchmark_runs_hugging_face_boundaries_offline():
    text = WORKFLOW.read_text(encoding="utf-8")
    benchmark_step = text.split(
        "- name: Run quality-first connected GPU benchmark with production visual QC",
        maxsplit=1,
    )[1].split("- name: Upload production GPU evidence", maxsplit=1)[0]

    assert 'HF_HUB_OFFLINE: "1"' in benchmark_step
    assert 'TRANSFORMERS_OFFLINE: "1"' in benchmark_step
    assert "quality_first_gpu_benchmark_cli" in benchmark_step


def test_prefetch_remains_online_before_offline_execution_boundary():
    text = WORKFLOW.read_text(encoding="utf-8")
    prefetch_step = text.split(
        "- name: Prefetch and verify selected immutable foundation and QC snapshots",
        maxsplit=1,
    )[1].split(
        "- name: Run quality-first connected GPU benchmark with production visual QC",
        maxsplit=1,
    )[0]

    assert "snapshot_download(" in prefetch_step
    assert "revision=revision" in prefetch_step
    assert "revision=SIGLIP2_QC_REVISION" in prefetch_step
    assert "HF_HUB_OFFLINE" not in prefetch_step
