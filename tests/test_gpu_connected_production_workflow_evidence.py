"""Regression coverage for production workflow evidence retention."""

from pathlib import Path


def test_gpu_workflow_persists_input_preflight_evidence() -> None:
    workflow = Path(".github/workflows/gpu-connected-production.yml").read_text(
        encoding="utf-8"
    )

    assert "production-input-preflight.json" in workflow
    assert 'tee "$output_dir/production-input-preflight.json"' in workflow
    assert 'test -s "$output_dir/production-input-preflight.json"' in workflow
    assert "path: ${{ runner.temp }}/cineos-connected-production" in workflow
