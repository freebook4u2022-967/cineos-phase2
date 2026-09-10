"""Regression coverage for production workflow evidence retention."""

from pathlib import Path


def _workflow() -> str:
    return Path(".github/workflows/gpu-connected-production.yml").read_text(
        encoding="utf-8"
    )


def test_gpu_workflow_persists_input_preflight_evidence() -> None:
    workflow = _workflow()

    assert "production-input-preflight.json" in workflow
    assert 'tee "$output_dir/production-input-preflight.json"' in workflow
    assert 'test -s "$output_dir/production-input-preflight.json"' in workflow
    assert "path: ${{ runner.temp }}/cineos-connected-production" in workflow


def test_gpu_workflow_binds_model_acquisition_to_persisted_selection() -> None:
    workflow = _workflow()

    assert "foundation-selection-preacquisition.json" in workflow
    assert (
        "SELECTION_PATH: ${{ runner.temp }}/cineos-connected-production/foundation-selection-preacquisition.json"
        in workflow
    )
    assert (
        'selection = json.loads(Path(os.environ["SELECTION_PATH"]).read_text(encoding="utf-8"))'
        in workflow
    )
    assert 'model_id = selection.get("model_id")' in workflow
    assert 'revision = selection.get("revision")' in workflow
    assert (
        "select_strongest_production_foundation(\n              inspect_cuda_environment(), requests\n          )"
        not in workflow
    )


def test_gpu_workflow_rejects_foundation_drift_after_acquisition() -> None:
    workflow = _workflow()

    assert "Verify execution did not drift from acquired foundation" in workflow
    assert 'root / "foundation-selection-preacquisition.json"' in workflow
    assert 'root / "foundation-selection.json"' in workflow
    assert '"profile_id",' in workflow
    assert '"model_id",' in workflow
    assert '"revision",' in workflow
    assert '"license_id",' in workflow
    assert '"source_url",' in workflow
    assert '"foundation_name",' in workflow
    assert "foundation selection drifted after acquisition" in workflow
