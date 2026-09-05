from pathlib import Path


def _workflow_text() -> str:
    return Path(".github/workflows/gpu-connected-production.yml").read_text(
        encoding="utf-8"
    )


def test_gpu_workflow_prefetches_and_verifies_pinned_foundation_revision():
    workflow = _workflow_text()

    assert "Prefetch and verify immutable foundation and QC snapshots" in workflow
    assert (
        "from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE" in workflow
    )
    assert "snapshot_download(" in workflow
    assert "repo_id=provenance.model_id" in workflow
    assert "revision=revision" in workflow
    assert "resolved_revision != revision" in workflow
    assert "foundation snapshot resolved to unexpected revision" in workflow


def test_gpu_workflow_prefetches_and_verifies_pinned_learned_qc_revision():
    workflow = _workflow_text()

    assert "from cineos.atlas.siglip2_video_scorer import (" in workflow
    assert "SIGLIP2_QC_MODEL_ID" in workflow
    assert "SIGLIP2_QC_REVISION" in workflow
    assert "repo_id=SIGLIP2_QC_MODEL_ID" in workflow
    assert "revision=SIGLIP2_QC_REVISION" in workflow
    assert "resolved_qc_revision != SIGLIP2_QC_REVISION" in workflow
    assert "QC snapshot resolved to unexpected revision" in workflow


def test_gpu_workflow_uses_same_hf_cache_for_prefetch_and_render():
    workflow = _workflow_text()

    cache_binding = "HF_HOME: ${{ runner.temp }}/cineos-hf-cache"
    assert workflow.count(cache_binding) == 2


def test_gpu_workflow_uses_cineos_memory_planner_before_foundation_download():
    workflow = _workflow_text()

    planner_import = "from cineos.atlas.gpu_preflight import ("
    planner_call = "plan = select_gpu_execution("
    model_requirement = (
        "estimated_model_vram_gb=WAN22_TI2V_5B_PROFILE.minimum_gpu_vram_gb"
    )
    prefetch_step = "- name: Prefetch and verify immutable foundation and QC snapshots"

    assert planner_import in workflow
    assert planner_call in workflow
    assert model_requirement in workflow
    assert workflow.index(planner_call) < workflow.index(prefetch_step)


def test_gpu_workflow_records_selected_memory_strategy_for_audit_logs():
    workflow = _workflow_text()

    assert 'print(f"selected_device={plan.device}")' in workflow
    assert 'print(f"memory_strategy={plan.memory_strategy}")' in workflow
    assert (
        'print(f"estimated_model_vram_gb={plan.estimated_model_vram_gb:.2f}")'
        in workflow
    )
    assert 'print(f"fit_margin_gb={plan.fit_margin_gb:.2f}")' in workflow


def test_gpu_workflow_runs_production_cli_after_qc_snapshot_is_pinned():
    workflow = _workflow_text()

    prefetch_step = "- name: Prefetch and verify immutable foundation and QC snapshots"
    run_step = "- name: Run real connected GPU benchmark with production visual QC"
    assert workflow.index(prefetch_step) < workflow.index(run_step)
    assert "python -m cineos.atlas.gpu_benchmark_cli" in workflow
