from pathlib import Path


def test_gpu_workflow_prefetches_and_verifies_pinned_foundation_revision():
    workflow = Path(".github/workflows/gpu-connected-production.yml").read_text(
        encoding="utf-8"
    )

    assert "Prefetch and verify immutable foundation snapshot" in workflow
    assert (
        "from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE" in workflow
    )
    assert "snapshot_download(" in workflow
    assert "repo_id=provenance.model_id" in workflow
    assert "revision=revision" in workflow
    assert "resolved_revision != revision" in workflow
    assert "foundation snapshot resolved to unexpected revision" in workflow


def test_gpu_workflow_uses_same_hf_cache_for_prefetch_and_render():
    workflow = Path(".github/workflows/gpu-connected-production.yml").read_text(
        encoding="utf-8"
    )

    cache_binding = "HF_HOME: ${{ runner.temp }}/cineos-hf-cache"
    assert workflow.count(cache_binding) == 2
