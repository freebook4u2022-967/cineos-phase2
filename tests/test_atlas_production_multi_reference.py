from cineos.atlas.production_multi_reference import (
    MULTI_REFERENCE_RUNTIME_SCHEMA,
    ProductionReferenceBoardAdapter,
    bind_production_multi_reference_runtime,
)


def _runtime():
    return {
        "schema": "cineos-gpu-runtime-provenance/0.1",
        "runtime_mode": "default",
        "production_default_runtime": True,
        "injected_boundaries": {
            "torch_module": False,
            "reference_loader": False,
            "pipeline_factory": False,
            "video_exporter": False,
        },
    }


def test_first_party_multi_reference_adapter_remains_production_evidence():
    adapter = ProductionReferenceBoardAdapter()

    bound = bind_production_multi_reference_runtime(_runtime(), adapter)

    assert bound["production_default_runtime"] is True
    assert bound["runtime_mode"] == "default"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is False
    assert bound["multi_reference_conditioning"]["schema"] == (
        MULTI_REFERENCE_RUNTIME_SCHEMA
    )
    assert bound["multi_reference_conditioning"]["adapter_id"] == adapter.adapter_id
    assert bound["multi_reference_conditioning"]["adapter_version"] == (
        adapter.adapter_version
    )


def test_arbitrary_multi_reference_adapter_downgrades_runtime_evidence():
    bound = bind_production_multi_reference_runtime(_runtime(), lambda *_args: None)

    assert bound["production_default_runtime"] is False
    assert bound["runtime_mode"] == "injected"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is True
    assert "multi_reference_conditioning" not in bound


def test_absent_multi_reference_adapter_preserves_default_runtime():
    bound = bind_production_multi_reference_runtime(_runtime(), None)

    assert bound["production_default_runtime"] is True
    assert bound["runtime_mode"] == "default"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is False
