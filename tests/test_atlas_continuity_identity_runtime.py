from cineos.atlas.production_continuity_identity import (
    CONTINUITY_IDENTITY_ADAPTER_ID,
    CONTINUITY_IDENTITY_ADAPTER_VERSION,
    compose_continuity_identity_board,
)
from cineos.atlas.production_continuity_identity_runtime import (
    CONTINUITY_IDENTITY_RUNTIME_SCHEMA,
    bind_continuity_identity_runtime,
)


def _runtime():
    return {
        "schema": "cineos-gpu-runtime-provenance/0.1",
        "runtime_mode": "default",
        "production_default_runtime": True,
        "cuda_device": "cuda:0",
        "dtype": "bfloat16",
        "injected_boundaries": {
            "torch_module": False,
            "reference_loader": False,
            "pipeline_factory": False,
            "video_exporter": False,
        },
    }


def test_baseline_strategy_remains_default_production_runtime():
    bound = bind_continuity_identity_runtime(_runtime(), None)

    assert bound["runtime_mode"] == "default"
    assert bound["production_default_runtime"] is True
    assert bound["injected_boundaries"]["continuity_identity_adapter"] is False
    assert bound["continuity_identity_strategy"] == {
        "schema": CONTINUITY_IDENTITY_RUNTIME_SCHEMA,
        "mode": "predecessor_terminal_frame_baseline",
        "adapter_id": None,
        "adapter_version": None,
        "experimental": False,
    }


def test_exact_first_party_candidate_is_production_but_stays_experimental():
    bound = bind_continuity_identity_runtime(
        _runtime(), compose_continuity_identity_board
    )

    assert bound["runtime_mode"] == "default"
    assert bound["production_default_runtime"] is True
    strategy = bound["continuity_identity_strategy"]
    assert strategy["schema"] == CONTINUITY_IDENTITY_RUNTIME_SCHEMA
    assert strategy["mode"] == "predecessor_terminal_frame_plus_fresh_references"
    assert strategy["adapter_id"] == CONTINUITY_IDENTITY_ADAPTER_ID
    assert strategy["adapter_version"] == CONTINUITY_IDENTITY_ADAPTER_VERSION
    assert strategy["experimental"] is True


def test_unrecognized_adapter_cannot_masquerade_as_production_candidate():
    def injected_adapter(*args, **kwargs):
        return None

    bound = bind_continuity_identity_runtime(_runtime(), injected_adapter)

    assert bound["runtime_mode"] == "injected"
    assert bound["production_default_runtime"] is False
    assert bound["injected_boundaries"]["continuity_identity_adapter"] is True
    assert bound["continuity_identity_strategy"]["mode"] == (
        "injected_or_unrecognized"
    )
