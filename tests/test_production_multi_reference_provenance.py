from cineos.atlas.production_multi_reference import (
    PRODUCTION_REFERENCE_BOARD_ADAPTER_ID,
    PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION,
    ProductionReferenceBoardAdapter,
    bind_production_multi_reference_runtime,
)


def _runtime() -> dict[str, object]:
    return {
        "injected_boundaries": {
            "pipeline_factory": False,
            "reference_loader": False,
            "multi_reference_adapter": True,
        }
    }


def test_exact_first_party_multi_reference_adapter_is_promoted() -> None:
    adapter = ProductionReferenceBoardAdapter()

    bound = bind_production_multi_reference_runtime(_runtime(), adapter)

    assert bound["production_default_runtime"] is True
    assert bound["runtime_mode"] == "default"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is False
    provenance = bound["multi_reference_conditioning"]
    assert provenance["adapter_id"] == PRODUCTION_REFERENCE_BOARD_ADAPTER_ID
    assert provenance["adapter_version"] == PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION


def test_subclass_cannot_masquerade_as_first_party_multi_reference_adapter() -> None:
    class SubstitutedAdapter(ProductionReferenceBoardAdapter):
        adapter_id = PRODUCTION_REFERENCE_BOARD_ADAPTER_ID
        adapter_version = PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION

        def runtime_provenance(self) -> dict[str, object]:
            return {
                "adapter_id": PRODUCTION_REFERENCE_BOARD_ADAPTER_ID,
                "adapter_version": PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION,
                "spoofed": True,
            }

    bound = bind_production_multi_reference_runtime(_runtime(), SubstitutedAdapter())

    assert bound["production_default_runtime"] is False
    assert bound["runtime_mode"] == "injected"
    assert bound["injected_boundaries"]["multi_reference_adapter"] is True
    assert "multi_reference_conditioning" not in bound


def test_instance_attribute_spoof_does_not_change_canonical_runtime_provenance() -> None:
    adapter = ProductionReferenceBoardAdapter()
    adapter.adapter_id = "substituted.adapter"
    adapter.adapter_version = "999"
    adapter.maximum_references = 99

    provenance = adapter.runtime_provenance()

    assert provenance["adapter_id"] == PRODUCTION_REFERENCE_BOARD_ADAPTER_ID
    assert provenance["adapter_version"] == PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION
    assert provenance["maximum_references"] == 4
