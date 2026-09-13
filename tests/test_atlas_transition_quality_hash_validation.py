from types import SimpleNamespace

import pytest

from cineos.atlas.transition_quality import (
    TRANSITION_QUALITY_SCHEMA,
    TransitionQualityError,
    validate_transition_quality_evidence,
)


def _report() -> dict[str, object]:
    return {
        "schema": TRANSITION_QUALITY_SCHEMA,
        "production_measurement_evidence": True,
        "accepted": True,
        "observer_id": "measured-transition-observer/v1",
    }


def test_transition_evidence_rejects_non_hex_predecessor_receipt_hash() -> None:
    previous_receipt = SimpleNamespace(output_sha256="z" * 64)
    current_receipt = SimpleNamespace(output_sha256="a" * 64)

    with pytest.raises(TransitionQualityError, match="hexadecimal SHA-256"):
        validate_transition_quality_evidence(
            _report(),
            previous_receipt=previous_receipt,
            current_receipt=current_receipt,
            current_request=SimpleNamespace(),
        )


def test_transition_evidence_rejects_non_hex_current_receipt_hash() -> None:
    previous_receipt = SimpleNamespace(output_sha256="a" * 64)
    current_receipt = SimpleNamespace(output_sha256="g" * 64)

    with pytest.raises(TransitionQualityError, match="hexadecimal SHA-256"):
        validate_transition_quality_evidence(
            _report(),
            previous_receipt=previous_receipt,
            current_receipt=current_receipt,
            current_request=SimpleNamespace(),
        )
