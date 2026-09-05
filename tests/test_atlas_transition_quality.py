from types import SimpleNamespace

import pytest

from cineos.atlas.transition_quality import (
    TRANSITION_QUALITY_SCHEMA,
    TransitionQualityError,
    validate_transition_quality_evidence,
)


def _receipt(scene_id: str, shot_id: str, digest: str) -> SimpleNamespace:
    return SimpleNamespace(
        output_sha256=digest,
        result=SimpleNamespace(scene_id=scene_id, shot_id=shot_id),
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        scene_id="scene-1",
        shot_id="shot-2",
        continuity={"previous_shot": "shot-1"},
    )


def _report() -> dict[str, object]:
    return {
        "schema": TRANSITION_QUALITY_SCHEMA,
        "production_measurement_evidence": True,
        "accepted": True,
        "observer_id": "terminal-initial-embedding-v1",
        "previous_scene_id": "scene-1",
        "previous_shot_id": "shot-1",
        "current_scene_id": "scene-1",
        "current_shot_id": "shot-2",
        "previous_output_sha256": "a" * 64,
        "current_output_sha256": "b" * 64,
        "measured_sample_count": 4,
        "metrics": {"visual_seam_similarity": 0.91},
    }


def test_transition_evidence_binds_both_artifacts_and_lineage() -> None:
    validated = validate_transition_quality_evidence(
        _report(),
        previous_receipt=_receipt("scene-1", "shot-1", "a" * 64),
        current_receipt=_receipt("scene-1", "shot-2", "b" * 64),
        current_request=_request(),
    )

    assert validated["accepted"] is True
    assert validated["measured_sample_count"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("previous_output_sha256", "c" * 64),
        ("current_output_sha256", "c" * 64),
        ("previous_shot_id", "wrong-shot"),
        ("current_shot_id", "wrong-shot"),
    ],
)
def test_transition_evidence_rejects_substituted_artifact_or_identity(
    field: str,
    value: str,
) -> None:
    report = _report()
    report[field] = value

    with pytest.raises(TransitionQualityError):
        validate_transition_quality_evidence(
            report,
            previous_receipt=_receipt("scene-1", "shot-1", "a" * 64),
            current_receipt=_receipt("scene-1", "shot-2", "b" * 64),
            current_request=_request(),
        )


def test_transition_evidence_rejects_nonproduction_measurement() -> None:
    report = _report()
    report["production_measurement_evidence"] = False

    with pytest.raises(TransitionQualityError, match="not production"):
        validate_transition_quality_evidence(
            report,
            previous_receipt=_receipt("scene-1", "shot-1", "a" * 64),
            current_receipt=_receipt("scene-1", "shot-2", "b" * 64),
            current_request=_request(),
        )


def test_transition_evidence_rejects_zero_measured_samples() -> None:
    report = _report()
    report["measured_sample_count"] = 0

    with pytest.raises(TransitionQualityError, match="at least one"):
        validate_transition_quality_evidence(
            report,
            previous_receipt=_receipt("scene-1", "shot-1", "a" * 64),
            current_receipt=_receipt("scene-1", "shot-2", "b" * 64),
            current_request=_request(),
        )


def test_transition_evidence_rejects_request_with_wrong_predecessor() -> None:
    request = _request()
    request.continuity = {"previous_shot": "shot-x"}

    with pytest.raises(TransitionQualityError, match="predecessor"):
        validate_transition_quality_evidence(
            _report(),
            previous_receipt=_receipt("scene-1", "shot-1", "a" * 64),
            current_receipt=_receipt("scene-1", "shot-2", "b" * 64),
            current_request=request,
        )
