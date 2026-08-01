from pathlib import Path

import pytest

from cineos.validation import (
    FakeValidatorBackend,
    TemporalValidator,
    ValidationPipeline,
    ValidationStatus,
    ValidationThresholds,
)
from cineos.validation.serializer import load, save


@pytest.fixture
def render(tmp_path: Path) -> Path:
    path = tmp_path / "shot.mp4"
    path.write_bytes(b"test render")
    return path


@pytest.fixture
def conditioning():
    return {
        "character_conditioning": [{"approved_reference_ids": ["ref"]}],
        "wardrobe_conditioning": [{"wardrobe_asset_id": "coat"}],
        "prop_conditioning": [{"asset_id": "key"}],
        "environment_conditioning": {"environment_asset_id": "room"},
    }


def test_pass_warn_and_fail_thresholds(render, conditioning):
    assert (
        ValidationPipeline(FakeValidatorBackend())
        .validate(render, conditioning)
        .overall_status
        is ValidationStatus.PASS
    )
    warn = ValidationPipeline(
        FakeValidatorBackend(scores={"identity.face": None})
    ).validate(render, conditioning)
    assert warn.overall_status is ValidationStatus.PASS_WITH_WARNINGS
    failed = ValidationPipeline(
        FakeValidatorBackend(scores={"wardrobe.colors": 0.0})
    ).validate(render, conditioning)
    assert failed.overall_status is ValidationStatus.FAIL
    assert failed.should_rerender


def test_missing_references_require_review(render):
    report = ValidationPipeline().validate(render, {})
    assert report.overall_status is ValidationStatus.MANUAL_REVIEW_REQUIRED


def test_unsupported_validator(render, conditioning):
    backend = FakeValidatorBackend(temporal={})
    backend.temporal_metrics = lambda frames: None
    report = ValidationPipeline(backend, validators=[TemporalValidator()]).validate(
        render, conditioning
    )
    assert report.overall_status is ValidationStatus.UNSUPPORTED


def test_temporal_drift_detection(render, conditioning):
    report = ValidationPipeline(
        FakeValidatorBackend(temporal={"face_drift": 0.8}),
        validators=[TemporalValidator()],
    ).validate(render, conditioning)
    assert report.overall_status is ValidationStatus.FAIL
    assert "face_drift" in report.failures[0]


@pytest.mark.parametrize(
    ("capability", "category"),
    [
        ("wardrobe.asset", "wardrobe"),
        ("props.presence", "props"),
        ("environment.asset", "environment"),
    ],
)
def test_category_mismatch(render, conditioning, capability, category):
    report = ValidationPipeline(
        FakeValidatorBackend(scores={capability: 0.0})
    ).validate(render, conditioning)
    result = next(item for item in report.results if item.category == category)
    assert result.status is ValidationStatus.FAIL


def test_serialization_round_trip(render, conditioning, tmp_path):
    report = ValidationPipeline().validate(render, conditioning)
    path = tmp_path / "report.json"
    save(report, path)
    restored = load(path)
    assert restored.report_uuid == report.report_uuid
    assert restored.content_hash == report.content_hash
    assert restored.results[0].status is ValidationStatus.PASS


def test_threshold_validation():
    with pytest.raises(ValueError):
        ValidationThresholds(pass_threshold=0.5, warning_threshold=0.8)
