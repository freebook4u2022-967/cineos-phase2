from pathlib import Path

import pytest

from cineos.atlas.sequence_quality import CineosSequenceQualityEvaluator
from cineos.atlas.video_quality_metrics import (
    MeasuredVideoQualityExtractor,
    VideoQualityMetricError,
)


class FakeFrame:
    def __init__(self, width: int, height: int, rgb: bytes):
        self.shape = (height, width, 3)
        self._rgb = rgb

    def tobytes(self):
        return self._rgb


def _pattern(delta: int = 0) -> FakeFrame:
    values = (45, 70, 95, 120, 145, 170, 195, 220, 80, 130, 180, 205)
    rgb = bytearray()
    for value in values:
        channel = max(8, min(242, value + delta))
        rgb.extend((channel, max(8, channel - 5), min(242, channel + 5)))
    return FakeFrame(4, 3, bytes(rgb))


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.mp4"
    path.write_bytes(b"real-render-placeholder-for-injected-decoder")
    return path


def test_measured_video_metrics_accept_stable_sampled_frames(tmp_path):
    artifact = _artifact(tmp_path)
    observed = {}

    def identity_source(output_path, *, shot, frames, attempt_index):
        observed["path"] = output_path
        observed["shot"] = shot
        observed["frames"] = len(frames)
        observed["attempt"] = attempt_index
        return 0.96

    extractor = MeasuredVideoQualityExtractor(
        identity_source=identity_source,
        frame_reader=lambda _path: [_pattern(0), _pattern(2), _pattern(4), _pattern(6)],
        sample_stride=1,
        max_samples=8,
        min_samples=3,
    )
    evaluator = CineosSequenceQualityEvaluator(extractor)

    report = evaluator(str(artifact), shot="shot-01", attempt_index=2)

    assert report["accepted"] is True
    assert report["metrics"]["identity_similarity"] == pytest.approx(0.96)
    assert report["metrics"]["temporal_consistency"] > 0.90
    assert report["metrics"]["artifact_integrity"] > 0.90
    assert report["metrics"]["motion_quality"] > 0.90
    assert observed == {
        "path": str(artifact),
        "shot": "shot-01",
        "frames": 4,
        "attempt": 2,
    }


def test_black_generated_frame_produces_artifact_rejection(tmp_path):
    artifact = _artifact(tmp_path)
    black = FakeFrame(4, 3, bytes([0] * 4 * 3 * 3))
    extractor = MeasuredVideoQualityExtractor(
        identity_source=lambda *_args, **_kwargs: 0.95,
        frame_reader=lambda _path: [_pattern(0), black, _pattern(2), _pattern(4)],
        sample_stride=1,
        min_samples=3,
    )
    evaluator = CineosSequenceQualityEvaluator(extractor)

    report = evaluator(str(artifact), shot="shot-02", attempt_index=0)

    assert report["accepted"] is False
    assert "artifact_integrity" in report["failed_metrics"]
    assert report["metrics"]["artifact_integrity"] < 0.90


def test_dimension_change_fails_closed_before_quality_acceptance(tmp_path):
    artifact = _artifact(tmp_path)
    extractor = MeasuredVideoQualityExtractor(
        identity_source=lambda *_args, **_kwargs: 1.0,
        frame_reader=lambda _path: [
            _pattern(0),
            _pattern(1),
            FakeFrame(2, 2, bytes([80] * 2 * 2 * 3)),
        ],
        sample_stride=1,
        min_samples=3,
    )

    with pytest.raises(VideoQualityMetricError, match="changed dimensions"):
        extractor(str(artifact), shot="shot-03", attempt_index=0)


def test_identity_metric_is_mandatory_and_range_checked(tmp_path):
    artifact = _artifact(tmp_path)
    with pytest.raises(TypeError, match="identity_source"):
        MeasuredVideoQualityExtractor(identity_source=None)  # type: ignore[arg-type]

    extractor = MeasuredVideoQualityExtractor(
        identity_source=lambda *_args, **_kwargs: 1.2,
        frame_reader=lambda _path: [_pattern(0), _pattern(1), _pattern(2)],
        sample_stride=1,
        min_samples=3,
    )
    with pytest.raises(VideoQualityMetricError, match=r"score in \[0, 1\]"):
        extractor(str(artifact), shot="shot-04", attempt_index=0)


def test_short_or_empty_decode_fails_closed(tmp_path):
    artifact = _artifact(tmp_path)
    extractor = MeasuredVideoQualityExtractor(
        identity_source=lambda *_args, **_kwargs: 1.0,
        frame_reader=lambda _path: [_pattern(0)],
        sample_stride=1,
        min_samples=3,
    )

    with pytest.raises(VideoQualityMetricError, match="sampled frames"):
        extractor(str(artifact), shot="shot-05", attempt_index=0)
