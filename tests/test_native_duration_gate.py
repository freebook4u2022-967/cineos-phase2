from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cineos.native_video.duration_gate import (
    DurationIntegrityPolicy,
    DurationProbeError,
    FFprobeDurationIntegrityGate,
    evaluate_duration_integrity,
    planned_duration,
)


@dataclass(frozen=True, slots=True)
class Shot:
    duration: float


def test_planned_duration_sums_authored_shots() -> None:
    assert planned_duration([Shot(1.25), Shot(2.0), Shot(0.75)]) == pytest.approx(4.0)


def test_planned_duration_rejects_empty_or_invalid_plan() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        planned_duration([])
    with pytest.raises(ValueError, match="positive"):
        planned_duration([Shot(1.0), Shot(0.0)])


def test_duration_integrity_accepts_small_container_rounding_error() -> None:
    report = evaluate_duration_integrity(10.0, 10.18)

    assert report.accepted
    assert report.decision == "accept"
    assert report.allowed_error_seconds == pytest.approx(0.25)
    assert report.directives == ()


def test_duration_integrity_uses_relative_tolerance_for_long_films() -> None:
    policy = DurationIntegrityPolicy(
        absolute_tolerance_seconds=0.25,
        relative_tolerance=0.01,
    )
    report = evaluate_duration_integrity(120.0, 120.9, policy)

    assert report.accepted
    assert report.allowed_error_seconds == pytest.approx(1.2)


def test_duration_integrity_rejects_truncated_movie() -> None:
    report = evaluate_duration_integrity(30.0, 27.0)

    assert not report.accepted
    assert report.decision == "reject"
    assert report.delta_seconds == pytest.approx(-3.0)
    assert "shorter" in report.directives[0]
    assert "rebuild assembly" in report.directives[0]


def test_duration_integrity_rejects_duplicated_or_overlong_movie() -> None:
    report = evaluate_duration_integrity(30.0, 34.0)

    assert not report.accepted
    assert report.delta_seconds == pytest.approx(4.0)
    assert "longer" in report.directives[0]


def test_duration_policy_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DurationIntegrityPolicy(absolute_tolerance_seconds=-0.1)
    with pytest.raises(ValueError, match="relative_tolerance"):
        DurationIntegrityPolicy(relative_tolerance=1.1)


def test_ffprobe_gate_fails_closed_for_missing_movie(tmp_path: Path) -> None:
    gate = FFprobeDurationIntegrityGate()

    with pytest.raises(DurationProbeError, match="missing or empty"):
        gate.evaluate(tmp_path / "missing.mp4", [Shot(1.0)])


def test_ffprobe_gate_parses_duration_and_compares_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"encoded-movie")

    class Completed:
        returncode = 0
        stdout = "2.04\n"
        stderr = ""

    monkeypatch.setattr(
        "cineos.native_video.duration_gate.shutil.which",
        lambda binary: "/usr/bin/ffprobe" if binary == "ffprobe" else None,
    )
    monkeypatch.setattr(
        "cineos.native_video.duration_gate.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    report = FFprobeDurationIntegrityGate().evaluate(
        movie,
        [Shot(1.0), Shot(1.0)],
    )

    assert report.accepted
    assert report.measured_seconds == pytest.approx(2.04)


def test_ffprobe_gate_rejects_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"encoded-movie")

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "invalid container"

    monkeypatch.setattr(
        "cineos.native_video.duration_gate.shutil.which",
        lambda binary: "/usr/bin/ffprobe",
    )
    monkeypatch.setattr(
        "cineos.native_video.duration_gate.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    with pytest.raises(DurationProbeError, match="failed to inspect"):
        FFprobeDurationIntegrityGate().evaluate(movie, [Shot(1.0)])
