from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_video import FFmpegSceneBoundaryEvaluator, SceneBoundaryPoint


def test_ffmpeg_scene_boundary_evaluator_measures_real_boundary_pixels(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(
        sample_width=4,
        sample_height=2,
        sample_offset_seconds=0.05,
    )
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )

    def fake_run(command, *, check, capture_output):
        assert check is True
        assert capture_output is True
        calls.append(command)
        timestamp = float(command[command.index("-ss") + 1])
        value = 80 if timestamp < 2.0 else 88
        return SimpleNamespace(stdout=bytes([value]) * evaluator.frame_size)

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.subprocess.run",
        fake_run,
    )

    report = evaluator.evaluate(
        movie,
        (
            SceneBoundaryPoint(
                from_scene_id="scene-01",
                to_scene_id="scene-02",
                boundary_seconds=2.0,
                transition="match",
            ),
        ),
    )

    assert report.decision == "accept"
    assert report.boundary_count == 1
    assert report.boundaries[0].boundary_mad == pytest.approx(8.0)
    assert len(calls) == 2
    assert float(calls[0][calls[0].index("-ss") + 1]) == pytest.approx(1.95)
    assert float(calls[1][calls[1].index("-ss") + 1]) == pytest.approx(2.05)


def test_ffmpeg_scene_boundary_evaluator_rejects_incomplete_decoded_evidence(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(sample_width=4, sample_height=2)

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which", lambda _name: "/bin/ffmpeg"
    )
    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"short"),
    )

    with pytest.raises(RuntimeError, match="incomplete scene-boundary frame evidence"):
        evaluator.evaluate(
            movie,
            (
                SceneBoundaryPoint(
                    from_scene_id="scene-a",
                    to_scene_id="scene-b",
                    boundary_seconds=1.0,
                ),
            ),
        )


def test_ffmpeg_scene_boundary_evaluator_fails_closed_on_invalid_timeline(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator()
    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which", lambda _name: "/bin/ffmpeg"
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        evaluator.evaluate(
            movie,
            (
                SceneBoundaryPoint("scene-a", "scene-b", 4.0),
                SceneBoundaryPoint("scene-b", "scene-c", 3.0),
            ),
        )


def test_scene_boundary_point_validates_timestamp_and_transition() -> None:
    with pytest.raises(ValueError, match="boundary_seconds"):
        SceneBoundaryPoint("scene-a", "scene-b", 0.0)

    with pytest.raises(ValueError, match="transition must be"):
        SceneBoundaryPoint("scene-a", "scene-b", 1.0, transition="wipe")
