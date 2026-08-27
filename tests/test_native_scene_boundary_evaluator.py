from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_video import FFmpegSceneBoundaryEvaluator, SceneBoundaryPoint


def test_ffmpeg_scene_boundary_evaluator_measures_temporal_boundary_pixels(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(
        sample_width=4,
        sample_height=2,
        sample_offset_seconds=0.05,
        sample_count=3,
        sample_stride_seconds=0.04,
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
    assert len(calls) == 6
    timestamps = [float(call[call.index("-ss") + 1]) for call in calls]
    assert timestamps == pytest.approx([1.87, 1.91, 1.95, 2.05, 2.09, 2.13])


def test_temporal_window_rejects_transient_match_boundary_drift(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(
        sample_width=4,
        sample_height=2,
        sample_offset_seconds=0.05,
        sample_count=3,
        sample_stride_seconds=0.04,
    )

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which",
        lambda _name: "/usr/bin/ffmpeg",
    )

    def fake_run(command, *, check, capture_output):
        assert check is True
        assert capture_output is True
        timestamp = float(command[command.index("-ss") + 1])
        if timestamp < 2.0:
            value = 80
        elif timestamp == pytest.approx(2.09):
            value = 220
        else:
            value = 88
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

    assert report.decision == "reject"
    assert report.boundaries[0].boundary_mad > evaluator.policy.match_reject_mad


def test_temporal_window_rejects_single_black_frame_hidden_by_window_average(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(
        sample_width=4,
        sample_height=2,
        sample_offset_seconds=0.05,
        sample_count=3,
        sample_stride_seconds=0.04,
    )

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which",
        lambda _name: "/usr/bin/ffmpeg",
    )

    def fake_run(command, *, check, capture_output):
        assert check is True
        assert capture_output is True
        timestamp = float(command[command.index("-ss") + 1])
        if timestamp == pytest.approx(2.09):
            value = 0
        elif timestamp < 2.0:
            value = 80
        else:
            value = 88
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
                transition="cut",
            ),
        ),
    )

    assert report.decision == "reject"
    assert any(
        "transient near-black frame" in directive
        for directive in report.boundaries[0].directives
    )


def test_temporal_window_rejects_peak_match_drift_even_when_mean_stays_below_limit(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(
        sample_width=4,
        sample_height=2,
        sample_offset_seconds=0.05,
        sample_count=3,
        sample_stride_seconds=0.04,
    )

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which",
        lambda _name: "/usr/bin/ffmpeg",
    )

    def fake_run(command, *, check, capture_output):
        assert check is True
        assert capture_output is True
        timestamp = float(command[command.index("-ss") + 1])
        if timestamp < 2.0:
            value = 80
        elif timestamp == pytest.approx(2.09):
            value = 125
        else:
            value = 88
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

    assert report.boundaries[0].boundary_mad < evaluator.policy.match_reject_mad
    assert report.decision == "reject"
    assert any(
        "transient match-boundary drift" in directive
        for directive in report.boundaries[0].directives
    )


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


def test_scene_boundary_evaluator_rejects_window_that_would_clamp_before_movie_start(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(
        sample_offset_seconds=0.05,
        sample_count=3,
        sample_stride_seconds=0.04,
    )
    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which", lambda _name: "/bin/ffmpeg"
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("decoder must not run with an invalid sampling window")

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.subprocess.run", fail_if_called
    )

    with pytest.raises(ValueError, match="too close to movie start"):
        evaluator.evaluate(
            movie,
            (SceneBoundaryPoint("scene-a", "scene-b", 0.10),),
        )


def test_scene_boundary_evaluator_validates_temporal_sampling_policy() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        FFmpegSceneBoundaryEvaluator(sample_count=0)

    with pytest.raises(ValueError, match="sample_stride_seconds"):
        FFmpegSceneBoundaryEvaluator(sample_stride_seconds=0.0)


def test_scene_boundary_point_validates_timestamp_and_transition() -> None:
    with pytest.raises(ValueError, match="boundary_seconds"):
        SceneBoundaryPoint("scene-a", "scene-b", 0.0)

    with pytest.raises(ValueError, match="transition must be"):
        SceneBoundaryPoint("scene-a", "scene-b", 1.0, transition="wipe")
