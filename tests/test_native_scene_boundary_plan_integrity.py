from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_video import FFmpegSceneBoundaryEvaluator, SceneBoundaryPoint


def test_scene_boundary_point_rejects_self_transition() -> None:
    with pytest.raises(ValueError, match="two different scenes"):
        SceneBoundaryPoint("scene-a", "scene-a", 1.0)


def test_scene_boundary_evaluator_rejects_discontinuous_scene_chain_before_decode(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator()

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which",
        lambda _name: "/usr/bin/ffmpeg",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("decoder must not run for a discontinuous boundary plan")

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.subprocess.run",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="scene boundary chain must be contiguous"):
        evaluator.evaluate(
            movie,
            (
                SceneBoundaryPoint("scene-a", "scene-b", 2.0),
                SceneBoundaryPoint("scene-x", "scene-y", 4.0),
            ),
        )


def test_scene_boundary_evaluator_accepts_contiguous_scene_chain(
    tmp_path, monkeypatch
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-movie")
    evaluator = FFmpegSceneBoundaryEvaluator(
        sample_width=4,
        sample_height=2,
        sample_count=1,
    )

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.shutil.which",
        lambda _name: "/usr/bin/ffmpeg",
    )

    def fake_run(command, *, check, capture_output):
        assert check is True
        assert capture_output is True
        timestamp = float(command[command.index("-ss") + 1])
        value = 80 if timestamp < 2.0 else 96
        return SimpleNamespace(stdout=bytes([value]) * evaluator.frame_size)

    monkeypatch.setattr(
        "cineos.native_video.boundary_eval.subprocess.run",
        fake_run,
    )

    report = evaluator.evaluate(
        movie,
        (
            SceneBoundaryPoint("scene-a", "scene-b", 2.0, transition="cut"),
            SceneBoundaryPoint("scene-b", "scene-c", 4.0, transition="cut"),
        ),
    )

    assert report.boundary_count == 2
    assert report.decision in {"accept", "warn"}
