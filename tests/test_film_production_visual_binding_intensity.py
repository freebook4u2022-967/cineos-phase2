import random

import pytest

from cineos.film import visual_binding
from cineos.film.visual_binding import VisualBindingError, measure_visual_binding


def _frames(seed: int, count: int = 8) -> bytes:
    rng = random.Random(seed)
    return bytes(
        rng.randrange(0, 256)
        for _ in range(visual_binding.VISUAL_BINDING_FRAME_BYTES * count)
    )


def _offset_pixels(payload: bytes, delta: int) -> bytes:
    return bytes(max(0, min(255, value + delta)) for value in payload)


def _measure(tmp_path, monkeypatch, approved: bytes, final_frames: bytes):
    shot = tmp_path / "shot-1.mp4"
    final = tmp_path / "film.mp4"
    shot.write_bytes(b"approved-shot")
    final.write_bytes(b"final-film")
    decoded = iter((approved, final_frames))
    monkeypatch.setattr(
        visual_binding,
        "_decode_binding_rgb",
        lambda _path, duration_seconds=None: next(decoded),
    )
    return measure_visual_binding([shot], final)


def test_visual_binding_accepts_small_transcode_intensity_drift(tmp_path, monkeypatch):
    approved = _frames(811)
    evidence = _measure(tmp_path, monkeypatch, approved, _offset_pixels(approved, 12))

    assert evidence.accepted is True
    assert evidence.correlation > 0.99
    assert evidence.mean_absolute_error < evidence.maximum_mean_absolute_error
    assert (
        evidence.maximum_mean_absolute_error
        == visual_binding.MAX_VISUAL_BINDING_MEAN_ABSOLUTE_ERROR
    )
    assert evidence.to_dict()["schema"] == "cineos-production-visual-binding/0.4"


def test_visual_binding_rejects_large_brightness_substitution_despite_high_correlation(
    tmp_path, monkeypatch
):
    approved = _frames(821)
    substituted = _offset_pixels(approved, 60)

    with pytest.raises(
        VisualBindingError,
        match="mean_absolute_error=.*allowed<=",
    ):
        _measure(tmp_path, monkeypatch, approved, substituted)


def test_visual_binding_rejects_invalid_absolute_error_threshold(tmp_path, monkeypatch):
    shot = tmp_path / "shot-1.mp4"
    final = tmp_path / "film.mp4"
    shot.write_bytes(b"approved-shot")
    final.write_bytes(b"final-film")

    with pytest.raises(
        VisualBindingError,
        match="maximum mean absolute error must be in the interval",
    ):
        measure_visual_binding(
            [shot],
            final,
            maximum_mean_absolute_error=float("inf"),
        )
