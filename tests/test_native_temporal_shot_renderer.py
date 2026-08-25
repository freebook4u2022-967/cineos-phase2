from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video.shot_renderer import (
    AnalyticLatentRGBDecoder,
    CINEOSNativeTemporalShotRenderer,
    NativeShotRenderError,
    _latent_to_rgb,
)
from cineos.native_video.temporal_model import NativeTemporalModel


@dataclass(frozen=True, slots=True)
class Planned:
    shot_id: str = "shot-001"
    scene_id: str = "scene-001"
    duration: float = 0.5
    payload: dict[str, object] = field(
        default_factory=lambda: {
            "approved_reference_ids": ["ref-front", "ref-side"],
            "cinedna_ids": ["hero-v1"],
            "character_ids": ["hero"],
            "prompt": "A restrained close-up with subtle breathing motion.",
            "location_id": "room-a",
            "continuity_key": "scene-001-night",
        }
    )


@dataclass(frozen=True, slots=True)
class BrokenDecoder:
    decoder_id: str = "broken-test-decoder"

    def decode(self, latent: Tensor, *, width: int, height: int) -> bytes:
        del latent, width, height
        return b"too-short"


def test_latent_decoder_produces_real_rgb_payload() -> None:
    latent = Tensor((0.1, -0.2, 0.3, -0.4), (4,), "cpu")
    rgb = _latent_to_rgb(latent, 8, 4)

    assert len(rgb) == 8 * 4 * 3
    assert len(set(rgb)) > 8


def test_default_decoder_has_explicit_native_provenance() -> None:
    decoder = AnalyticLatentRGBDecoder()

    assert decoder.decoder_id == "cineos-analytic-latent-rgb/1"


def test_renderer_rejects_decoder_without_provenance() -> None:
    class DecoderWithoutId:
        def decode(self, latent: Tensor, *, width: int, height: int) -> bytes:
            del latent
            return bytes(width * height * 3)

    with pytest.raises(ValueError, match="decoder_id"):
        CINEOSNativeTemporalShotRenderer(decoder=DecoderWithoutId())  # type: ignore[arg-type]


def test_renderer_fails_closed_without_encoder(tmp_path: Path) -> None:
    model = NativeTemporalModel.initialized()
    renderer = CINEOSNativeTemporalShotRenderer(
        width=16,
        height=16,
        fps=2,
        ffmpeg_binary="cineos-ffmpeg-does-not-exist",
    )
    state = model.initial_state("shot-001")

    with pytest.raises(NativeShotRenderError, match="not available"):
        renderer.render(Planned(), tmp_path / "shot.mp4", temporal_state=state)

    assert state.last_frame_index == -1
    assert state.last_latent is None


def test_renderer_rolls_back_temporal_state_when_decoder_fails(tmp_path: Path) -> None:
    renderer = CINEOSNativeTemporalShotRenderer(
        decoder=BrokenDecoder(),
        width=16,
        height=16,
        fps=2,
    )
    state = renderer.runtime.model.initial_state("shot-001")
    original = state.snapshot()

    with pytest.raises(NativeShotRenderError, match="broken-test-decoder"):
        renderer.render(Planned(), tmp_path / "shot.mp4", temporal_state=state)

    assert state.snapshot() == original
    assert not (tmp_path / "shot.mp4").exists()


def test_renderer_rolls_back_temporal_state_when_encoding_fails(tmp_path: Path) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg-fail"
    fake_ffmpeg.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
    fake_ffmpeg.chmod(0o755)

    renderer = CINEOSNativeTemporalShotRenderer(
        width=16,
        height=16,
        fps=2,
        ffmpeg_binary=str(fake_ffmpeg),
    )
    state = renderer.runtime.model.initial_state("shot-001")
    original = state.snapshot()
    target = tmp_path / "shot.mp4"

    with pytest.raises(NativeShotRenderError, match="failed to encode"):
        renderer.render(Planned(duration=1.0), target, temporal_state=state)

    assert state.snapshot() == original
    assert not target.exists()


def test_renderer_preserves_existing_artifact_when_encoding_fails(
    tmp_path: Path,
) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg-fail"
    fake_ffmpeg.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    fake_ffmpeg.chmod(0o755)

    renderer = CINEOSNativeTemporalShotRenderer(
        width=16,
        height=16,
        fps=2,
        ffmpeg_binary=str(fake_ffmpeg),
    )
    state = renderer.runtime.model.initial_state("shot-001")
    original = state.snapshot()
    target = tmp_path / "shot.mp4"
    target.write_bytes(b"durable-prior-shot")

    with pytest.raises(NativeShotRenderError, match="failed to encode"):
        renderer.render(Planned(duration=1.0), target, temporal_state=state)

    assert state.snapshot() == original
    assert target.read_bytes() == b"durable-prior-shot"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_renderer_generates_mp4_and_advances_only_native_temporal_state(
    tmp_path: Path,
) -> None:
    renderer = CINEOSNativeTemporalShotRenderer(width=32, height=18, fps=4)
    state = renderer.runtime.model.initial_state("shot-001")
    target = tmp_path / "shot.mp4"

    result = renderer.render(Planned(duration=0.5), target, temporal_state=state)

    assert result == target
    assert target.is_file()
    assert target.stat().st_size > 0
    assert state.last_frame_index == 1
    assert state.last_latent is not None
    assert state.metadata["frames_generated"] == 2
    assert state.metadata["native_renderer"] == "cineos-temporal-pixel/0.3"
    assert state.metadata["native_decoder"] == "cineos-analytic-latent-rgb/1"
    assert state.metadata["native_frame_count"] == 2


def test_renderer_rejects_state_for_another_shot(tmp_path: Path) -> None:
    renderer = CINEOSNativeTemporalShotRenderer(width=16, height=16, fps=2)
    state = renderer.runtime.model.initial_state("other-shot")

    with pytest.raises(ValueError, match="must belong"):
        renderer.render(Planned(), tmp_path / "shot.mp4", temporal_state=state)
