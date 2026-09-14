from __future__ import annotations

import pytest

from cineos.film import assembly
from cineos.film.exceptions import AssemblyError


def _media(*, audio: bool = True) -> dict[str, object]:
    return {
        "video_codecs": ["h264"],
        "video_dimensions": [{"width": 1920, "height": 1080}],
        "video_frame_rates": ["24/1"],
        "audio_streams": (
            [
                {
                    "codec_name": "aac",
                    "sample_rate_hz": 48_000,
                    "channels": 2,
                    "duration_seconds": 10.0,
                }
            ]
            if audio
            else []
        ),
    }


def test_final_delivery_contract_accepts_h264_with_48khz_aac() -> None:
    assembly._validate_delivery_contract(_media(), expect_audio=True)


def test_video_only_delivery_does_not_require_audio_evidence() -> None:
    assembly._validate_delivery_contract(_media(audio=False), expect_audio=False)


@pytest.mark.parametrize("codec", ["", "hevc", "vp9", None])
def test_final_delivery_contract_rejects_unexpected_video_codec(codec) -> None:
    media = _media()
    media["video_codecs"] = [codec]
    with pytest.raises(AssemblyError, match="video codec"):
        assembly._validate_delivery_contract(media, expect_audio=True)


@pytest.mark.parametrize(
    "dimensions",
    [
        [],
        [{}],
        [{"width": 0, "height": 1080}],
        [{"width": 1920, "height": 0}],
        [{"width": "bad", "height": 1080}],
    ],
)
def test_final_delivery_contract_rejects_invalid_dimensions(dimensions) -> None:
    media = _media()
    media["video_dimensions"] = dimensions
    with pytest.raises(AssemblyError, match="dimension"):
        assembly._validate_delivery_contract(media, expect_audio=True)


@pytest.mark.parametrize("frame_rate", ["", "N/A", "0/0", "0/1", "bad"])
def test_final_delivery_contract_rejects_invalid_frame_rate(frame_rate: str) -> None:
    media = _media()
    media["video_frame_rates"] = [frame_rate]
    with pytest.raises(AssemblyError, match="frame-rate"):
        assembly._validate_delivery_contract(media, expect_audio=True)


@pytest.mark.parametrize("codec", ["", "mp3", "pcm_s16le", None])
def test_final_delivery_contract_rejects_unexpected_audio_codec(codec) -> None:
    media = _media()
    media["audio_streams"][0]["codec_name"] = codec
    with pytest.raises(AssemblyError, match="audio codec"):
        assembly._validate_delivery_contract(media, expect_audio=True)


@pytest.mark.parametrize("sample_rate", [0, 44_100, 96_000, None])
def test_final_delivery_contract_requires_48khz_audio(sample_rate) -> None:
    media = _media()
    media["audio_streams"][0]["sample_rate_hz"] = sample_rate
    with pytest.raises(AssemblyError, match="48 kHz"):
        assembly._validate_delivery_contract(media, expect_audio=True)


@pytest.mark.parametrize("channels", [0, -1, None])
def test_final_delivery_contract_rejects_invalid_audio_channels(channels) -> None:
    media = _media()
    media["audio_streams"][0]["channels"] = channels
    with pytest.raises(AssemblyError, match="audio channel"):
        assembly._validate_delivery_contract(media, expect_audio=True)
