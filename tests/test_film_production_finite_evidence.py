from pathlib import Path

import pytest

import cineos.film.production_assembly as production
from cineos.film.exceptions import AssemblyError


def _video_probe(*, duration: float = 5.0) -> dict[str, object]:
    return {
        "duration_seconds": duration,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "audio_stream_count": 0,
        "video_codecs": ["h264"],
        "audio_codecs": [],
        "video_dimensions": [{"width": 1280, "height": 720}],
        "audio_streams": [],
    }


def test_rejects_nonfinite_source_shot_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production,
        "probe_media",
        lambda _path: _video_probe(duration=float("nan")),
    )

    with pytest.raises(AssemblyError, match="finite positive duration"):
        production._validate_bound_shot_media("shot-0", Path("unused.mp4"))


def test_rejects_nonfinite_requested_duration_before_artifact_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production,
        "assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )

    with pytest.raises(AssemblyError, match="durations must all be positive"):
        production.assemble_production_film(
            [{} for _ in range(5)],
            "unused.mp4",
            durations=[1.0, 1.0, float("nan"), 1.0, 1.0],
        )


def test_rejects_nonfinite_final_video_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production,
        "probe_media",
        lambda _path: _video_probe(duration=float("inf")),
    )

    with pytest.raises(AssemblyError, match="finite positive duration"):
        production._validate_final_media(
            Path("unused.mp4"),
            audio_required=False,
            expected_duration=5.0,
            expected_width=1280,
            expected_height=720,
            expected_frame_rate=None,
        )


def test_rejects_nonfinite_final_audio_stream_duration() -> None:
    media = {
        "audio_streams": [
            {
                "codec_name": "aac",
                "sample_rate_hz": 48_000,
                "channels": 2,
                "duration_seconds": float("nan"),
            }
        ]
    }

    with pytest.raises(AssemblyError, match="finite positive stream duration"):
        production._validate_audio_stream(media, expected_duration=5.0)


@pytest.mark.parametrize(
    ("mean_volume", "max_volume", "message"),
    [
        (float("nan"), -3.0, "mean-volume evidence must be finite"),
        (-24.0, float("inf"), "max-volume evidence must be finite"),
    ],
)
def test_rejects_nonfinite_decoded_audio_signal(
    monkeypatch: pytest.MonkeyPatch,
    mean_volume: float,
    max_volume: float,
    message: str,
) -> None:
    monkeypatch.setattr(
        production,
        "probe_audio_signal",
        lambda _path: {
            "mean_volume_db": mean_volume,
            "max_volume_db": max_volume,
        },
    )

    with pytest.raises(AssemblyError, match=message):
        production._validate_audio_signal(Path("unused.mp4"))
