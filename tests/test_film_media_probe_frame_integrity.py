import json
import subprocess

import pytest

import cineos.film.media_probe as media_probe
from cineos.film.media_probe import MediaProbeError


def _payload(*, frame_count: str, duration: str = "5.000000") -> dict[str, object]:
    return {
        "format": {
            "duration": duration,
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "duration": duration,
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "24/1",
                "nb_read_frames": frame_count,
            }
        ],
    }


def _install_probe(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    monkeypatch.setattr(media_probe.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        media_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )


def test_probe_rejects_decoded_frame_count_that_conflicts_with_stream_timeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "sparse.mp4"
    source.write_bytes(b"not-empty")
    _install_probe(monkeypatch, _payload(frame_count="3"))

    with pytest.raises(MediaProbeError, match="frame count conflicts"):
        media_probe.probe_media(source)


def test_probe_accepts_decoded_frame_count_matching_stream_timeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "healthy.mp4"
    source.write_bytes(b"not-empty")
    _install_probe(monkeypatch, _payload(frame_count="120"))

    evidence = media_probe.probe_media(source)

    assert evidence["video_frame_counts"] == [120]
    assert evidence["duration_seconds"] == pytest.approx(5.0)


def test_probe_tolerates_small_frame_count_rounding_difference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "rounded.mp4"
    source.write_bytes(b"not-empty")
    _install_probe(monkeypatch, _payload(frame_count="118"))

    evidence = media_probe.probe_media(source)

    assert evidence["video_frame_counts"] == [118]


def test_probe_rejects_nonfinite_duration_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "nonfinite.mp4"
    source.write_bytes(b"not-empty")
    _install_probe(monkeypatch, _payload(frame_count="120", duration="inf"))

    with pytest.raises(MediaProbeError, match="positive media duration"):
        media_probe.probe_media(source)
