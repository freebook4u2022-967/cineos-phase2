import hashlib
from pathlib import Path

import pytest

from cineos.film.exceptions import AssemblyError
from cineos.film.production_assembly import assemble_production_film


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(tmp_path: Path, count: int = 5):
    records = []
    for index in range(count):
        path = tmp_path / f"shot-{index}.mp4"
        path.write_bytes(f"shot-{index}".encode())
        records.append(
            {
                "shot_id": f"shot-{index}",
                "accepted": True,
                "decision": "accept",
                "production_gpu_evidence": True,
                "output_path": str(path),
                "output_sha256": _sha(path),
                "evidence_sha256": hashlib.sha256(
                    f"evidence-{index}".encode()
                ).hexdigest(),
            }
        )
    return records


def _probe(*, duration=5.0, audio=0, video_codec="h264", audio_codec="aac"):
    return {
        "duration_seconds": duration,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "audio_stream_count": audio,
        "video_codecs": [video_codec],
        "audio_codecs": [audio_codec] if audio else [],
        "video_dimensions": [{"width": 1280, "height": 720}],
    }


def test_assembles_only_bound_qc_approved_gpu_artifacts(tmp_path, monkeypatch):
    records = _records(tmp_path)
    output = tmp_path / "final.mp4"

    def fake_assemble(
        shots,
        destination,
        *,
        durations=None,
        crossfade=0.0,
        audio_path=None,
    ):
        assert [Path(item).name for item in shots] == [
            f"shot-{i}.mp4" for i in range(5)
        ]
        assert durations == [1.0] * 5
        assert audio_path is None
        Path(destination).write_bytes(b"assembled-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media", lambda _path: _probe()
    )
    manifest = assemble_production_film(records, output, durations=[1.0] * 5)

    assert manifest["shot_count"] == 5
    assert manifest["final_mp4_sha256"] == _sha(output)
    assert manifest["final_media"]["video_stream_count"] == 1
    assert manifest["schema"] == "cineos-production-film-evidence/0.3"
    assert len(manifest["manifest_sha256"]) == 64
    assert (tmp_path / "final.production.json").is_file()


def test_rejects_post_qc_artifact_swap(tmp_path, monkeypatch):
    records = _records(tmp_path)
    Path(records[2]["output_path"]).write_bytes(b"swapped-after-qc")
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )

    with pytest.raises(AssemblyError, match="artifact hash does not match QC evidence"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_nonproduction_or_rejected_shot(tmp_path, monkeypatch):
    records = _records(tmp_path)
    records[1]["production_gpu_evidence"] = False
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )

    with pytest.raises(AssemblyError, match="lacks real GPU evidence"):
        assemble_production_film(records, tmp_path / "final.mp4")

    records = _records(tmp_path)
    records[1]["accepted"] = False
    records[1]["decision"] = "reject"
    with pytest.raises(AssemblyError, match="not QC-approved"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_duplicate_or_wrong_connected_shot_count(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )
    with pytest.raises(AssemblyError, match="requires 5 to 10 shots"):
        assemble_production_film(_records(tmp_path, 4), tmp_path / "final.mp4")

    records = _records(tmp_path)
    records[4]["shot_id"] = records[3]["shot_id"]
    with pytest.raises(AssemblyError, match="duplicate production shot_id"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_binds_and_muxes_audio_hash_and_rejects_audio_swap(tmp_path, monkeypatch):
    records = _records(tmp_path)
    output = tmp_path / "final.mp4"
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-mix")
    captured = {}

    def fake_assemble(
        _shots,
        destination,
        *,
        durations=None,
        crossfade=0.0,
        audio_path=None,
    ):
        captured["audio_path"] = Path(audio_path) if audio_path is not None else None
        Path(destination).write_bytes(b"assembled-film-with-audio")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(audio=1),
    )
    manifest = assemble_production_film(
        records,
        output,
        audio_path=audio,
        audio_sha256=_sha(audio),
    )
    assert captured["audio_path"] == audio.resolve()
    assert manifest["audio"]["sha256"] == _sha(audio)
    assert manifest["final_media"]["audio_stream_count"] == 1

    audio.write_bytes(b"swapped-mix")
    with pytest.raises(AssemblyError, match="audio artifact hash does not match"):
        assemble_production_film(
            records,
            tmp_path / "final-2.mp4",
            audio_path=audio,
            audio_sha256=manifest["audio"]["sha256"],
        )


def test_rejects_silent_output_when_approved_audio_was_required(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-mix")

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"silent-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media", lambda _path: _probe(audio=0)
    )

    with pytest.raises(AssemblyError, match="exactly one approved audio stream"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_path=audio,
            audio_sha256=_sha(audio),
        )


def test_rejects_truncated_final_timeline(tmp_path, monkeypatch):
    records = _records(tmp_path)

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"truncated-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(duration=3.0),
    )

    with pytest.raises(
        AssemblyError, match="duration deviates from the approved visual timeline"
    ):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            durations=[1.0] * 5,
        )


def test_rejects_nonpositive_shot_duration_before_assembly(tmp_path, monkeypatch):
    records = _records(tmp_path)
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )
    with pytest.raises(AssemblyError, match="durations must all be positive"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            durations=[1.0, 1.0, 0.0, 1.0, 1.0],
        )


def test_rejects_unapproved_or_wrong_final_media_profile(tmp_path, monkeypatch):
    records = _records(tmp_path)

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"assembled-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)

    bad_profiles = [
        ({**_probe(), "format_name": "matroska,webm"}, "not an MP4 container"),
        ({**_probe(), "video_stream_count": 2, "video_codecs": ["h264", "h264"]}, "exactly one video stream"),
        (_probe(video_codec="hevc"), "exactly one H.264 video stream"),
        ({**_probe(), "video_dimensions": [{"width": 1279, "height": 720}]}, "invalid H.264/yuv420p video dimensions"),
    ]
    for profile, message in bad_profiles:
        monkeypatch.setattr(
            "cineos.film.production_assembly.probe_media", lambda _path, p=profile: p
        )
        with pytest.raises(AssemblyError, match=message):
            assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_unapproved_audio_or_unexpected_audio_stream(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-mix")

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"assembled-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(audio=1, audio_codec="opus"),
    )
    with pytest.raises(AssemblyError, match="approved audio stream must be AAC"):
        assemble_production_film(
            records,
            tmp_path / "final-with-audio.mp4",
            audio_path=audio,
            audio_sha256=_sha(audio),
        )

    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(audio=1),
    )
    with pytest.raises(AssemblyError, match="no approved audio artifact was supplied"):
        assemble_production_film(records, tmp_path / "final-unexpected-audio.mp4")
