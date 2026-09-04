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


def _probe(
    *,
    duration=5.0,
    audio=0,
    video_codec="h264",
    audio_codec="aac",
    audio_duration=None,
    sample_rate=48_000,
    channels=2,
):
    stream_duration = duration if audio_duration is None else audio_duration
    return {
        "duration_seconds": duration,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "audio_stream_count": audio,
        "video_codecs": [video_codec],
        "audio_codecs": [audio_codec] if audio else [],
        "video_dimensions": [{"width": 1280, "height": 720}],
        "audio_streams": (
            [
                {
                    "codec_name": audio_codec,
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "duration_seconds": stream_duration,
                }
            ]
            if audio
            else []
        ),
    }


def _audible(_path):
    return {"mean_volume_db": -24.0, "max_volume_db": -3.0}


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
    assert manifest["schema"] == "cineos-production-film-evidence/0.9"
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


def test_rejects_duplicate_rendered_artifact_even_with_distinct_shot_ids(
    tmp_path, monkeypatch
):
    records = _records(tmp_path)
    records[4]["output_path"] = records[3]["output_path"]
    records[4]["output_sha256"] = records[3]["output_sha256"]
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )

    with pytest.raises(AssemblyError, match="reuses a rendered artifact"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_duplicate_qc_evidence_even_with_distinct_artifacts(
    tmp_path, monkeypatch
):
    records = _records(tmp_path)
    records[4]["evidence_sha256"] = records[3]["evidence_sha256"]
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )

    with pytest.raises(AssemblyError, match="reuses QC evidence"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_invalid_duration_contract(tmp_path, monkeypatch):
    records = _records(tmp_path)
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )

    with pytest.raises(AssemblyError, match="duration count"):
        assemble_production_film(records, tmp_path / "final.mp4", durations=[1.0])
    with pytest.raises(AssemblyError, match="must all be positive"):
        assemble_production_film(
            records, tmp_path / "final.mp4", durations=[1, 1, 0, 1, 1]
        )


def test_binds_optional_audio_artifact(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-audio")
    output = tmp_path / "final.mp4"

    def fake_assemble(shots, destination, *, durations=None, audio_path=None, **_kwargs):
        assert len(shots) == 5
        assert audio_path == audio.resolve()
        Path(destination).write_bytes(b"film-with-audio")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(audio=1),
    )
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_audio_signal", _audible
    )
    manifest = assemble_production_film(
        records,
        output,
        audio_path=audio,
        audio_sha256=_sha(audio),
    )

    assert manifest["audio"]["sha256"] == _sha(audio)
    assert manifest["final_media"]["audio_stream_count"] == 1
    assert manifest["final_media"]["production_audio_stream"] == {
        "codec_name": "aac",
        "sample_rate_hz": 48_000,
        "channels": 2,
        "duration_seconds": 5.0,
        "duration_delta_seconds": 0.0,
    }
    assert manifest["final_media"]["production_audio_signal"]["mean_volume_db"] == -24.0


def test_rejects_unbound_or_changed_audio_artifact(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-audio")
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("assembly must not run"),
    )

    with pytest.raises(AssemblyError, match="explicit SHA-256"):
        assemble_production_film(records, tmp_path / "final.mp4", audio_path=audio)

    with pytest.raises(AssemblyError, match="without an audio artifact"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_sha256="0" * 64,
        )

    with pytest.raises(AssemblyError, match="hash does not match"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_path=audio,
            audio_sha256="0" * 64,
        )


def test_rejects_final_container_or_codec_drift(tmp_path, monkeypatch):
    records = _records(tmp_path)

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"bad-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(video_codec="vp9"),
    )

    with pytest.raises(AssemblyError, match="H.264"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_missing_or_wrong_final_audio_stream(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-audio")

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media", lambda _path: _probe(audio=0)
    )
    with pytest.raises(AssemblyError, match="exactly one approved audio"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_path=audio,
            audio_sha256=_sha(audio),
        )

    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(audio=1, audio_codec="mp3"),
    )
    with pytest.raises(AssemblyError, match="must be AAC"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_path=audio,
            audio_sha256=_sha(audio),
        )


def test_rejects_audio_when_no_approved_audio_was_supplied(tmp_path, monkeypatch):
    records = _records(tmp_path)

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media", lambda _path: _probe(audio=1)
    )
    with pytest.raises(AssemblyError, match="no approved audio artifact"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_wrong_audio_sample_rate(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-audio")

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(audio=1, sample_rate=44_100),
    )
    with pytest.raises(AssemblyError, match="48000 Hz"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_path=audio,
            audio_sha256=_sha(audio),
        )


def test_rejects_truncated_audio_stream(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-audio")

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _probe(audio=1, audio_duration=1.0),
    )
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_audio_signal", _audible
    )
    with pytest.raises(AssemblyError, match="audio duration deviates"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_path=audio,
            audio_sha256=_sha(audio),
        )


def test_rejects_silent_approved_audio(tmp_path, monkeypatch):
    records = _records(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-audio")

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media", lambda _path: _probe(audio=1)
    )
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_audio_signal",
        lambda _path: {"mean_volume_db": -120.0, "max_volume_db": -120.0},
    )
    with pytest.raises(AssemblyError, match="effectively silent"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            audio_path=audio,
            audio_sha256=_sha(audio),
        )
