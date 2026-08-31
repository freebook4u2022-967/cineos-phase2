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


def test_assembles_only_bound_qc_approved_gpu_artifacts(tmp_path, monkeypatch):
    records = _records(tmp_path)
    output = tmp_path / "final.mp4"

    def fake_assemble(shots, destination, *, durations=None, crossfade=0.0):
        assert [Path(item).name for item in shots] == [
            f"shot-{i}.mp4" for i in range(5)
        ]
        assert durations == [1.0] * 5
        Path(destination).write_bytes(b"assembled-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    manifest = assemble_production_film(records, output, durations=[1.0] * 5)

    assert manifest["shot_count"] == 5
    assert manifest["final_mp4_sha256"] == _sha(output)
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


def test_binds_audio_hash_and_rejects_audio_swap(tmp_path, monkeypatch):
    records = _records(tmp_path)
    output = tmp_path / "final.mp4"
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"approved-mix")

    def fake_assemble(_shots, destination, *, durations=None, crossfade=0.0):
        Path(destination).write_bytes(b"assembled-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    manifest = assemble_production_film(
        records,
        output,
        audio_path=audio,
        audio_sha256=_sha(audio),
    )
    assert manifest["audio"]["sha256"] == _sha(audio)

    audio.write_bytes(b"swapped-mix")
    with pytest.raises(AssemblyError, match="audio artifact hash does not match"):
        assemble_production_film(
            records,
            tmp_path / "final-2.mp4",
            audio_path=audio,
            audio_sha256=manifest["audio"]["sha256"],
        )
