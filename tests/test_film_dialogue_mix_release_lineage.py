from __future__ import annotations

from pathlib import Path

import pytest

import cineos.film.benchmark_film_assembly as assembly
from cineos.film.exceptions import AssemblyError


def _sha(index: int) -> str:
    return f"{index:064x}"


def _mix_evidence(
    output_path: Path,
    *,
    dialogue_shots: tuple[str, ...] = ("shot-2", "shot-4"),
) -> dict:
    inputs = [
        {
            "kind": "dialogue",
            "shot_id": shot_id,
            "sha256": _sha(100 + index),
        }
        for index, shot_id in enumerate(dialogue_shots)
    ]
    inputs.append({"kind": "music", "shot_id": None, "sha256": _sha(999)})
    return {
        "output_path": str(output_path),
        "inputs": inputs,
    }


def test_dialogue_mix_release_lineage_requires_production_mix_evidence(
    tmp_path,
) -> None:
    audio_path = tmp_path / "final.wav"

    with pytest.raises(AssemblyError, match="requires production audio mix evidence"):
        assembly._validate_dialogue_mix_lineage(
            dialogue_ids=("shot-2",),
            audio_path=audio_path,
            audio_sha256=_sha(1),
            audio_mix_evidence=None,
        )


def test_dialogue_mix_release_lineage_binds_exact_final_mix_hash(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "final.wav"
    evidence = _mix_evidence(audio_path)
    monkeypatch.setattr(
        assembly,
        "validate_production_audio_mix_evidence",
        lambda _evidence: _sha(2),
    )

    with pytest.raises(
        AssemblyError, match="does not match the final released audio hash"
    ):
        assembly._validate_dialogue_mix_lineage(
            dialogue_ids=("shot-2", "shot-4"),
            audio_path=audio_path,
            audio_sha256=_sha(1),
            audio_mix_evidence=evidence,
        )


def test_dialogue_mix_release_lineage_rejects_substituted_dialogue_scope(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "final.wav"
    evidence = _mix_evidence(audio_path, dialogue_shots=("shot-2", "shot-5"))
    monkeypatch.setattr(
        assembly,
        "validate_production_audio_mix_evidence",
        lambda _evidence: _sha(1),
    )

    with pytest.raises(AssemblyError, match="dialogue-shot lineage does not match"):
        assembly._validate_dialogue_mix_lineage(
            dialogue_ids=("shot-2", "shot-4"),
            audio_path=audio_path,
            audio_sha256=_sha(1),
            audio_mix_evidence=evidence,
        )


def test_dialogue_mix_release_lineage_accepts_exact_dialogue_scope(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "final.wav"
    evidence = _mix_evidence(audio_path)
    monkeypatch.setattr(
        assembly,
        "validate_production_audio_mix_evidence",
        lambda _evidence: _sha(1),
    )

    assembly._validate_dialogue_mix_lineage(
        dialogue_ids=("shot-2", "shot-4"),
        audio_path=audio_path,
        audio_sha256=_sha(1),
        audio_mix_evidence=evidence,
    )


def test_dialogue_mix_lineage_rejects_other_release_path(tmp_path, monkeypatch) -> None:
    mix_path = tmp_path / "mix.wav"
    release_path = tmp_path / "release.wav"
    evidence = _mix_evidence(mix_path)
    monkeypatch.setattr(
        assembly,
        "validate_production_audio_mix_evidence",
        lambda _evidence: _sha(1),
    )

    with pytest.raises(AssemblyError, match="output path does not match"):
        assembly._validate_dialogue_mix_lineage(
            dialogue_ids=("shot-2", "shot-4"),
            audio_path=release_path,
            audio_sha256=_sha(1),
            audio_mix_evidence=evidence,
        )
