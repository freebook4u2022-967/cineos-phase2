from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.audio.production_mix_evidence import (
    ProductionMixInput,
    mix_production_audio,
)
from cineos.film.connected_production_evidence import (
    ConnectedProductionFilmEvidenceError,
    _validate_dialogue_mix_binding,
)


def _wav(path: Path, value: int) -> Path:
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, 48_000, 4_800, "NONE", "not compressed"))
        target.writeframes(int(value).to_bytes(2, "little", signed=True) * 4_800)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    dialogue = _wav(tmp_path / "dialogue.wav", 800)
    evidence = mix_production_audio(
        [
            ProductionMixInput(
                dialogue,
                _sha(dialogue),
                kind="dialogue",
                shot_id="shot-2",
            )
        ],
        tmp_path / "mix.wav",
        duration=0.1,
    )
    benchmark = SimpleNamespace(dialogue_scope_declared=True)
    assembly = {"audio": {"sha256": evidence["output_sha256"]}}
    return benchmark, assembly, dialogue, evidence


def test_gpu_declared_dialogue_requires_final_mix_lineage(tmp_path) -> None:
    benchmark, assembly, dialogue, _ = _fixture(tmp_path)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="requires production audio mix evidence",
    ):
        _validate_dialogue_mix_binding(
            benchmark,
            assembly,
            dialogue_shot_ids=["shot-2"],
            dialogue_audio_sha256_by_shot={"shot-2": _sha(dialogue)},
            audio_mix_evidence=None,
        )


def test_gpu_declared_dialogue_binds_same_lipsync_audio_to_final_mix(tmp_path) -> None:
    benchmark, assembly, dialogue, evidence = _fixture(tmp_path)

    evidence_sha = _validate_dialogue_mix_binding(
        benchmark,
        assembly,
        dialogue_shot_ids=["shot-2"],
        dialogue_audio_sha256_by_shot={"shot-2": _sha(dialogue)},
        audio_mix_evidence=evidence,
    )

    assert evidence_sha == evidence["evidence_sha256"]


def test_gpu_declared_dialogue_rejects_substituted_final_mix_source(tmp_path) -> None:
    benchmark, assembly, _dialogue, evidence = _fixture(tmp_path)
    approved_other = _wav(tmp_path / "approved-other.wav", 300)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="substitutes dialogue audio",
    ):
        _validate_dialogue_mix_binding(
            benchmark,
            assembly,
            dialogue_shot_ids=["shot-2"],
            dialogue_audio_sha256_by_shot={"shot-2": _sha(approved_other)},
            audio_mix_evidence=evidence,
        )


def test_gpu_declared_dialogue_rejects_unrelated_assembly_audio(tmp_path) -> None:
    benchmark, assembly, dialogue, evidence = _fixture(tmp_path)
    assembly["audio"]["sha256"] = f"{42:064x}"

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="does not match approved assembly audio",
    ):
        _validate_dialogue_mix_binding(
            benchmark,
            assembly,
            dialogue_shot_ids=["shot-2"],
            dialogue_audio_sha256_by_shot={"shot-2": _sha(dialogue)},
            audio_mix_evidence=evidence,
        )
