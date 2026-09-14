from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import pytest

from cineos.audio.production_mix_evidence import (
    ProductionAudioMixEvidenceError,
    ProductionMixInput,
    mix_production_audio,
    validate_production_audio_mix_evidence,
)


def _wav(path: Path, value: int) -> Path:
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, 48_000, 4_800, "NONE", "not compressed"))
        target.writeframes(int(value).to_bytes(2, "little", signed=True) * 4_800)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_mix_binds_exact_dialogue_sources_and_output(tmp_path) -> None:
    dialogue = _wav(tmp_path / "dialogue.wav", 800)
    music = _wav(tmp_path / "music.wav", 100)

    evidence = mix_production_audio(
        [
            ProductionMixInput(
                dialogue,
                _sha(dialogue),
                kind="dialogue",
                shot_id="shot-2",
            ),
            ProductionMixInput(music, _sha(music), kind="music", gain=0.2),
        ],
        tmp_path / "mix.wav",
        duration=0.1,
    )

    assert validate_production_audio_mix_evidence(evidence) == evidence["output_sha256"]
    assert evidence["inputs"][0]["shot_id"] == "shot-2"
    assert evidence["inputs"][0]["sha256"] == _sha(dialogue)


def test_production_mix_rejects_substituted_source_artifact(tmp_path) -> None:
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
    _wav(dialogue, 200)

    with pytest.raises(
        ProductionAudioMixEvidenceError,
        match="input 0 does not match evidence",
    ):
        validate_production_audio_mix_evidence(evidence)


def test_production_mix_requires_shot_identity_for_dialogue(tmp_path) -> None:
    dialogue = _wav(tmp_path / "dialogue.wav", 800)

    with pytest.raises(
        ProductionAudioMixEvidenceError,
        match="dialogue mix input 0 requires shot_id",
    ):
        mix_production_audio(
            [ProductionMixInput(dialogue, _sha(dialogue), kind="dialogue")],
            tmp_path / "mix.wav",
        )
