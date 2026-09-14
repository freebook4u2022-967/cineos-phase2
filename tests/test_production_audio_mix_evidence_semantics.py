from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cineos.audio.production_mix_evidence import (
    PRODUCTION_AUDIO_MIX_EVIDENCE_SCHEMA,
    ProductionAudioMixEvidenceError,
    validate_production_audio_mix_evidence,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign(evidence: dict) -> dict:
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    payload = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    evidence["evidence_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return evidence


def _evidence(tmp_path: Path) -> dict:
    source = tmp_path / "dialogue.wav"
    output = tmp_path / "mix.wav"
    source.write_bytes(b"dialogue-source")
    output.write_bytes(b"mixed-output")
    return _sign(
        {
            "schema": PRODUCTION_AUDIO_MIX_EVIDENCE_SCHEMA,
            "sample_rate_hz": 48000,
            "channel_layout": "stereo",
            "duration_seconds": 2.0,
            "inputs": [
                {
                    "index": 0,
                    "path": str(source.resolve()),
                    "sha256": _sha(source),
                    "start_time_seconds": 0.25,
                    "gain": 1.0,
                    "kind": "dialogue",
                    "shot_id": "shot-1",
                    "fade_in_seconds": 0.0,
                    "fade_out_seconds": 0.0,
                    "pan": 0.0,
                }
            ],
            "output_path": str(output.resolve()),
            "output_sha256": _sha(output),
        }
    )


def test_valid_semantic_mix_evidence_is_accepted(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    assert validate_production_audio_mix_evidence(evidence) == evidence["output_sha256"]


def test_resigned_negative_timeline_control_fails_closed(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["inputs"][0]["start_time_seconds"] = -0.25
    _sign(evidence)

    with pytest.raises(ProductionAudioMixEvidenceError, match="negative controls"):
        validate_production_audio_mix_evidence(evidence)


def test_resigned_duplicate_dialogue_shot_fails_closed(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    duplicate = dict(evidence["inputs"][0])
    duplicate["index"] = 1
    evidence["inputs"].append(duplicate)
    _sign(evidence)

    with pytest.raises(ProductionAudioMixEvidenceError, match="duplicate dialogue"):
        validate_production_audio_mix_evidence(evidence)


def test_resigned_non_dialogue_input_cannot_claim_shot_id(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["inputs"][0]["kind"] = "music"
    _sign(evidence)

    with pytest.raises(ProductionAudioMixEvidenceError, match="cannot claim shot_id"):
        validate_production_audio_mix_evidence(evidence)


def test_resigned_invalid_duration_fails_closed(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["duration_seconds"] = float("nan")
    _sign(evidence)

    with pytest.raises(ProductionAudioMixEvidenceError, match="duration"):
        validate_production_audio_mix_evidence(evidence)
