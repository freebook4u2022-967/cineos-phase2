"""Evidence-bound production audio mixing.

This layer wraps the portable mixer without claiming synthesis or scoring capability.
It proves which exact source artifacts were consumed to create the approved final mix.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mixer import Mixer, MixInput

PRODUCTION_AUDIO_MIX_EVIDENCE_SCHEMA = "cineos-production-audio-mix-evidence/0.1"


class ProductionAudioMixEvidenceError(RuntimeError):
    """Raised when production mix lineage is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_sha256(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64:
        raise ProductionAudioMixEvidenceError(f"{field} must be a SHA-256 hex digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ProductionAudioMixEvidenceError(
            f"{field} must be a SHA-256 hex digest"
        ) from exc
    return normalized


def _validated_duration(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProductionAudioMixEvidenceError(
            "production audio mix duration must be finite and positive"
        )
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ProductionAudioMixEvidenceError(
            "production audio mix duration must be finite and positive"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ProductionAudioMixEvidenceError(
            "production audio mix duration must be finite and positive"
        )
    return duration


def _validated_controls(item: Mapping[str, Any], *, index: int) -> tuple[float, ...]:
    fields = (
        "start_time_seconds",
        "gain",
        "fade_in_seconds",
        "fade_out_seconds",
        "pan",
    )
    try:
        values = tuple(float(item.get(field)) for field in fields)
    except (TypeError, ValueError) as exc:
        raise ProductionAudioMixEvidenceError(
            f"production audio mix input {index} has invalid numeric controls"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise ProductionAudioMixEvidenceError(
            f"production audio mix input {index} has non-finite controls"
        )
    start_time, gain, fade_in, fade_out, _pan = values
    if start_time < 0 or gain < 0 or fade_in < 0 or fade_out < 0:
        raise ProductionAudioMixEvidenceError(
            f"production audio mix input {index} has invalid negative controls"
        )
    return values


@dataclass(frozen=True, slots=True)
class ProductionMixInput:
    path: Path
    sha256: str
    start_time: float = 0.0
    gain: float = 1.0
    kind: str = "dialogue"
    shot_id: str | None = None
    fade_in: float = 0.0
    fade_out: float = 0.0
    pan: float = 0.0


def mix_production_audio(
    inputs: Sequence[ProductionMixInput],
    output: str | Path,
    *,
    duration: float | None = None,
    mixer: Mixer | None = None,
) -> dict[str, Any]:
    """Mix exact hash-pinned inputs and return a signed-by-content evidence manifest."""
    if not inputs:
        raise ProductionAudioMixEvidenceError("production audio mix requires inputs")
    duration = _validated_duration(duration)
    engine = mixer or Mixer()
    records: list[dict[str, Any]] = []
    mix_inputs: list[MixInput] = []
    seen_dialogue_shots: set[str] = set()
    for index, item in enumerate(inputs):
        path = Path(item.path).resolve()
        if not path.is_file():
            raise ProductionAudioMixEvidenceError(
                f"production mix input {index} does not exist"
            )
        expected = _required_sha256(item.sha256, field=f"input {index} SHA-256")
        actual = _sha256(path)
        if actual != expected:
            raise ProductionAudioMixEvidenceError(
                f"production mix input {index} artifact hash does not match evidence"
            )
        try:
            start_time = float(item.start_time)
            gain = float(item.gain)
            fade_in = float(item.fade_in)
            fade_out = float(item.fade_out)
            pan = float(item.pan)
        except (TypeError, ValueError) as exc:
            raise ProductionAudioMixEvidenceError(
                f"production mix input {index} has invalid numeric controls"
            ) from exc
        if not all(
            math.isfinite(value) for value in (start_time, gain, fade_in, fade_out, pan)
        ):
            raise ProductionAudioMixEvidenceError(
                f"production mix input {index} has non-finite controls"
            )
        if start_time < 0 or gain < 0 or fade_in < 0 or fade_out < 0:
            raise ProductionAudioMixEvidenceError(
                f"production mix input {index} has invalid negative controls"
            )
        kind = str(item.kind or "").strip()
        if not kind:
            raise ProductionAudioMixEvidenceError(
                f"production mix input {index} requires kind"
            )
        shot_id = str(item.shot_id or "").strip() or None
        if kind == "dialogue":
            if shot_id is None:
                raise ProductionAudioMixEvidenceError(
                    f"dialogue mix input {index} requires shot_id"
                )
            if shot_id in seen_dialogue_shots:
                raise ProductionAudioMixEvidenceError(
                    f"duplicate dialogue mix input for shot {shot_id}"
                )
            seen_dialogue_shots.add(shot_id)
        elif shot_id is not None:
            raise ProductionAudioMixEvidenceError(
                f"non-dialogue mix input {index} cannot claim shot_id"
            )
        records.append(
            {
                "index": index,
                "path": str(path),
                "sha256": actual,
                "start_time_seconds": start_time,
                "gain": gain,
                "kind": kind,
                "shot_id": shot_id,
                "fade_in_seconds": fade_in,
                "fade_out_seconds": fade_out,
                "pan": pan,
            }
        )
        mix_inputs.append(
            MixInput(
                path=path,
                start_time=start_time,
                gain=gain,
                kind=kind,
                fade_in=fade_in,
                fade_out=fade_out,
                pan=pan,
            )
        )

    target = Path(output).resolve()
    mixed = engine.mix(mix_inputs, target, duration=duration)
    output_path = Path(mixed).resolve()
    output_sha = _sha256(output_path)
    manifest: dict[str, Any] = {
        "schema": PRODUCTION_AUDIO_MIX_EVIDENCE_SCHEMA,
        "sample_rate_hz": engine.sample_rate,
        "channel_layout": engine.channel_layout,
        "duration_seconds": duration,
        "inputs": records,
        "output_path": str(output_path),
        "output_sha256": output_sha,
    }
    manifest["evidence_sha256"] = _canonical_hash(manifest)
    return manifest


def validate_production_audio_mix_evidence(evidence: Mapping[str, Any]) -> str:
    """Validate manifest integrity, semantics, and exact artifact lineage."""
    if evidence.get("schema") != PRODUCTION_AUDIO_MIX_EVIDENCE_SCHEMA:
        raise ProductionAudioMixEvidenceError("unsupported production audio mix schema")
    recorded = _required_sha256(
        str(evidence.get("evidence_sha256") or ""), field="mix evidence SHA-256"
    )
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    if _canonical_hash(unsigned) != recorded:
        raise ProductionAudioMixEvidenceError(
            "production audio mix evidence SHA-256 does not match contents"
        )

    sample_rate = evidence.get("sample_rate_hz")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
    ):
        raise ProductionAudioMixEvidenceError(
            "production audio mix evidence requires a positive sample rate"
        )
    channel_layout = str(evidence.get("channel_layout") or "").strip()
    if not channel_layout:
        raise ProductionAudioMixEvidenceError(
            "production audio mix evidence requires channel layout"
        )
    _validated_duration(evidence.get("duration_seconds"))

    output = Path(str(evidence.get("output_path") or "")).resolve()
    expected_output = _required_sha256(
        str(evidence.get("output_sha256") or ""), field="mix output SHA-256"
    )
    if not output.is_file() or _sha256(output) != expected_output:
        raise ProductionAudioMixEvidenceError(
            "production audio mix output does not match evidence"
        )
    inputs = evidence.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ProductionAudioMixEvidenceError(
            "production audio mix evidence requires inputs"
        )

    seen_dialogue_shots: set[str] = set()
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            raise ProductionAudioMixEvidenceError(
                f"production audio mix input {index} must be a mapping"
            )
        if item.get("index") != index:
            raise ProductionAudioMixEvidenceError(
                f"production audio mix input {index} has invalid ordered index"
            )
        _validated_controls(item, index=index)
        kind = str(item.get("kind") or "").strip()
        if not kind:
            raise ProductionAudioMixEvidenceError(
                f"production audio mix input {index} requires kind"
            )
        shot_id = str(item.get("shot_id") or "").strip() or None
        if kind == "dialogue":
            if shot_id is None:
                raise ProductionAudioMixEvidenceError(
                    f"dialogue mix input {index} requires shot_id"
                )
            if shot_id in seen_dialogue_shots:
                raise ProductionAudioMixEvidenceError(
                    f"duplicate dialogue mix input for shot {shot_id}"
                )
            seen_dialogue_shots.add(shot_id)
        elif shot_id is not None:
            raise ProductionAudioMixEvidenceError(
                f"non-dialogue mix input {index} cannot claim shot_id"
            )

        path = Path(str(item.get("path") or "")).resolve()
        expected = _required_sha256(
            str(item.get("sha256") or ""), field=f"mix input {index} SHA-256"
        )
        if not path.is_file() or _sha256(path) != expected:
            raise ProductionAudioMixEvidenceError(
                f"production audio mix input {index} does not match evidence"
            )
    return expected_output


__all__ = [
    "PRODUCTION_AUDIO_MIX_EVIDENCE_SCHEMA",
    "ProductionAudioMixEvidenceError",
    "ProductionMixInput",
    "mix_production_audio",
    "validate_production_audio_mix_evidence",
]
