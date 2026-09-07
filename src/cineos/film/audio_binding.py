"""Measured binding between an approved production mix and final-film audio.

The final delivery encoder is allowed to transcode the approved mix (for example WAV to
AAC), so byte equality cannot prove soundtrack identity. This module independently
decodes both artifacts to the same low-rate mono PCM representation and requires strong
waveform correlation with a small bounded alignment search. It is an integrity gate, not
an audio-quality or intelligibility metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIO_BINDING_SCHEMA = "cineos-production-audio-binding/0.1"
AUDIO_BINDING_SAMPLE_RATE_HZ = 400
AUDIO_BINDING_MAX_LAG_MS = 40
MIN_AUDIO_BINDING_CORRELATION = 0.90
MIN_AUDIO_BINDING_SAMPLES = 200


class AudioBindingError(RuntimeError):
    """Raised when final-film audio cannot be bound to the approved mix."""


@dataclass(frozen=True, slots=True)
class AudioBindingEvidence:
    """Measured evidence that final-film audio derives from an approved mix."""

    approved_audio_sha256: str
    final_artifact_sha256: str
    sample_rate_hz: int
    compared_samples: int
    alignment_lag_samples: int
    correlation: float
    minimum_correlation: float
    evidence_sha256: str

    @property
    def accepted(self) -> bool:
        return self.correlation >= self.minimum_correlation

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AUDIO_BINDING_SCHEMA,
            "approved_audio_sha256": self.approved_audio_sha256,
            "final_artifact_sha256": self.final_artifact_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "compared_samples": self.compared_samples,
            "alignment_lag_samples": self.alignment_lag_samples,
            "correlation": self.correlation,
            "minimum_correlation": self.minimum_correlation,
            "accepted": self.accepted,
            "evidence_sha256": self.evidence_sha256,
        }


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise AudioBindingError(
            "FFmpeg is unavailable; install ffmpeg for production audio binding"
        )
    return executable


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AudioBindingError(f"cannot hash audio-binding artifact: {path}") from exc
    return digest.hexdigest()


def _decode_binding_pcm(path: Path) -> tuple[int, ...]:
    source = path.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise AudioBindingError(f"missing or empty audio-binding artifact: {source}")
    command = [
        _ffmpeg(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_BINDING_SAMPLE_RATE_HZ),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioBindingError(f"FFmpeg audio-binding decode failed: {stderr}")
    payload = result.stdout
    if not isinstance(payload, (bytes, bytearray)) or len(payload) % 2:
        raise AudioBindingError("FFmpeg returned malformed audio-binding PCM")
    samples = array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) < MIN_AUDIO_BINDING_SAMPLES:
        raise AudioBindingError(
            "audio-binding decode is too short for reliable correlation evidence"
        )
    return tuple(int(value) for value in samples)


def _correlation(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        raise AudioBindingError("audio-binding correlation requires equal non-empty spans")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = 0.0
    left_energy = 0.0
    right_energy = 0.0
    for left_value, right_value in zip(left, right):
        l = left_value - left_mean
        r = right_value - right_mean
        numerator += l * r
        left_energy += l * l
        right_energy += r * r
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 0 or not math.isfinite(denominator):
        raise AudioBindingError("audio-binding PCM has insufficient non-constant signal")
    value = numerator / denominator
    if not math.isfinite(value):
        raise AudioBindingError("audio-binding correlation is non-finite")
    return max(-1.0, min(1.0, value))


def _best_alignment(
    approved: tuple[int, ...], encoded: tuple[int, ...]
) -> tuple[float, int, int]:
    max_lag = max(
        1,
        round(AUDIO_BINDING_SAMPLE_RATE_HZ * AUDIO_BINDING_MAX_LAG_MS / 1000),
    )
    best: tuple[float, int, int] | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            left = approved[lag:]
            right = encoded
        else:
            left = approved
            right = encoded[-lag:]
        compared = min(len(left), len(right))
        if compared < MIN_AUDIO_BINDING_SAMPLES:
            continue
        score = _correlation(left[:compared], right[:compared])
        candidate = (score, lag, compared)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise AudioBindingError(
            "approved mix and final soundtrack have no sufficient overlapping audio span"
        )
    return best


def _evidence_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def measure_audio_binding(
    approved_audio: str | Path,
    final_artifact: str | Path,
    *,
    minimum_correlation: float = MIN_AUDIO_BINDING_CORRELATION,
) -> AudioBindingEvidence:
    """Measure whether final-film decoded audio matches the approved production mix."""
    try:
        threshold = float(minimum_correlation)
    except (TypeError, ValueError) as exc:
        raise AudioBindingError("audio-binding threshold must be finite") from exc
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise AudioBindingError("audio-binding threshold must be in the interval (0, 1]")

    approved_path = Path(approved_audio).resolve()
    final_path = Path(final_artifact).resolve()
    approved_pcm = _decode_binding_pcm(approved_path)
    final_pcm = _decode_binding_pcm(final_path)
    correlation, lag, compared = _best_alignment(approved_pcm, final_pcm)
    unsigned = {
        "schema": AUDIO_BINDING_SCHEMA,
        "approved_audio_sha256": _file_hash(approved_path),
        "final_artifact_sha256": _file_hash(final_path),
        "sample_rate_hz": AUDIO_BINDING_SAMPLE_RATE_HZ,
        "compared_samples": compared,
        "alignment_lag_samples": lag,
        "correlation": correlation,
        "minimum_correlation": threshold,
        "accepted": correlation >= threshold,
    }
    evidence = AudioBindingEvidence(
        approved_audio_sha256=unsigned["approved_audio_sha256"],
        final_artifact_sha256=unsigned["final_artifact_sha256"],
        sample_rate_hz=AUDIO_BINDING_SAMPLE_RATE_HZ,
        compared_samples=compared,
        alignment_lag_samples=lag,
        correlation=correlation,
        minimum_correlation=threshold,
        evidence_sha256=_evidence_hash(unsigned),
    )
    if not evidence.accepted:
        raise AudioBindingError(
            "final-film soundtrack does not match the approved production mix: "
            f"correlation={correlation:.6f}, required>={threshold:.6f}"
        )
    return evidence


__all__ = [
    "AUDIO_BINDING_MAX_LAG_MS",
    "AUDIO_BINDING_SAMPLE_RATE_HZ",
    "AUDIO_BINDING_SCHEMA",
    "MIN_AUDIO_BINDING_CORRELATION",
    "AudioBindingError",
    "AudioBindingEvidence",
    "measure_audio_binding",
]
