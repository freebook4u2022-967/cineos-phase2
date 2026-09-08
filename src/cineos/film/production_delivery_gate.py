"""Strong final delivery gate for connected production films.

This gate composes the existing connected GPU/QC/lip-sync/mix evidence contract with
independent decode-level checks that both the final soundtrack and final video derive
from the exact approved production assets. These integrity bindings do not replace
aesthetic, listening, or semantic quality evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from cineos.audio.lipsync_qc import LipSyncAnalyzerProvenance, LipSyncThresholds

from .audio_binding import (
    AudioBindingError,
    AudioBindingEvidence,
    measure_audio_binding,
)
from .connected_production_evidence import (
    ConnectedProductionFilmEvidence,
    ConnectedProductionFilmEvidenceError,
    validate_connected_production_film_evidence,
)
from .visual_binding import (
    VisualBindingError,
    VisualBindingEvidence,
    measure_visual_binding,
)

PRODUCTION_DELIVERY_EVIDENCE_SCHEMA = "cineos-production-delivery-evidence/0.3"


class ProductionDeliveryEvidenceError(RuntimeError):
    """Raised when the exact connected film cannot satisfy final delivery evidence."""


@dataclass(frozen=True, slots=True)
class ProductionDeliveryEvidence:
    """Final connected-film evidence including decoded audiovisual bindings."""

    connected_film: ConnectedProductionFilmEvidence
    audio_binding: AudioBindingEvidence | None
    visual_binding: VisualBindingEvidence | None = None

    @property
    def accepted(self) -> bool:
        return (
            self.connected_film.accepted
            and (self.audio_binding is None or self.audio_binding.accepted)
            and (self.visual_binding is None or self.visual_binding.accepted)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCTION_DELIVERY_EVIDENCE_SCHEMA,
            "accepted": self.accepted,
            "connected_film": self.connected_film.to_dict(),
            "audio_binding": (
                self.audio_binding.to_dict() if self.audio_binding is not None else None
            ),
            "visual_binding": (
                self.visual_binding.to_dict()
                if self.visual_binding is not None
                else None
            ),
        }


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionDeliveryEvidenceError(
            f"production delivery evidence requires {field}"
        )
    return value.strip()


def _required_sha256(value: Any, *, field: str) -> str:
    normalized = _required_text(value, field=field).lower()
    if len(normalized) != 64:
        raise ProductionDeliveryEvidenceError(
            f"production delivery evidence requires a valid {field}"
        )
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ProductionDeliveryEvidenceError(
            f"production delivery evidence requires a valid {field}"
        ) from exc
    return normalized


def _validate_final_audio_binding(
    assembly: Mapping[str, Any],
    *,
    required: bool = False,
) -> AudioBindingEvidence | None:
    audio = assembly.get("audio")
    if audio is None:
        if required:
            raise ProductionDeliveryEvidenceError(
                "dialogue-bearing production delivery requires approved audio evidence "
                "bound to the final MP4"
            )
        return None
    if not isinstance(audio, Mapping):
        raise ProductionDeliveryEvidenceError(
            "production assembly contains malformed approved audio evidence"
        )
    approved_path = _required_text(audio.get("path"), field="approved audio path")
    expected_audio_sha = _required_sha256(
        audio.get("sha256"), field="approved audio SHA-256"
    )
    final_path = _required_text(assembly.get("final_mp4"), field="final MP4 path")
    expected_final_sha = _required_sha256(
        assembly.get("final_mp4_sha256"), field="final MP4 SHA-256"
    )
    try:
        binding = measure_audio_binding(approved_path, final_path)
    except AudioBindingError as exc:
        raise ProductionDeliveryEvidenceError(
            f"final-film audio is not bound to the approved production mix: {exc}"
        ) from exc
    if binding.approved_audio_sha256 != expected_audio_sha:
        raise ProductionDeliveryEvidenceError(
            "audio-binding measurement does not match the approved production mix hash"
        )
    if binding.final_artifact_sha256 != expected_final_sha:
        raise ProductionDeliveryEvidenceError(
            "audio-binding measurement does not match the approved final MP4 hash"
        )
    return binding


def _visual_edit_durations(assembly: Mapping[str, Any]) -> Sequence[float] | None:
    timeline = assembly.get("timeline")
    if not isinstance(timeline, Mapping):
        return None
    compatibility = timeline.get("compatibility")
    if not isinstance(compatibility, Mapping):
        return None
    durations = compatibility.get("edit_durations_seconds")
    if durations is None:
        return None
    if not isinstance(durations, list):
        raise ProductionDeliveryEvidenceError(
            "production visual binding requires valid edit-duration evidence"
        )
    return durations


def _validate_final_visual_binding(
    assembly: Mapping[str, Any],
) -> VisualBindingEvidence:
    shots = assembly.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ProductionDeliveryEvidenceError(
            "production delivery requires ordered approved shot artifacts for visual binding"
        )

    source_paths: list[str] = []
    expected_source_sha: list[str] = []
    for index, shot in enumerate(shots):
        if not isinstance(shot, Mapping):
            raise ProductionDeliveryEvidenceError(
                f"production visual binding shot {index} is malformed"
            )
        source_paths.append(
            _required_text(
                shot.get("output_path"),
                field=f"approved shot {index} output path",
            )
        )
        expected_source_sha.append(
            _required_sha256(
                shot.get("output_sha256"),
                field=f"approved shot {index} output SHA-256",
            )
        )

    final_path = _required_text(assembly.get("final_mp4"), field="final MP4 path")
    expected_final_sha = _required_sha256(
        assembly.get("final_mp4_sha256"), field="final MP4 SHA-256"
    )
    durations = _visual_edit_durations(assembly)
    if durations is not None and len(durations) != len(source_paths):
        raise ProductionDeliveryEvidenceError(
            "production visual-binding edit duration count does not match shot count"
        )

    try:
        binding = measure_visual_binding(
            source_paths,
            final_path,
            durations_seconds=durations,
        )
    except VisualBindingError as exc:
        raise ProductionDeliveryEvidenceError(
            f"final-film video is not bound to approved connected shots: {exc}"
        ) from exc

    if tuple(binding.source_sha256) != tuple(expected_source_sha):
        raise ProductionDeliveryEvidenceError(
            "visual-binding measurement does not match approved shot artifact hashes"
        )
    if binding.final_artifact_sha256 != expected_final_sha:
        raise ProductionDeliveryEvidenceError(
            "visual-binding measurement does not match the approved final MP4 hash"
        )
    return binding


def validate_production_delivery_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
    assembly: Mapping[str, Any],
    *,
    lipsync_evidence: Sequence[Mapping[str, Any]] | None = None,
    required_dialogue_shot_ids: Sequence[str] = (),
    dialogue_audio_sha256_by_shot: Mapping[str, str] | None = None,
    expected_lipsync_analyzer: LipSyncAnalyzerProvenance | None = None,
    lipsync_thresholds: LipSyncThresholds | None = None,
    audio_mix_evidence: Mapping[str, Any] | None = None,
) -> ProductionDeliveryEvidence:
    """Validate the strongest available final-film evidence chain.

    Real connected-production evidence always receives decode-level visual binding.
    The ``shot_count`` feature check retains compatibility with older test/plugin
    connected-evidence objects that predate delivery-level visual binding; native CINEOS
    connected evidence always declares this field.
    """
    try:
        connected = validate_connected_production_film_evidence(
            benchmark,
            assembly,
            lipsync_evidence=lipsync_evidence,
            required_dialogue_shot_ids=required_dialogue_shot_ids,
            dialogue_audio_sha256_by_shot=dialogue_audio_sha256_by_shot,
            expected_lipsync_analyzer=expected_lipsync_analyzer,
            lipsync_thresholds=lipsync_thresholds,
            audio_mix_evidence=audio_mix_evidence,
        )
    except (ConnectedProductionFilmEvidenceError, TypeError) as exc:
        raise ProductionDeliveryEvidenceError(
            f"connected production film evidence is invalid: {exc}"
        ) from exc

    dialogue_shot_ids = tuple(getattr(connected, "dialogue_shot_ids", ()))
    audio_binding = _validate_final_audio_binding(
        assembly,
        required=bool(dialogue_shot_ids),
    )
    visual_binding = (
        _validate_final_visual_binding(assembly)
        if hasattr(connected, "shot_count")
        else None
    )
    evidence = ProductionDeliveryEvidence(
        connected_film=connected,
        audio_binding=audio_binding,
        visual_binding=visual_binding,
    )
    if not evidence.accepted:
        raise ProductionDeliveryEvidenceError(
            "production delivery evidence did not satisfy the final gate"
        )
    return evidence


__all__ = [
    "PRODUCTION_DELIVERY_EVIDENCE_SCHEMA",
    "ProductionDeliveryEvidence",
    "ProductionDeliveryEvidenceError",
    "validate_production_delivery_evidence",
]
