"""Production semantic-QC capability contract for competitive CINEOS runs.

This module bridges the benchmark challenge contract to the semantic scorer ensemble.
It deliberately does not invent difficult-case scores. A production competitive run
must have an explicitly owned, production-attested semantic measurement for every
challenge-specific metric it declares before expensive video generation should begin.

The default core scorer may provide identity and motion measurements. Specialist
anatomy, locomotion, interaction, camera, lighting, physics, and lip-sync measurements
remain separate components with their own provenance. External pretrained foundations
are valid components when their provenance is transparent; they are never relabelled
as CINEOS-native models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .latentsync_syncnet_scorer import (
    SPEAKER_FACE_TRACK_INDEX_KEY,
    LatentSyncSyncNetScorer,
    latentsync_syncnet_component,
)
from .qwen25vl_semantic_judge import QWEN25VL_METRICS, Qwen25VLSemanticJudge
from .semantic_video_ensemble import (
    ProductionSemanticScorerEnsemble,
    SemanticScorerComponent,
    SemanticScorerEnsembleError,
)
from .sequence_quality import CHALLENGE_METRIC_REQUIREMENTS

PRODUCTION_SEMANTIC_QC_SCHEMA = "cineos-production-semantic-qc-capabilities/0.5"
CORE_SEMANTIC_METRICS = ("identity_similarity", "motion_quality")
_CHALLENGE_METADATA_KEYS = ("competitive_challenges", "benchmark_challenges")
_DIALOGUE_CHALLENGES = frozenset(("dialogue", "dialogue_lip_sync"))
VISUAL_DIFFICULT_CASE_CHALLENGES = tuple(
    challenge
    for challenge, metric in CHALLENGE_METRIC_REQUIREMENTS.items()
    if metric in QWEN25VL_METRICS
)


class ProductionSemanticQCError(RuntimeError):
    """Raised when competitive semantic QC cannot be proven before production."""


def _declared_challenges(shot: Any) -> frozenset[str]:
    metadata = getattr(shot, "metadata", None)
    if not isinstance(metadata, dict):
        return frozenset()
    declared: set[str] = set()
    for key in _CHALLENGE_METADATA_KEYS:
        raw = metadata.get(key)
        if raw is None or isinstance(raw, (str, bytes)):
            continue
        try:
            values = tuple(raw)
        except TypeError:
            continue
        declared.update(
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
    return frozenset(declared)


def _validated_dialogue_speakers(
    timing: list[Any], *, shot_index: int
) -> frozenset[str]:
    """Validate cue ownership syntax before deciding whether a shot is multi-speaker.

    A malformed second cue previously disappeared from ``valid_speakers`` and could
    make an actually ambiguous dialogue shot look single-speaker at preflight. The
    expensive renderer would then run before SyncNet rejected the cue. Validate the
    ownership syntax of every supplied cue first, while intentionally leaving face
    tracks and cue windows optional for well-formed single-speaker dialogue.
    """

    speakers: set[str] = set()
    for cue_index, cue in enumerate(timing):
        prefix = f"shot[{shot_index}] dialogue_timing[{cue_index}]"
        if not isinstance(cue, dict):
            raise ProductionSemanticQCError(
                f"{prefix} must be a mapping when dialogue_timing is supplied"
            )
        speaker_id = cue.get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise ProductionSemanticQCError(
                f"{prefix}.speaker_id must be non-empty when dialogue_timing is supplied"
            )
        speakers.add(speaker_id.strip())
    return frozenset(speakers)


def validate_dialogue_speaker_bindings(shots: Sequence[Any]) -> None:
    """Fail closed on ambiguous multi-speaker SyncNet cue ownership.

    The current production LatentSync adapter evaluates alternating speakers by
    extracting each declared cue window from one detected face track while retaining
    the shot's mixed audio. That is auditable only when every speaker maps stably and
    uniquely to one face track and cue windows do not overlap. Overlapping speakers
    require isolated per-speaker audio stems before they can produce defensible SyncNet
    evidence; accepting the mixed soundtrack would risk scoring the wrong voice against
    a face and overstating dialogue quality.

    Single-speaker dialogue remains compatible without explicit track/time metadata,
    but if ``dialogue_timing`` is supplied, every cue must at least have unambiguous
    speaker ownership. This prevents malformed multi-cue metadata from bypassing the
    preflight and failing only after expensive GPU generation.
    """

    for shot_index, shot in enumerate(shots):
        if not (_declared_challenges(shot) & _DIALOGUE_CHALLENGES):
            continue
        performance = getattr(shot, "performance", None)
        if not isinstance(performance, dict):
            continue
        timing = performance.get("dialogue_timing")
        if not isinstance(timing, list) or not timing:
            continue

        valid_speakers = _validated_dialogue_speakers(timing, shot_index=shot_index)
        if len(valid_speakers) < 2:
            continue

        speaker_to_track: dict[str, int] = {}
        track_to_speaker: dict[int, str] = {}
        windows: list[tuple[float, float, str]] = []
        for cue_index, cue in enumerate(timing):
            prefix = f"shot[{shot_index}] dialogue_timing[{cue_index}]"
            # _validated_dialogue_speakers established both mapping type and speaker_id.
            speaker_id = cue["speaker_id"].strip()
            track_index = cue.get(SPEAKER_FACE_TRACK_INDEX_KEY)
            if (
                isinstance(track_index, bool)
                or not isinstance(track_index, int)
                or track_index < 0
            ):
                raise ProductionSemanticQCError(
                    f"{prefix}.{SPEAKER_FACE_TRACK_INDEX_KEY} must be a non-negative integer"
                )

            previous_track = speaker_to_track.setdefault(speaker_id, track_index)
            if previous_track != track_index:
                raise ProductionSemanticQCError(
                    "multi-speaker lip-sync requires stable speaker-to-face ownership; "
                    f"{speaker_id!r} maps to both track {previous_track} and {track_index}"
                )
            previous_speaker = track_to_speaker.setdefault(track_index, speaker_id)
            if previous_speaker != speaker_id:
                raise ProductionSemanticQCError(
                    "multi-speaker lip-sync requires distinct speakers to bind distinct "
                    f"face tracks; track {track_index} is assigned to both "
                    f"{previous_speaker!r} and {speaker_id!r}"
                )

            start = cue.get("start_seconds")
            end = cue.get("end_seconds")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
            ):
                raise ProductionSemanticQCError(
                    f"{prefix} requires numeric start_seconds/end_seconds"
                )
            start_f = float(start)
            end_f = float(end)
            if (
                not math.isfinite(start_f)
                or not math.isfinite(end_f)
                or start_f < 0.0
                or end_f <= start_f
            ):
                raise ProductionSemanticQCError(
                    f"{prefix} requires a finite positive dialogue cue window"
                )
            windows.append((start_f, end_f, speaker_id))

        windows.sort(key=lambda item: (item[0], item[1], item[2]))
        for previous, current in zip(windows, windows[1:]):
            if current[0] < previous[1]:
                raise ProductionSemanticQCError(
                    "multi-speaker lip-sync cue windows must not overlap until isolated "
                    "per-speaker audio stems are available; "
                    f"{previous[2]!r} [{previous[0]:.3f}, {previous[1]:.3f}) overlaps "
                    f"{current[2]!r} [{current[0]:.3f}, {current[1]:.3f})"
                )


def required_semantic_metrics(shots: Sequence[Any]) -> frozenset[str]:
    """Return semantic metrics required by the difficult cases declared by ``shots``.

    Unknown challenge names are intentionally ignored here because benchmark request
    validation owns the challenge vocabulary. This helper only translates known
    quality-contract declarations into scorer capabilities without duplicating request
    validation policy.
    """

    required = set(CORE_SEMANTIC_METRICS)
    for shot in shots:
        metadata = getattr(shot, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        for key in _CHALLENGE_METADATA_KEYS:
            raw = metadata.get(key)
            if raw is None or isinstance(raw, (str, bytes)):
                continue
            try:
                values = tuple(raw)
            except TypeError:
                continue
            for challenge in values:
                if not isinstance(challenge, str):
                    continue
                metric = CHALLENGE_METRIC_REQUIREMENTS.get(challenge.strip())
                if metric is not None:
                    required.add(metric)
    return frozenset(required)


def validate_production_semantic_capabilities(
    scorer: ProductionSemanticScorerEnsemble,
    shots: Sequence[Any],
) -> frozenset[str]:
    """Fail closed unless every required metric has an attested scorer owner.

    The check happens against declared ownership before artifact decoding or model
    rendering. This prevents a 5-10 shot CUDA benchmark from spending substantial GPU
    time only to discover that anatomy, lip-sync, physics, or another required metric
    could never have been measured by the configured QC stack.
    """

    if not isinstance(scorer, ProductionSemanticScorerEnsemble):
        raise TypeError("scorer must be ProductionSemanticScorerEnsemble")
    if scorer.semantic_measurement_evidence is not True:
        raise ProductionSemanticQCError(
            "production semantic QC requires every scorer component to attest real "
            "semantic measurement evidence"
        )

    validate_dialogue_speaker_bindings(shots)
    required = required_semantic_metrics(shots)
    available = frozenset(scorer.metric_owners)
    missing = sorted(required - available)
    if missing:
        raise ProductionSemanticQCError(
            "production semantic QC is missing real measured scorer capability for: "
            + ", ".join(missing)
        )
    return required


def build_production_semantic_scorer(
    core_scorer: Any,
    shots: Sequence[Any],
    *,
    specialists: Sequence[SemanticScorerComponent] = (),
) -> ProductionSemanticScorerEnsemble:
    """Compose core plus specialist scorers and validate production metric coverage.

    ``core_scorer`` is restricted to identity and motion ownership. Difficult-case
    metrics must come from separately named specialist components, making provenance
    substitution visible and preventing a generic visual encoder from being relabelled
    as anatomy, physics, interaction, or lip-sync evidence.
    """

    if not callable(core_scorer):
        raise TypeError("core_scorer must be callable")
    if isinstance(specialists, (str, bytes)) or not isinstance(specialists, Sequence):
        raise TypeError("specialists must be a sequence of SemanticScorerComponent")
    materialized = tuple(specialists)
    if any(not isinstance(item, SemanticScorerComponent) for item in materialized):
        raise TypeError("specialists must contain only SemanticScorerComponent values")

    core = SemanticScorerComponent(
        name="core_identity_motion",
        scorer=core_scorer,
        measured_metrics=CORE_SEMANTIC_METRICS,
    )
    try:
        ensemble = ProductionSemanticScorerEnsemble((core, *materialized))
    except (TypeError, ValueError, SemanticScorerEnsembleError) as exc:
        raise ProductionSemanticQCError(
            f"cannot compose production semantic QC scorers: {exc}"
        ) from exc
    validate_production_semantic_capabilities(ensemble, shots)
    return ensemble


def build_seedance_challenge_semantic_scorer(
    core_scorer: Any,
    shots: Sequence[Any],
    *,
    visual_judge: Qwen25VLSemanticJudge,
    av_sync_scorer: LatentSyncSyncNetScorer,
) -> ProductionSemanticScorerEnsemble:
    """Compose the audited difficult-case production scorer stack.

    Qwen2.5-VL owns only the seven visual difficult-case metrics it actually judges
    and is activated only when a shot declares at least one matching visual challenge.
    This preserves the same fail-closed coverage while avoiding a large multimodal
    inference on ordinary or dialogue-only shots. LatentSync SyncNet independently
    owns dialogue lip-sync from real audio/video evidence. Keeping the components
    disjoint prevents a visual-only model from satisfying the AV challenge and
    preserves transparent external-model provenance.
    """

    if not isinstance(visual_judge, Qwen25VLSemanticJudge):
        raise TypeError("visual_judge must be Qwen25VLSemanticJudge")
    if not isinstance(av_sync_scorer, LatentSyncSyncNetScorer):
        raise TypeError("av_sync_scorer must be LatentSyncSyncNetScorer")
    visual_component = SemanticScorerComponent(
        name="qwen25vl_visual_difficult_cases",
        scorer=visual_judge,
        measured_metrics=QWEN25VL_METRICS,
        required_challenges=VISUAL_DIFFICULT_CASE_CHALLENGES,
    )
    return build_production_semantic_scorer(
        core_scorer,
        shots,
        specialists=(visual_component, latentsync_syncnet_component(av_sync_scorer)),
    )


__all__ = [
    "CORE_SEMANTIC_METRICS",
    "PRODUCTION_SEMANTIC_QC_SCHEMA",
    "ProductionSemanticQCError",
    "VISUAL_DIFFICULT_CASE_CHALLENGES",
    "build_production_semantic_scorer",
    "build_seedance_challenge_semantic_scorer",
    "required_semantic_metrics",
    "validate_dialogue_speaker_bindings",
    "validate_production_semantic_capabilities",
]
