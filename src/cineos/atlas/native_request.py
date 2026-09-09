"""CINEOS-native, renderer-independent shot request contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from cineos.conditioning import ConditioningPackage

NATIVE_SHOT_SCHEMA = "cineos-native-shot-request/0.1"
_COMPETITIVE_CHALLENGE_METADATA_KEY = "competitive_challenges"
_DIALOGUE_LIP_SYNC_CHALLENGE = "dialogue_lip_sync"
_SEEDANCE_CHALLENGE_METADATA_KEY = "benchmark_challenges"
_SEEDANCE_DIALOGUE_CHALLENGE = "dialogue"


def _finite_positive_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number")
    return normalized


def _finite_nonnegative_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return normalized


def _dialogue_time(
    cue: dict[str, Any], *, canonical: str, legacy: str, index: int
) -> Any:
    """Read a canonical dialogue timestamp while accepting the 0.1 legacy alias."""
    canonical_value = cue.get(canonical)
    legacy_value = cue.get(legacy)
    if canonical_value is not None and legacy_value is not None:
        if canonical_value != legacy_value:
            raise ValueError(
                f"performance.dialogue_timing[{index}] has conflicting {canonical}/{legacy} values"
            )
        return canonical_value
    return canonical_value if canonical_value is not None else legacy_value


def _challenge_declared(metadata: dict[str, Any], *, key: str, value: str) -> bool:
    raw = metadata.get(key)
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return False
    return value in raw


def _requires_seedance_dialogue_evidence(metadata: dict[str, Any]) -> bool:
    return _challenge_declared(
        metadata,
        key=_SEEDANCE_CHALLENGE_METADATA_KEY,
        value=_SEEDANCE_DIALOGUE_CHALLENGE,
    )


def _requires_competitive_dialogue_grounding(metadata: dict[str, Any]) -> bool:
    return _challenge_declared(
        metadata,
        key=_COMPETITIVE_CHALLENGE_METADATA_KEY,
        value=_DIALOGUE_LIP_SYNC_CHALLENGE,
    ) or _requires_seedance_dialogue_evidence(metadata)


def _conditioned_character_id(character: dict[str, Any], *, index: int) -> str | None:
    canonical = character.get("character_uuid")
    legacy = character.get("character_id")
    if canonical is not None and (
        not isinstance(canonical, str) or not canonical.strip()
    ):
        raise ValueError(
            f"characters[{index}].character_uuid must be non-empty when supplied"
        )
    if legacy is not None and (not isinstance(legacy, str) or not legacy.strip()):
        raise ValueError(
            f"characters[{index}].character_id must be non-empty when supplied"
        )
    if canonical is not None and legacy is not None:
        if canonical.strip() != legacy.strip():
            raise ValueError(
                f"characters[{index}] has conflicting character_uuid/character_id"
            )
        return canonical.strip()
    if canonical is not None:
        return canonical.strip()
    if legacy is not None:
        return legacy.strip()
    return None


@dataclass(slots=True)
class NativeShotRequest:
    shot_id: str
    scene_id: str
    camera: dict[str, Any]
    characters: list[dict[str, Any]]
    environment: dict[str, Any] | None
    wardrobe: list[dict[str, Any]]
    props: list[dict[str, Any]]
    continuity: dict[str, Any]
    performance: dict[str, Any]
    approved_reference_ids: list[str]
    deterministic_seed: int
    renderer_requirements: dict[str, Any]
    schema: str = NATIVE_SHOT_SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def validate_timing_integrity(self) -> None:
        """Fail closed on malformed timing while preserving 0.1 frame aliases.

        ``camera`` is the timing authority consumed by the current Diffusers runtime.
        ``renderer_requirements`` may independently carry the same contract for
        capability selection. When both declare FPS or duration they must agree;
        otherwise dialogue can validate against one timeline while inference renders
        another. A camera-only duration/FPS is therefore also used to bound dialogue.
        """
        requirement_duration: float | None = None
        if "duration_seconds" in self.renderer_requirements:
            requirement_duration = _finite_positive_number(
                self.renderer_requirements["duration_seconds"],
                field_name="renderer_requirements.duration_seconds",
            )
        camera_duration: float | None = None
        if "duration" in self.camera:
            camera_duration = _finite_positive_number(
                self.camera["duration"], field_name="camera.duration"
            )
        if requirement_duration is not None and camera_duration is not None:
            if not math.isclose(
                requirement_duration, camera_duration, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    "camera.duration conflicts with renderer_requirements.duration_seconds"
                )
        duration_seconds = (
            camera_duration if camera_duration is not None else requirement_duration
        )

        requirement_fps: float | None = None
        if "fps" in self.renderer_requirements:
            requirement_fps = _finite_positive_number(
                self.renderer_requirements["fps"],
                field_name="renderer_requirements.fps",
            )
        camera_fps: float | None = None
        if "fps" in self.camera:
            camera_fps = _finite_positive_number(
                self.camera["fps"], field_name="camera.fps"
            )
        if requirement_fps is not None and camera_fps is not None:
            if not math.isclose(requirement_fps, camera_fps, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("camera.fps conflicts with renderer_requirements.fps")
        fps = camera_fps if camera_fps is not None else requirement_fps

        require_dialogue_grounding = _requires_competitive_dialogue_grounding(
            self.metadata
        )
        require_seedance_dialogue = _requires_seedance_dialogue_evidence(self.metadata)
        dialogue_timing = self.performance.get("dialogue_timing")
        if dialogue_timing in (None, []):
            if require_seedance_dialogue:
                raise ValueError(
                    "competitive dialogue benchmark requires non-empty performance.dialogue_timing"
                )
            return
        if not isinstance(dialogue_timing, list):
            raise ValueError("performance.dialogue_timing must be a list when supplied")

        character_ids: set[str] = set()
        if require_dialogue_grounding:
            for index, character in enumerate(self.characters):
                if not isinstance(character, dict):
                    continue
                character_id = _conditioned_character_id(character, index=index)
                if character_id is not None:
                    if character_id in character_ids:
                        raise ValueError(
                            "competitive dialogue_lip_sync requires unique conditioned "
                            f"character identities; duplicate {character_id!r}"
                        )
                    character_ids.add(character_id)
            if not character_ids:
                raise ValueError(
                    "competitive dialogue_lip_sync requires at least one conditioned "
                    "character_uuid or legacy character_id"
                )

        for index, cue in enumerate(dialogue_timing):
            if not isinstance(cue, dict):
                raise ValueError(
                    f"performance.dialogue_timing[{index}] must be a mapping"
                )
            speaker_id = cue.get("speaker_id")
            if speaker_id is not None and (
                not isinstance(speaker_id, str) or not speaker_id.strip()
            ):
                raise ValueError(
                    f"performance.dialogue_timing[{index}].speaker_id must be non-empty when supplied"
                )
            if require_dialogue_grounding:
                if speaker_id is None:
                    raise ValueError(
                        f"performance.dialogue_timing[{index}] requires speaker_id for competitive dialogue_lip_sync"
                    )
                if speaker_id not in character_ids:
                    raise ValueError(
                        f"performance.dialogue_timing[{index}].speaker_id {speaker_id!r} "
                        "does not match a conditioned character_id; conditioned character "
                        "identity may use character_id or character_uuid"
                    )

            raw_start = _dialogue_time(
                cue, canonical="start_seconds", legacy="start", index=index
            )
            raw_end = _dialogue_time(
                cue, canonical="end_seconds", legacy="end", index=index
            )
            has_seconds = raw_start is not None or raw_end is not None
            has_frames = "start_frame" in cue or "end_frame" in cue
            if has_seconds and has_frames:
                raise ValueError(
                    f"performance.dialogue_timing[{index}] cannot mix seconds and frame timing"
                )
            if has_seconds:
                start_seconds = _finite_nonnegative_number(
                    raw_start,
                    field_name=f"performance.dialogue_timing[{index}].start_seconds",
                )
                end_seconds = _finite_nonnegative_number(
                    raw_end,
                    field_name=f"performance.dialogue_timing[{index}].end_seconds",
                )
            elif has_frames:
                start_frame = _finite_nonnegative_number(
                    cue.get("start_frame"),
                    field_name=f"performance.dialogue_timing[{index}].start_frame",
                )
                end_frame = _finite_nonnegative_number(
                    cue.get("end_frame"),
                    field_name=f"performance.dialogue_timing[{index}].end_frame",
                )
                if end_frame <= start_frame:
                    raise ValueError(
                        f"performance.dialogue_timing[{index}] must have end_frame greater than start_frame"
                    )
                if fps is None:
                    continue
                start_seconds = start_frame / fps
                end_seconds = end_frame / fps
            else:
                if require_seedance_dialogue:
                    raise ValueError(
                        f"performance.dialogue_timing[{index}] requires seconds or frame timing"
                    )
                continue

            if end_seconds <= start_seconds:
                raise ValueError(
                    f"performance.dialogue_timing[{index}] must have end_seconds greater than start_seconds"
                )
            if duration_seconds is not None and end_seconds > duration_seconds:
                raise ValueError(
                    f"performance.dialogue_timing[{index}] extends beyond the shot duration"
                )

    def payload(self) -> dict[str, Any]:
        self.validate_timing_integrity()
        data = asdict(self)
        data.pop("content_hash", None)
        return data

    def _expected_content_hash(self) -> str:
        payload = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def content_hash_is_current(self) -> bool:
        return (
            bool(self.content_hash)
            and self.content_hash == self._expected_content_hash()
        )

    def refresh_hash(self) -> str:
        self.content_hash = self._expected_content_hash()
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        self.validate_timing_integrity()
        if not self.content_hash:
            self.refresh_hash()
        return asdict(self)


def compile_native_shot_request(package: ConditioningPackage) -> NativeShotRequest:
    """Compile an existing CINEOS ConditioningPackage into a native shot request."""
    if not package.character_conditioning and not package.approved_reference_ids:
        raise ValueError(
            "native shot request requires approved conditioning references"
        )

    request = NativeShotRequest(
        shot_id=package.shot_id,
        scene_id=package.scene_id,
        camera=asdict(package.camera_conditioning),
        characters=[asdict(item) for item in package.character_conditioning],
        environment=(
            asdict(package.environment_conditioning)
            if package.environment_conditioning is not None
            else None
        ),
        wardrobe=[asdict(item) for item in package.wardrobe_conditioning],
        props=[asdict(item) for item in package.prop_conditioning],
        continuity=asdict(package.continuity_constraints),
        performance={
            "performance_package_id": package.performance_package_id,
            "dialogue_timing": package.dialogue_timing,
            "facial_targets": package.facial_targets,
            "body_performance_tracks": package.body_performance_tracks,
            "gesture_tracks": package.gesture_tracks,
            "eye_lines": package.eye_lines,
            "capability_requirements": package.performance_capability_requirements,
        },
        approved_reference_ids=list(package.approved_reference_ids),
        deterministic_seed=package.deterministic_seed,
        renderer_requirements=asdict(package.renderer_capability_requirements),
        metadata={
            "source_conditioning_schema": package.schema_version,
            "source_conditioning_hash": package.content_hash,
            **dict(package.metadata),
        },
    )
    request.refresh_hash()
    return request
