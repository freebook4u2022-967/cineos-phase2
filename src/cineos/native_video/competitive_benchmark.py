"""Connected hard-case benchmark for CINEOS native video renderers.

The suite deliberately exercises the failure modes that separate a renderer which
can emit clips from one that can sustain a complete film: identity persistence,
multi-character interaction, hands, locomotion, dialogue, prop interaction, fast
camera motion, lighting changes, physics, and long-range continuity.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.quality_retry import (
    QualityRetryPolicy,
    render_with_quality_retry,
)
from cineos.native_video.video_renderer import VideoRenderer

COMPETITIVE_BENCHMARK_SCHEMA = "cineos-native-video-competitive-benchmark/0.2"


class CompetitiveBenchmarkError(RuntimeError):
    """Raised when competitive benchmark evidence is incomplete or invalid."""


class VisualEvaluator(Protocol):
    def __call__(
        self,
        *,
        case: "BenchmarkCase",
        request: NativeShotRequest,
        output_path: Path,
        previous_output_path: Path | None,
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    shot_id: str
    prompt: str
    negative_prompt: str
    challenge_tags: tuple[str, ...]
    camera: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkShotEvidence:
    shot_id: str
    request_hash: str
    output_path: str
    output_sha256: str
    output_bytes: int
    elapsed_seconds: float
    metrics: Mapping[str, float]
    challenge_tags: tuple[str, ...]
    attempts: int = 1
    selected_attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "request_hash": self.request_hash,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": dict(self.metrics),
            "challenge_tags": list(self.challenge_tags),
            "attempts": self.attempts,
            "selected_attempt": self.selected_attempt,
        }


@dataclass(frozen=True, slots=True)
class CompetitiveBenchmarkReport:
    renderer_backend: str
    foundation: Mapping[str, Any]
    shot_evidence: tuple[BenchmarkShotEvidence, ...]
    evaluator_present: bool
    metric_thresholds: Mapping[str, float]
    failed_metrics: tuple[str, ...]
    aggregate_metrics: Mapping[str, float]
    production_passed: bool
    schema: str = COMPETITIVE_BENCHMARK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "renderer_backend": self.renderer_backend,
            "foundation": dict(self.foundation),
            "shot_evidence": [item.to_dict() for item in self.shot_evidence],
            "evaluator_present": self.evaluator_present,
            "metric_thresholds": dict(self.metric_thresholds),
            "failed_metrics": list(self.failed_metrics),
            "aggregate_metrics": dict(self.aggregate_metrics),
            "production_passed": self.production_passed,
        }
        payload["manifest_sha256"] = _canonical_sha256(payload)
        return payload


DEFAULT_METRIC_THRESHOLDS: dict[str, float] = {
    "identity_consistency": 0.82,
    "multi_character_interaction": 0.72,
    "hands_anatomy": 0.70,
    "locomotion": 0.72,
    "dialogue": 0.70,
    "object_interaction": 0.72,
    "fast_camera_movement": 0.68,
    "lighting_change": 0.70,
    "physics": 0.70,
    "long_range_continuity": 0.78,
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finite_metric(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompetitiveBenchmarkError(f"metric {name!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CompetitiveBenchmarkError(f"metric {name!r} must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise CompetitiveBenchmarkError(f"metric {name!r} must be in [0, 1]")
    return normalized


def default_connected_cases() -> tuple[BenchmarkCase, ...]:
    """Return the canonical ten-shot connected hard-case film benchmark."""

    return (
        BenchmarkCase(
            "shot-01-identity",
            "The same lead character enters a realistic corridor, turns toward camera, and pauses.",
            "identity drift, face morphing, age change, wardrobe change, temporal flicker",
            ("identity_consistency", "long_range_continuity"),
            {"shot_size": "medium", "movement": "slow dolly in"},
        ),
        BenchmarkCase(
            "shot-02-two-character",
            "The lead meets a second character; they exchange eye contact and move past each other naturally.",
            "merged bodies, duplicate limbs, identity swaps, frozen extras, temporal flicker",
            ("identity_consistency", "multi_character_interaction"),
            {"shot_size": "two-shot", "movement": "lateral track"},
        ),
        BenchmarkCase(
            "shot-03-hands-prop",
            "The lead picks up a small metal key, rotates it between both hands, and pockets it.",
            "extra fingers, fused fingers, disappearing prop, floating object, temporal flicker",
            ("hands_anatomy", "object_interaction", "physics"),
            {"shot_size": "close medium", "movement": "gentle handheld"},
        ),
        BenchmarkCase(
            "shot-04-walk",
            "The lead walks briskly down the corridor with natural full-body gait and cloth motion.",
            "foot sliding, limb warping, duplicated legs, frozen cloth, temporal flicker",
            ("locomotion", "identity_consistency", "physics"),
            {"shot_size": "full body", "movement": "backward tracking"},
        ),
        BenchmarkCase(
            "shot-05-run-camera",
            "The lead suddenly runs as the camera whip-pans and accelerates alongside them.",
            "motion smear face, broken anatomy, foot sliding, camera teleport, temporal flicker",
            ("locomotion", "fast_camera_movement", "identity_consistency"),
            {"shot_size": "full body", "movement": "whip pan into fast tracking"},
        ),
        BenchmarkCase(
            "shot-06-dialogue",
            "The lead stops, looks to the second character, and clearly says: Stay behind me.",
            "identity drift, frozen mouth, broken teeth, facial warping, temporal flicker",
            ("dialogue", "facial_performance", "identity_consistency"),
            {"shot_size": "medium close-up", "movement": "subtle push-in"},
        ),
        BenchmarkCase(
            "shot-07-lighting",
            "Emergency lights switch from neutral overhead light to pulsing red while the lead keeps moving.",
            "identity drift, exposure pumping, texture reset, scene teleport, temporal flicker",
            ("lighting_change", "identity_consistency", "long_range_continuity"),
            {"shot_size": "medium wide", "movement": "steady tracking"},
        ),
        BenchmarkCase(
            "shot-08-physics",
            "A rolling cart is struck, tips naturally, and scatters lightweight objects while both characters react.",
            "floating objects, clipping bodies, nonphysical motion, duplicate objects, temporal flicker",
            ("physics", "multi_character_interaction", "object_interaction"),
            {"shot_size": "wide", "movement": "reactive handheld"},
        ),
        BenchmarkCase(
            "shot-09-fast-orbit",
            "Camera performs a fast half-orbit around the lead as they dodge the fallen cart and keep running.",
            "identity drift, face collapse, broken limbs, background teleport, temporal flicker",
            ("fast_camera_movement", "locomotion", "identity_consistency"),
            {"shot_size": "medium full", "movement": "fast half orbit"},
        ),
        BenchmarkCase(
            "shot-10-continuity",
            "The lead reaches the exit, reveals the same metal key from shot three, and looks back toward the corridor.",
            "identity drift, prop substitution, wardrobe change, location reset, temporal flicker",
            (
                "identity_consistency",
                "hands_anatomy",
                "object_interaction",
                "long_range_continuity",
            ),
            {"shot_size": "medium", "movement": "slow push-in"},
        ),
    )


def _foundation_dict(renderer: VideoRenderer) -> dict[str, Any]:
    foundation = getattr(renderer, "foundation", None)
    if foundation is None:
        return {"model_id": "unknown", "provenance_declared": False}
    to_dict = getattr(foundation, "to_dict", None)
    if callable(to_dict):
        return {**dict(to_dict()), "provenance_declared": True}
    return {"model_id": str(foundation), "provenance_declared": True}


def _request_for_case(
    case: BenchmarkCase,
    *,
    index: int,
    previous_shot_id: str | None,
    scene_id: str,
    approved_reference_ids: tuple[str, ...],
    seed: int,
    resolution: tuple[int, int],
    fps: float,
    duration: float,
) -> NativeShotRequest:
    camera = {
        "resolution": resolution,
        "fps": fps,
        "duration": duration,
        **dict(case.camera),
    }
    performance: dict[str, Any] = {}
    if "dialogue" in case.challenge_tags:
        cue_start = max(0.0, duration * 0.10)
        cue_end = max(cue_start + 0.05, duration * 0.80)
        performance["dialogue_timing"] = [
            {
                "start_seconds": cue_start,
                "end_seconds": cue_end,
                "speaker_id": "benchmark-hero",
                "text": "Stay behind me.",
            }
        ]

    request = NativeShotRequest(
        shot_id=case.shot_id,
        scene_id=scene_id,
        camera=camera,
        characters=[
            {
                "character_id": "benchmark-hero",
                "identity_invariants": [
                    "same approved face in every shot",
                    "same hair, age, skin tone, and body proportions",
                ],
            }
        ],
        environment={
            "name": "benchmark corridor",
            "description": "same realistic interior corridor across all ten shots",
        },
        wardrobe=[{"character_id": "benchmark-hero", "description": "same dark coat"}],
        props=[{"prop_id": "metal-key", "continuity": "same key from shot 3 onward"}],
        continuity={
            "previous_shot_id": previous_shot_id,
            "scene_anchor": "same corridor geography and character identity",
            "challenge_tags": list(case.challenge_tags),
        },
        performance=performance,
        approved_reference_ids=list(approved_reference_ids),
        deterministic_seed=seed + index,
        renderer_requirements={
            "benchmark": {"require_real_artifact": True},
            "fps": fps,
            "duration_seconds": duration,
        },
        metadata={
            "prompt": case.prompt,
            "negative_prompt": case.negative_prompt,
            "benchmark_challenges": list(case.challenge_tags),
        },
    )
    request.refresh_hash()
    return request


def run_competitive_benchmark(
    renderer: VideoRenderer,
    *,
    approved_reference_ids: tuple[str, ...],
    evaluator: VisualEvaluator | None = None,
    cases: tuple[BenchmarkCase, ...] | None = None,
    scene_id: str = "cineos-competitive-connected-scene",
    seed: int = 20260929,
    resolution: tuple[int, int] = (832, 480),
    fps: float = 16.0,
    duration: float = 2.0,
    quality_retry_policy: Any | None = None,
) -> CompetitiveBenchmarkReport:
    """Render and measure a connected film benchmark against a real backend.

    A benchmark can record execution evidence without an evaluator, but it can
    never report ``production_passed=True`` in that state. This prevents a working
    GPU pipeline or a generated MP4 from being mislabeled as competitive quality.

    When ``quality_retry_policy`` is supplied, each shot is executed through the
    measured QC retry primitive. Failed renders or measured visual failures are
    rerendered without weakening identity, camera, scene, wardrobe, prop, or
    continuity constraints. The report records retry counts and the selected
    attempt so competitive evidence remains auditable.
    """

    benchmark_cases = tuple(cases or default_connected_cases())
    if not 5 <= len(benchmark_cases) <= 10:
        raise CompetitiveBenchmarkError(
            "competitive film benchmark requires between 5 and 10 connected shots"
        )
    if not approved_reference_ids:
        raise CompetitiveBenchmarkError(
            "competitive benchmark requires approved identity references"
        )

    thresholds = dict(DEFAULT_METRIC_THRESHOLDS)
    evidence: list[BenchmarkShotEvidence] = []
    aggregate: dict[str, list[float]] = {}
    previous_shot_id: str | None = None
    previous_output_path: Path | None = None

    for index, case in enumerate(benchmark_cases):
        request = _request_for_case(
            case,
            index=index,
            previous_shot_id=previous_shot_id,
            scene_id=scene_id,
            approved_reference_ids=approved_reference_ids,
            seed=seed,
            resolution=resolution,
            fps=fps,
            duration=duration,
        )
        started = time.monotonic()
        attempts = 1
        selected_attempt = 1

        if quality_retry_policy is None:
            result = renderer.render_video(request)
            output_path = Path(result.output_path)
            if not output_path.is_file():
                raise CompetitiveBenchmarkError(
                    f"renderer did not produce benchmark artifact: {output_path}"
                )
            if result.request_hash != request.content_hash:
                raise CompetitiveBenchmarkError(
                    f"renderer request hash mismatch for {case.shot_id}"
                )
            metrics = (
                dict(
                    evaluator(
                        case=case,
                        request=request,
                        output_path=output_path,
                        previous_output_path=previous_output_path,
                    )
                )
                if evaluator is not None
                else {}
            )
        else:
            if evaluator is None:
                raise CompetitiveBenchmarkError(
                    "quality retry benchmark requires a measured visual evaluator"
                )
            retry_policy = (
                quality_retry_policy
                if isinstance(quality_retry_policy, QualityRetryPolicy)
                else QualityRetryPolicy(**dict(quality_retry_policy))
            )

            def quality_evaluator(result: Any) -> Mapping[str, Any]:
                path = Path(result.output_path)
                measured = evaluator(
                    case=case,
                    request=request,
                    output_path=path,
                    previous_output_path=previous_output_path,
                )
                normalized = {
                    name: _finite_metric(value, name=name)
                    for name, value in measured.items()
                }
                relevant = [
                    normalized[tag]
                    for tag in case.challenge_tags
                    if tag in normalized and tag in thresholds
                ]
                accepted = bool(relevant) and all(
                    normalized[tag] >= thresholds[tag]
                    for tag in case.challenge_tags
                    if tag in normalized and tag in thresholds
                )
                return {
                    "accepted": accepted,
                    "metrics": normalized,
                    "failed_metrics": [
                        tag
                        for tag in case.challenge_tags
                        if tag in thresholds
                        and (
                            tag not in normalized or normalized[tag] < thresholds[tag]
                        )
                    ],
                }

            retry = render_with_quality_retry(
                request,
                renderer,
                quality_evaluator=quality_evaluator,
                policy=retry_policy,
            )
            result = retry.result
            output_path = Path(result.output_path)
            attempts = retry.attempts
            selected_attempt = retry.selected_attempt
            metrics = dict(retry.quality_report.get("metrics", {}))

        elapsed = max(0.0, time.monotonic() - started)
        payload = output_path.read_bytes()
        if not payload:
            raise CompetitiveBenchmarkError(
                f"benchmark artifact is empty: {output_path}"
            )
        output_sha = hashlib.sha256(payload).hexdigest()

        normalized_metrics: dict[str, float] = {}
        for metric_name, metric_value in metrics.items():
            normalized_metrics[metric_name] = _finite_metric(
                metric_value, name=metric_name
            )
            aggregate.setdefault(metric_name, []).append(
                normalized_metrics[metric_name]
            )

        evidence.append(
            BenchmarkShotEvidence(
                shot_id=case.shot_id,
                request_hash=request.content_hash,
                output_path=str(output_path),
                output_sha256=output_sha,
                output_bytes=len(payload),
                elapsed_seconds=elapsed,
                metrics=normalized_metrics,
                challenge_tags=case.challenge_tags,
                attempts=attempts,
                selected_attempt=selected_attempt,
            )
        )
        previous_shot_id = case.shot_id
        previous_output_path = output_path

    aggregate_metrics = {
        name: sum(values) / len(values) for name, values in aggregate.items() if values
    }
    failed_metrics = tuple(
        name
        for name, threshold in thresholds.items()
        if name not in aggregate_metrics or aggregate_metrics[name] < threshold
    )
    evaluator_present = evaluator is not None
    production_passed = evaluator_present and not failed_metrics

    return CompetitiveBenchmarkReport(
        renderer_backend=type(renderer).__name__,
        foundation=_foundation_dict(renderer),
        shot_evidence=tuple(evidence),
        evaluator_present=evaluator_present,
        metric_thresholds=thresholds,
        failed_metrics=failed_metrics,
        aggregate_metrics=aggregate_metrics,
        production_passed=production_passed,
    )


__all__ = [
    "COMPETITIVE_BENCHMARK_SCHEMA",
    "DEFAULT_METRIC_THRESHOLDS",
    "BenchmarkCase",
    "BenchmarkShotEvidence",
    "CompetitiveBenchmarkError",
    "CompetitiveBenchmarkReport",
    "VisualEvaluator",
    "default_connected_cases",
    "run_competitive_benchmark",
]
