"""Execution-first competitive benchmark for CINEOS video foundations.

The harness deliberately separates *execution evidence* from *visual-quality
evidence*. Producing a non-empty MP4 proves that a real renderer executed; it
does not prove Seedance-class quality. Visual quality only becomes validated
when an explicit evaluator is supplied and passes the rendered artifact.

The default suite is a connected ten-shot scene covering the failure modes that
matter most for complete-film generation: identity persistence, multiple
characters, hands/object interaction, locomotion, dialogue, fast camera motion,
lighting change, and simple physical motion.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from cineos.atlas.native_request import NativeShotRequest


class VideoRenderer(Protocol):
    """Minimal contract implemented by real CINEOS video execution backends."""

    def render(self, request: NativeShotRequest) -> Any: ...


VisualEvaluator = Callable[[Path, NativeShotRequest], "VisualEvaluation"]


@dataclass(frozen=True, slots=True)
class VisualEvaluation:
    """Externally measured visual-quality verdict for one rendered shot."""

    passed: bool
    metrics: Mapping[str, float]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in self.metrics.items():
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"visual metric {name!r} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One connected shot in the competitive film benchmark."""

    shot_id: str
    prompt: str
    challenge_tags: tuple[str, ...]
    camera: Mapping[str, Any]
    negative_prompt: str = (
        "identity drift, deformed hands, extra fingers, broken anatomy, "
        "warped objects, text, watermark, temporal flicker"
    )


@dataclass(frozen=True, slots=True)
class ShotBenchmarkResult:
    shot_id: str
    challenge_tags: tuple[str, ...]
    request_hash: str
    output_path: str | None
    artifact_bytes: int
    frame_count: int | None
    execution_passed: bool
    quality_evaluated: bool
    quality_passed: bool | None
    quality_metrics: Mapping[str, float]
    notes: tuple[str, ...]
    attempt_count: int = 1
    selected_attempt: int | None = 1
    rerendered: bool = False


@dataclass(frozen=True, slots=True)
class CompetitiveBenchmarkReport:
    """Auditable result for a connected multi-shot real-render benchmark."""

    scene_id: str
    foundation: Mapping[str, Any]
    shots: tuple[ShotBenchmarkResult, ...]

    @property
    def execution_passed(self) -> bool:
        return bool(self.shots) and all(item.execution_passed for item in self.shots)

    @property
    def quality_validated(self) -> bool:
        return bool(self.shots) and all(item.quality_evaluated for item in self.shots)

    @property
    def quality_passed(self) -> bool:
        return self.quality_validated and all(
            item.quality_passed for item in self.shots
        )

    @property
    def production_passed(self) -> bool:
        """Fail closed: real artifacts *and* measured visual quality are required."""
        return self.execution_passed and self.quality_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cineos-competitive-video-benchmark/0.1",
            "scene_id": self.scene_id,
            "foundation": dict(self.foundation),
            "execution_passed": self.execution_passed,
            "quality_validated": self.quality_validated,
            "quality_passed": self.quality_passed,
            "production_passed": self.production_passed,
            "shots": [asdict(item) for item in self.shots],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target


def default_connected_cases() -> tuple[BenchmarkCase, ...]:
    """Return the canonical ten-shot Seedance-style stress scene.

    These are deliberately connected shots rather than unrelated prompt samples.
    The same hero identity and scene id are reused by ``run_competitive_benchmark``.
    """

    return (
        BenchmarkCase(
            "shot-01-identity-closeup",
            (
                "Close-up of the hero under soft window light, subtle breathing "
                "and eye movement."
            ),
            ("identity_consistency", "facial_detail"),
            {"shot_size": "close-up", "movement": "locked"},
        ),
        BenchmarkCase(
            "shot-02-two-character",
            (
                "The hero crosses frame and meets a second character; both faces "
                "remain clearly visible."
            ),
            ("identity_consistency", "multi_character_interaction"),
            {"shot_size": "medium-wide", "movement": "slow dolly"},
        ),
        BenchmarkCase(
            "shot-03-object-handoff",
            (
                "The second character hands a metal key to the hero; fingers grasp "
                "and release it naturally."
            ),
            (
                "hands_anatomy",
                "object_interaction",
                "multi_character_interaction",
            ),
            {"shot_size": "medium", "movement": "subtle push-in"},
        ),
        BenchmarkCase(
            "shot-04-walk",
            (
                "The hero walks down the corridor holding the same key, with natural "
                "full-body gait."
            ),
            ("walking", "identity_consistency", "prop_continuity"),
            {"shot_size": "full", "movement": "tracking"},
        ),
        BenchmarkCase(
            "shot-05-run",
            (
                "A sudden alarm sounds and the hero runs toward camera without body "
                "deformation."
            ),
            ("running", "anatomy", "identity_consistency"),
            {"shot_size": "full", "movement": "backward tracking"},
        ),
        BenchmarkCase(
            "shot-06-dialogue",
            (
                "The hero stops and speaks one short urgent sentence to the second "
                "character."
            ),
            ("dialogue", "facial_performance", "identity_consistency"),
            {"shot_size": "medium close-up", "movement": "locked"},
        ),
        BenchmarkCase(
            "shot-07-fast-camera",
            (
                "Fast whip-pan follows the hero turning through a doorway, preserving "
                "geometry and identity."
            ),
            ("fast_camera_movement", "temporal_consistency"),
            {"shot_size": "medium", "movement": "whip pan"},
        ),
        BenchmarkCase(
            "shot-08-light-change",
            (
                "The hero enters a dark room as warm corridor light shifts to cool "
                "emergency lighting."
            ),
            ("lighting_change", "identity_consistency", "temporal_consistency"),
            {"shot_size": "medium-wide", "movement": "steadicam"},
        ),
        BenchmarkCase(
            "shot-09-physics",
            (
                "A gust from an open window moves the hero's coat and loose papers "
                "across the floor naturally."
            ),
            ("physics", "cloth_motion", "object_motion"),
            {"shot_size": "wide", "movement": "slow arc"},
        ),
        BenchmarkCase(
            "shot-10-continuity-payoff",
            (
                "The hero unlocks the final door with the same key and looks back "
                "toward the second character."
            ),
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
        wardrobe=[
            {"character_id": "benchmark-hero", "description": "same dark coat"}
        ],
        props=[
            {"prop_id": "metal-key", "continuity": "same key from shot 3 onward"}
        ],
        continuity={
            "previous_shot_id": previous_shot_id,
            "scene_anchor": "same corridor geography and character identity",
            "challenge_tags": list(case.challenge_tags),
        },
        performance={},
        approved_reference_ids=list(approved_reference_ids),
        deterministic_seed=seed + index,
        renderer_requirements={"benchmark": {"require_real_artifact": True}},
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

    if not approved_reference_ids:
        raise ValueError("competitive benchmark requires approved identity references")
    selected = cases or default_connected_cases()
    if not selected:
        raise ValueError("competitive benchmark requires at least one case")
    if quality_retry_policy is not None and evaluator is None:
        raise ValueError("quality retries require a visual evaluator")

    results: list[ShotBenchmarkResult] = []
    for index, case in enumerate(selected):
        previous_shot_id = None if index == 0 else selected[index - 1].shot_id
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
        notes: list[str] = []
        output_path: Path | None = None
        artifact_bytes = 0
        frame_count: int | None = None
        execution_passed = False
        quality_evaluated = False
        quality_passed: bool | None = None
        quality_metrics: Mapping[str, float] = {}
        request_hash = request.content_hash
        attempt_count = 1
        selected_attempt: int | None = 1
        rerendered = False

        if quality_retry_policy is not None and evaluator is not None:
            from .quality_retry import render_with_quality_retries

            retry_result = render_with_quality_retries(
                renderer,
                evaluator,
                request,
                policy=quality_retry_policy,
            )
            attempt_count = retry_result.attempt_count
            selected_attempt = retry_result.selected_attempt
            rerendered = retry_result.rerendered
            selected_evidence = next(
                (
                    item
                    for item in retry_result.attempts
                    if item.attempt == selected_attempt
                ),
                None,
            )
            if selected_evidence is not None:
                request_hash = selected_evidence.request_hash
                if selected_evidence.output_path is not None:
                    output_path = Path(selected_evidence.output_path)
                artifact_bytes = selected_evidence.artifact_bytes
                frame_count = selected_evidence.frame_count
                execution_passed = selected_evidence.execution_passed
                quality_evaluated = selected_evidence.quality_evaluated
                quality_passed = selected_evidence.quality_passed
                quality_metrics = dict(selected_evidence.quality_metrics)
                notes.extend(selected_evidence.notes)
            notes.append(
                "measured quality retry attempts="
                f"{attempt_count}, selected_attempt={selected_attempt}"
            )
        else:
            try:
                rendered = renderer.render(request)
                raw_path = getattr(rendered, "output_path", rendered)
                output_path = Path(raw_path)
                frame_count_value = getattr(rendered, "frame_count", None)
                if frame_count_value is not None:
                    frame_count = int(frame_count_value)
                if output_path.is_file():
                    artifact_bytes = output_path.stat().st_size
                    execution_passed = artifact_bytes > 0
                if not execution_passed:
                    notes.append("renderer did not produce a non-empty artifact")
            except Exception as exc:  # benchmark must record failure and continue suite
                notes.append(f"render failed: {type(exc).__name__}: {exc}")

            if execution_passed and evaluator is not None and output_path is not None:
                try:
                    evaluation = evaluator(output_path, request)
                    quality_evaluated = True
                    quality_passed = bool(evaluation.passed)
                    quality_metrics = dict(evaluation.metrics)
                    notes.extend(evaluation.notes)
                except Exception as exc:
                    notes.append(
                        f"visual evaluation failed: {type(exc).__name__}: {exc}"
                    )

        results.append(
            ShotBenchmarkResult(
                shot_id=case.shot_id,
                challenge_tags=case.challenge_tags,
                request_hash=request_hash,
                output_path=str(output_path) if output_path is not None else None,
                artifact_bytes=artifact_bytes,
                frame_count=frame_count,
                execution_passed=execution_passed,
                quality_evaluated=quality_evaluated,
                quality_passed=quality_passed,
                quality_metrics=quality_metrics,
                notes=tuple(notes),
                attempt_count=attempt_count,
                selected_attempt=selected_attempt,
                rerendered=rerendered,
            )
        )

    return CompetitiveBenchmarkReport(
        scene_id=scene_id,
        foundation=_foundation_dict(renderer),
        shots=tuple(results),
    )
