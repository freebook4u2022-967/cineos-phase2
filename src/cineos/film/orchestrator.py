"""Backend-neutral complete film execution coordinator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

from .assembly import assemble
from .build import BuildStatus, FilmBuild
from .checkpoint import load_checkpoint_runtime_state, save_checkpoint
from .exceptions import BuildCancelled, FilmBuildError
from .planner import plan_shots
from .shot_state import ShotState
from .validator import ShotValidator, file_hash, validate_reusable_output

RuntimeStateProvider = Callable[[], dict[str, Any] | None]
RuntimeStateRestorer = Callable[[dict[str, Any]], None]


class FilmOrchestrator:
    """Render in timeline order with explicit validation and bounded recovery.

    The orchestrator stays renderer-neutral while exposing an optional runtime
    checkpoint boundary. Native renderers can use it to persist scene/temporal
    continuity memory alongside the durable ``FilmBuild`` without teaching the
    film layer model-specific tensor semantics.
    """

    def __init__(
        self,
        renderer: Any,
        validator: Any | None = None,
        *,
        max_recovery_attempts: int = 1,
        manual_review_on_failure: bool = False,
        checkpoint_state_provider: RuntimeStateProvider | None = None,
        checkpoint_state_restorer: RuntimeStateRestorer | None = None,
    ) -> None:
        self.renderer = renderer
        self.validator = validator or ShotValidator()
        self.max_recovery_attempts = max_recovery_attempts
        self.manual_review_on_failure = manual_review_on_failure
        self.checkpoint_state_provider = checkpoint_state_provider
        self.checkpoint_state_restorer = checkpoint_state_restorer
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()
        cancel = getattr(self.renderer, "cancel_pending", None)
        if cancel:
            cancel()

    def run(
        self,
        package: Any,
        build: FilmBuild,
        output_dir: str | Path,
        *,
        dry_run: bool = False,
        resume: bool = False,
        checkpoint_path: str | Path | None = None,
    ) -> FilmBuild:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None

        if (
            resume
            and checkpoint is not None
            and checkpoint.exists()
            and self.checkpoint_state_restorer is not None
        ):
            runtime_state = load_checkpoint_runtime_state(checkpoint)
            if runtime_state is not None:
                self.checkpoint_state_restorer(runtime_state)

        plan = plan_shots(package)
        existing = {state.shot_id: state for state in build.shot_states}
        build.shot_states = [
            existing.get(item.shot_id, ShotState(item.shot_id)) for item in plan
        ]
        self._checkpoint(build, checkpoint)
        if dry_run:
            build.metadata["dry_run"] = {
                "shot_count": len(plan),
                "output_dir": str(root),
                "renderer_compatible": True,
            }
            self._checkpoint(build, checkpoint)
            return build
        try:
            for item, state in zip(plan, build.shot_states, strict=True):
                if self.cancel_event.is_set():
                    raise BuildCancelled("build cancelled")
                if (
                    resume
                    and state.approved
                    and validate_reusable_output(
                        state.selected_output or "", state.output_hash
                    )
                ):
                    self._checkpoint(build, checkpoint)
                    continue
                self._render_shot(item, state, root, build, checkpoint)
                if not state.approved:
                    build.failures.append(f"{item.shot_id}: {state.failure_reason}")
                    build.transition(
                        BuildStatus.MANUAL_REVIEW_REQUIRED
                        if self.manual_review_on_failure
                        else BuildStatus.FAILED
                    )
                    self._checkpoint(build, checkpoint)
                    return build
            build.transition(BuildStatus.ASSEMBLING)
            self._checkpoint(build, checkpoint)
            movie = assemble(
                [
                    state.selected_output
                    for state in build.shot_states
                    if state.selected_output
                ],
                root / "final.mp4",
                durations=[item.duration for item in plan],
            )
            build.output_files["final_mp4"] = str(movie)
            build.transition(
                BuildStatus.COMPLETED_WITH_WARNINGS
                if build.warnings
                else BuildStatus.COMPLETED
            )
            self._checkpoint(build, checkpoint)
        except BuildCancelled:
            build.transition(BuildStatus.CANCELLED)
            self._checkpoint(build, checkpoint)
        except Exception as error:
            build.failures.append(str(error))
            build.transition(BuildStatus.FAILED)
            self._checkpoint(build, checkpoint)
        return build

    def _checkpoint(self, build: FilmBuild, checkpoint: Path | None) -> None:
        if checkpoint is None:
            return
        runtime_state = (
            self.checkpoint_state_provider()
            if self.checkpoint_state_provider is not None
            else None
        )
        save_checkpoint(build, checkpoint, runtime_state=runtime_state)

    def _render_shot(
        self,
        planned: Any,
        state: ShotState,
        root: Path,
        build: FilmBuild,
        checkpoint: Path | None = None,
    ) -> None:
        attempts = self.max_recovery_attempts + 1
        for attempt in range(attempts):
            build.transition(
                BuildStatus.RECOVERING if attempt else BuildStatus.RENDERING
            )
            state.attempt_count += 1
            self._checkpoint(build, checkpoint)
            started = monotonic()
            target = (
                root
                / "shots"
                / f"{planned.index:04d}-{planned.shot_id}-a{state.attempt_count}.mp4"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = self.renderer.render(planned, target)
                path = Path(result if isinstance(result, (str, Path)) else target)
                state.output_path = str(path)
                state.render_status = "completed"
                build.transition(BuildStatus.VALIDATING_SHOTS)
                report = self.validator.validate(path, planned)
                approved = (
                    bool(report.get("approved", False))
                    if isinstance(report, dict)
                    else bool(report)
                )
                build.validation_states.append(
                    {
                        "shot_id": planned.shot_id,
                        "attempt": state.attempt_count,
                        "approved": approved,
                    }
                )
                if approved:
                    state.validation_status = "approved"
                    state.selected_output = str(path)
                    state.output_hash = file_hash(path)
                    state.recovery_status = "recovered" if attempt else "not_required"
                    state.attempt_history.append(
                        {
                            "attempt": state.attempt_count,
                            "approved": True,
                            "output_hash": state.output_hash,
                        }
                    )
                    self._checkpoint(build, checkpoint)
                    return
                raise FilmBuildError("validator rejected output")
            except Exception as error:
                state.failure_reason = str(error)
                state.validation_status = "rejected"
                state.recovery_status = (
                    "retrying" if attempt + 1 < attempts else "exhausted"
                )
                state.attempt_history.append(
                    {
                        "attempt": state.attempt_count,
                        "approved": False,
                        "reason": str(error),
                    }
                )
                build.recovery_states.append(
                    {
                        "shot_id": planned.shot_id,
                        "attempt": state.attempt_count,
                        "reason": str(error),
                    }
                )
                self._checkpoint(build, checkpoint)
            finally:
                state.timing_metrics[f"attempt_{state.attempt_count}_seconds"] = (
                    monotonic() - started
                )
                self._checkpoint(build, checkpoint)
