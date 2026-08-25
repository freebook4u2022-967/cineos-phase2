"""Backend-neutral complete film execution coordinator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

from .assembly import assemble
from .build import BuildStatus, FilmBuild
from .checkpoint import load_checkpoint_bundle, save_checkpoint
from .exceptions import BuildCancelled, FilmBuildError
from .planner import plan_shots, shot_plan_fingerprint
from .shot_state import ShotState
from .validator import ShotValidator, file_hash, validate_reusable_output

RuntimeStateProvider = Callable[[], dict[str, Any] | None]
RuntimeStateRestorer = Callable[[dict[str, Any]], None]
RuntimeStateResetter = Callable[[], None]
ShotAttemptCallback = Callable[[Any, int, int], None]


class FilmOrchestrator:
    """Render in timeline order with explicit validation and bounded recovery.

    The orchestrator stays renderer-neutral while exposing optional runtime
    checkpoint and shot-attempt lifecycle boundaries. Native renderers can use
    these hooks to persist scene/temporal continuity memory alongside the durable
    ``FilmBuild`` without teaching the film layer model-specific tensor semantics.

    Attempt lifecycle callbacks are deliberately transactional: a runtime receives
    ``shot_attempt_start`` before each render attempt, ``shot_attempt_accepted``
    only after whole-shot validation and final artifact hashing succeed, and
    ``shot_attempt_rejected`` for any failed attempt. This prevents rejected or
    incomplete whole-shot retries from advancing long-range continuity state.

    Stateful resume is integrity-aware. The persisted ``FilmBuild`` and optional
    renderer runtime state are loaded from the same checkpoint document, validated
    against the requested build identity/creative contract and the deterministic
    shot-plan fingerprint, and only then reused. If the persisted shot timeline is
    no longer a contiguous reusable prefix (for example, an earlier approved
    artifact was deleted or corrupted), the orchestrator must not restore
    continuity memory that may already contain later-shot state. With a runtime
    reset hook it safely restarts the timeline from shot zero; without one it fails
    closed.
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
        checkpoint_state_resetter: RuntimeStateResetter | None = None,
        shot_attempt_start: ShotAttemptCallback | None = None,
        shot_attempt_accepted: ShotAttemptCallback | None = None,
        shot_attempt_rejected: ShotAttemptCallback | None = None,
    ) -> None:
        self.renderer = renderer
        self.validator = validator or ShotValidator()
        self.max_recovery_attempts = max_recovery_attempts
        self.manual_review_on_failure = manual_review_on_failure
        self.checkpoint_state_provider = checkpoint_state_provider
        self.checkpoint_state_restorer = checkpoint_state_restorer
        self.checkpoint_state_resetter = checkpoint_state_resetter
        self.shot_attempt_start = shot_attempt_start
        self.shot_attempt_accepted = shot_attempt_accepted
        self.shot_attempt_rejected = shot_attempt_rejected
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
        runtime_state = None
        if resume and checkpoint is not None and checkpoint.exists():
            saved_build, runtime_state = load_checkpoint_bundle(checkpoint)
            self._assert_resume_build_compatible(build, saved_build)
            build = saved_build

        plan = plan_shots(package)
        plan_fingerprint = shot_plan_fingerprint(plan)
        if resume:
            self._assert_resume_plan_compatible(build, plan, plan_fingerprint)
        build.metadata["shot_plan_fingerprint"] = plan_fingerprint

        scene_indices: dict[str, int] = {}
        for item in plan:
            scene_indices.setdefault(item.scene_id, len(scene_indices))

        existing = {state.shot_id: state for state in build.shot_states}
        build.shot_states = [
            existing.get(item.shot_id, ShotState(item.shot_id)) for item in plan
        ]

        if resume and runtime_state is not None:
            if self.checkpoint_state_restorer is None:
                raise FilmBuildError(
                    "stateful resume checkpoint contains runtime state but no runtime "
                    "state restorer is configured"
                )
            self._restore_runtime_for_resume(build, runtime_state, dry_run=dry_run)

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
                self._render_shot(
                    item,
                    state,
                    root,
                    build,
                    checkpoint,
                    scene_index=scene_indices[item.scene_id],
                )
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

    @staticmethod
    def _assert_resume_build_compatible(requested: FilmBuild, saved: FilmBuild) -> None:
        """Fail closed before reusing persisted work from a different build intent."""
        mismatches = [
            name
            for name in ("project_id", "film_package_id", "renderer_id")
            if getattr(requested, name) != getattr(saved, name)
        ]
        requested_contract = requested.metadata.get("resume_contract")
        saved_contract = saved.metadata.get("resume_contract")
        if requested_contract is not None and requested_contract != saved_contract:
            mismatches.append("resume_contract")
        if mismatches:
            raise FilmBuildError(
                "resume checkpoint is incompatible with requested build: "
                + ", ".join(mismatches)
            )

    @staticmethod
    def _assert_resume_plan_compatible(
        saved: FilmBuild, plan: list[Any], current_fingerprint: str
    ) -> None:
        """Prevent checkpoint reuse across a changed creative timeline.

        New checkpoints carry a fingerprint of the full renderer-facing plan. For
        legacy checkpoints that predate the fingerprint, preserve compatibility
        only when the persisted shot-state order exactly matches the current shot
        IDs. This catches reorders/additions/removals while allowing old checkpoints
        to be upgraded on their next successful checkpoint write.
        """
        saved_fingerprint = saved.metadata.get("shot_plan_fingerprint")
        if saved_fingerprint is not None and saved_fingerprint != current_fingerprint:
            raise FilmBuildError(
                "resume checkpoint shot plan differs from the current creative "
                "timeline; start a fresh build instead of reusing prior artifacts"
            )

        if saved_fingerprint is None and saved.shot_states:
            saved_ids = [state.shot_id for state in saved.shot_states]
            planned_ids = [item.shot_id for item in plan]
            if saved_ids != planned_ids:
                raise FilmBuildError(
                    "legacy resume checkpoint shot order differs from the current "
                    "creative timeline; start a fresh build instead of reusing "
                    "prior artifacts"
                )

    def _restore_runtime_for_resume(
        self,
        build: FilmBuild,
        runtime_state: dict[str, Any],
        *,
        dry_run: bool,
    ) -> None:
        """Restore stateful runtime only when reusable shots form a safe prefix.

        Continuity runtimes are order-dependent. Restoring a checkpoint that has
        accepted later shots and then rerendering an earlier missing/corrupt shot
        would seed that earlier shot from future recurrent state. We therefore
        require approved reusable outputs to form one contiguous prefix. If not,
        reset both runtime and shot state so the timeline is regenerated from the
        beginning. Dry runs preserve the historical restore-only behavior because
        they never render or advance continuity state.
        """
        if dry_run:
            self.checkpoint_state_restorer(runtime_state)
            return

        saw_gap = False
        unsafe = False
        approved_count = 0
        reusable_count = 0
        for state in build.shot_states:
            if not state.approved:
                saw_gap = True
                continue
            approved_count += 1
            reusable = validate_reusable_output(
                state.selected_output or "", state.output_hash
            )
            if reusable and not saw_gap:
                reusable_count += 1
                continue
            unsafe = True
            saw_gap = True

        if approved_count == 0:
            # A runtime checkpoint paired with no accepted shot state is internally
            # inconsistent for a stateful production resume. Start clean rather
            # than guessing what recurrent state the checkpoint represents.
            unsafe = True

        if unsafe:
            if self.checkpoint_state_resetter is None:
                raise FilmBuildError(
                    "stateful resume checkpoint is not a contiguous reusable shot "
                    "prefix and no runtime reset hook is configured"
                )
            self.checkpoint_state_resetter()
            build.warnings.append(
                "Stateful resume integrity mismatch; continuity runtime was reset "
                "and the shot timeline will be regenerated from the beginning"
            )
            build.metadata["resume_integrity"] = {
                "action": "full_timeline_regeneration",
                "approved_shots": approved_count,
                "reusable_prefix_shots": reusable_count,
            }
            build.shot_states = [
                ShotState(state.shot_id) for state in build.shot_states
            ]
            return

        self.checkpoint_state_restorer(runtime_state)
        build.metadata["resume_integrity"] = {
            "action": "restored_contiguous_prefix",
            "approved_shots": approved_count,
            "reusable_prefix_shots": reusable_count,
        }

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
        *,
        scene_index: int,
    ) -> None:
        attempts = self.max_recovery_attempts + 1
        for attempt in range(attempts):
            build.transition(
                BuildStatus.RECOVERING if attempt else BuildStatus.RENDERING
            )
            state.attempt_count += 1
            attempt_number = state.attempt_count
            if self.shot_attempt_start is not None:
                self.shot_attempt_start(planned, scene_index, attempt_number)
            self._checkpoint(build, checkpoint)
            started = monotonic()
            target = (
                root
                / "shots"
                / f"{planned.index:04d}-{planned.shot_id}-a{attempt_number}.mp4"
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
                        "attempt": attempt_number,
                        "approved": approved,
                    }
                )
                if approved:
                    # Hash the exact artifact before committing model-specific
                    # continuity state. Hashing can fail if the renderer returned a
                    # missing/unreadable/transient artifact; such an attempt must be
                    # rejected rather than promoted into durable scene memory.
                    output_hash = file_hash(path)
                    if self.shot_attempt_accepted is not None:
                        self.shot_attempt_accepted(planned, scene_index, attempt_number)
                    state.validation_status = "approved"
                    state.selected_output = str(path)
                    state.output_hash = output_hash
                    state.recovery_status = "recovered" if attempt else "not_required"
                    state.failure_reason = None
                    state.attempt_history.append(
                        {
                            "attempt": attempt_number,
                            "approved": True,
                            "output_hash": state.output_hash,
                        }
                    )
                    self._checkpoint(build, checkpoint)
                    return
                raise FilmBuildError("validator rejected output")
            except Exception as error:
                if self.shot_attempt_rejected is not None:
                    try:
                        self.shot_attempt_rejected(planned, scene_index, attempt_number)
                    except Exception as rollback_error:
                        error = FilmBuildError(
                            f"{error}; runtime rollback failed: {rollback_error}"
                        )
                state.failure_reason = str(error)
                state.validation_status = "rejected"
                state.recovery_status = (
                    "retrying" if attempt + 1 < attempts else "exhausted"
                )
                state.attempt_history.append(
                    {
                        "attempt": attempt_number,
                        "approved": False,
                        "reason": str(error),
                    }
                )
                build.recovery_states.append(
                    {
                        "shot_id": planned.shot_id,
                        "attempt": attempt_number,
                        "reason": str(error),
                    }
                )
                self._checkpoint(build, checkpoint)
            finally:
                state.timing_metrics[f"attempt_{attempt_number}_seconds"] = (
                    monotonic() - started
                )
                self._checkpoint(build, checkpoint)
