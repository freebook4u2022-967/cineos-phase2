"""Renderer-independent orchestration for versioned Film Packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from cineos.compiler import FilmPackage, verify


class RuntimeState(StrEnum):
    """Lifecycle states for an Atlas runtime job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RuntimeStateError(RuntimeError):
    """Raised when an operation is not valid for a job's current state."""


@dataclass(frozen=True, slots=True)
class RuntimeTask:
    """A single shot scheduled from a Film Package timeline."""

    scene_id: str
    shot_id: str
    scene: Mapping[str, Any]
    shot: Mapping[str, Any]


@dataclass(slots=True)
class RuntimeJob:
    """Observable state for one execution of a Film Package."""

    package: FilmPackage
    tasks: tuple[RuntimeTask, ...]
    job_id: str = field(default_factory=lambda: uuid4().hex)
    state: RuntimeState = RuntimeState.PENDING
    completed: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    error: BaseException | None = None
    _cancel_requested: bool = field(default=False, init=False, repr=False)

    @property
    def progress(self) -> float:
        """Return completed work as a value from zero through one."""

        if not self.tasks:
            return 1.0 if self.state is RuntimeState.COMPLETED else 0.0
        return len(self.completed) / len(self.tasks)

    def cancel(self) -> None:
        """Cancel pending work or request cancellation of a running job."""

        if self.state in {
            RuntimeState.COMPLETED,
            RuntimeState.FAILED,
            RuntimeState.CANCELLED,
        }:
            raise RuntimeStateError(f"cannot cancel a {self.state.value} job")
        self._cancel_requested = True
        if self.state is RuntimeState.PENDING:
            self.state = RuntimeState.CANCELLED


TaskHandler = Callable[[RuntimeTask], Any]


class AtlasRuntime:
    """Validate, schedule, and dispatch package tasks to injected application code.

    The runtime owns orchestration only. The task handler is an integration
    boundary: Atlas neither supplies a renderer nor interprets handler results.
    """

    def prepare(self, package: FilmPackage, *, job_id: str | None = None) -> RuntimeJob:
        """Validate *package* and build tasks in canonical timeline order."""

        if not isinstance(package, FilmPackage):
            raise TypeError("package must be a FilmPackage")
        verify(package)
        scenes = {scene["scene_id"]: scene for scene in package.scene_manifest}
        shots = {shot["shot_id"]: shot for shot in package.shot_manifest}
        tasks = tuple(
            RuntimeTask(
                scene_id=scene_id,
                shot_id=shot_id,
                scene=MappingProxyType(scenes[scene_id]),
                shot=MappingProxyType(shots[shot_id]),
            )
            for scene_id in package.timeline_manifest["scene_order"]
            for shot_id in package.timeline_manifest["shot_order"][scene_id]
        )
        return RuntimeJob(
            package=package, tasks=tasks, **({"job_id": job_id} if job_id else {})
        )

    def run(self, job: RuntimeJob, handler: TaskHandler) -> RuntimeJob:
        """Synchronously dispatch every remaining task and update *job*."""

        if not isinstance(job, RuntimeJob):
            raise TypeError("job must be a RuntimeJob")
        if job.state is not RuntimeState.PENDING:
            raise RuntimeStateError(f"cannot run a {job.state.value} job")
        if not callable(handler):
            raise TypeError("handler must be callable")

        job.state = RuntimeState.RUNNING
        try:
            for task in job.tasks:
                if job._cancel_requested:
                    job.state = RuntimeState.CANCELLED
                    return job
                job.results[task.shot_id] = handler(task)
                job.completed.append(task.shot_id)
        except BaseException as error:
            job.error = error
            job.state = RuntimeState.FAILED
            raise
        job.state = (
            RuntimeState.CANCELLED if job._cancel_requested else RuntimeState.COMPLETED
        )
        return job

    def execute(
        self,
        package: FilmPackage,
        handler: TaskHandler,
        *,
        job_id: str | None = None,
    ) -> RuntimeJob:
        """Prepare and immediately run a package."""

        return self.run(self.prepare(package, job_id=job_id), handler)
