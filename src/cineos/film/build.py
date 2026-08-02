"""Serializable state machine for a complete film build."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from .shot_state import ShotState


class BuildStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    COMPILING = "compiling"
    RENDERING = "rendering"
    VALIDATING_SHOTS = "validating_shots"
    RECOVERING = "recovering"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class FilmBuild:
    project_id: str
    film_package_id: str
    renderer_id: str
    build_id: str = field(default_factory=lambda: str(uuid4()))
    status: BuildStatus = BuildStatus.CREATED
    shot_states: list[ShotState] = field(default_factory=list)
    validation_states: list[dict[str, Any]] = field(default_factory=list)
    recovery_states: list[dict[str, Any]] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, status: BuildStatus | str) -> None:
        self.status = BuildStatus(status)
        self.updated_at = _now()
        if self.started_at is None and self.status != BuildStatus.CREATED:
            self.started_at = self.updated_at
        if self.status in {BuildStatus.COMPLETED, BuildStatus.COMPLETED_WITH_WARNINGS}:
            self.completed_at = self.updated_at

    @property
    def content_hash(self) -> str:
        """Hash reproducible inputs and decisions, excluding volatile fields."""
        value = asdict(self)
        for key in (
            "build_id",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
        ):
            value.pop(key, None)
        value["status"] = str(self.status)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def shot(self, shot_id: str) -> ShotState:
        return next(item for item in self.shot_states if item.shot_id == shot_id)

    def attach_audio(
        self, mixed_audio: str | None, metadata: dict[str, Any] | None = None
    ) -> None:
        """Attach a completed mix, preserving an explicit silent fallback."""
        if mixed_audio:
            self.output_files["mixed_audio"] = mixed_audio
        else:
            self.warnings.append("No mixed audio supplied; final assembly uses silence")
        self.metadata["audio"] = metadata or {"silent_fallback": not bool(mixed_audio)}
        self.updated_at = _now()
