"""Persistent continuity contracts for long-running short-drama production.

The ledger is renderer-independent.  It records the authoritative state that a
subsequent scene is allowed to inherit and validates proposed scene snapshots
before expensive rendering begins.  This keeps continuity logic deterministic,
serializable and suitable for resumable production jobs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from .models import SceneState


@dataclass(frozen=True)
class ContinuityViolation:
    """A single continuity contradiction detected between scene snapshots."""

    scene_index: int
    scope: str
    key: str
    previous: object
    proposed: object
    message: str


@dataclass
class ContinuityLedger:
    """Authoritative, append-only scene-state ledger.

    Validation is intentionally conservative: persistent character attributes
    (wardrobe, physical state and carried props) and persistent environment
    attributes must not change silently.  A caller can explicitly allow keys
    when a screenplay beat contains the corresponding transition.
    """

    scenes: list[SceneState] = field(default_factory=list)

    def latest(self) -> SceneState | None:
        return self.scenes[-1] if self.scenes else None

    def validate(
        self,
        proposed: SceneState,
        *,
        allowed_character_changes: Iterable[str] = (),
        allowed_environment_changes: Iterable[str] = (),
    ) -> list[ContinuityViolation]:
        previous = self.latest()
        if previous is None:
            return []
        if proposed.scene_index <= previous.scene_index:
            raise ValueError("scene_index must increase monotonically")

        allowed_character = set(allowed_character_changes)
        allowed_environment = set(allowed_environment_changes)
        violations: list[ContinuityViolation] = []

        persistent_character_keys = {"wardrobe", "physical_state", "props"}
        for character_id, current in proposed.characters.items():
            before = previous.characters.get(character_id)
            if before is None:
                continue
            for key in persistent_character_keys:
                token = f"{character_id}.{key}"
                if token in allowed_character or key in allowed_character:
                    continue
                old_value = before.get(key)
                new_value = current.get(key)
                if (
                    old_value is not None
                    and new_value is not None
                    and old_value != new_value
                ):
                    violations.append(
                        ContinuityViolation(
                            scene_index=proposed.scene_index,
                            scope=f"character:{character_id}",
                            key=key,
                            previous=old_value,
                            proposed=new_value,
                            message=f"{character_id} changed {key} without an explicit transition",
                        )
                    )

        for key, new_value in proposed.environment.items():
            if key in allowed_environment:
                continue
            old_value = previous.environment.get(key)
            if (
                old_value is not None
                and new_value is not None
                and old_value != new_value
            ):
                violations.append(
                    ContinuityViolation(
                        scene_index=proposed.scene_index,
                        scope="environment",
                        key=key,
                        previous=old_value,
                        proposed=new_value,
                        message=f"environment changed {key} without an explicit transition",
                    )
                )

        return violations

    def append(
        self,
        scene: SceneState,
        *,
        allowed_character_changes: Iterable[str] = (),
        allowed_environment_changes: Iterable[str] = (),
    ) -> None:
        violations = self.validate(
            scene,
            allowed_character_changes=allowed_character_changes,
            allowed_environment_changes=allowed_environment_changes,
        )
        if violations:
            messages = "; ".join(violation.message for violation in violations)
            raise ValueError(f"continuity validation failed: {messages}")
        self.scenes.append(scene)

    def to_dict(self) -> dict:
        """Return a stable JSON-safe representation for checkpoints/artifacts."""
        return {"version": 1, "scenes": [asdict(scene) for scene in self.scenes]}

    @classmethod
    def from_dict(cls, payload: dict) -> ContinuityLedger:
        """Restore a ledger from a versioned production checkpoint."""
        version = payload.get("version", 1)
        if version != 1:
            raise ValueError(f"unsupported continuity ledger version: {version}")
        scenes = [SceneState(**item) for item in payload.get("scenes", [])]
        ledger = cls()
        for scene in scenes:
            if ledger.scenes and scene.scene_index <= ledger.scenes[-1].scene_index:
                raise ValueError(
                    "checkpoint scene_index values must increase monotonically"
                )
            ledger.scenes.append(scene)
        return ledger
