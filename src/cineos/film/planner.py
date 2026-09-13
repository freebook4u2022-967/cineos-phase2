"""Deterministic timeline planning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


@dataclass(frozen=True, slots=True)
class PlannedShot:
    shot_id: str
    scene_id: str
    duration: float
    index: int
    payload: dict[str, Any] = field(default_factory=dict)


def plan_shots(package) -> list[PlannedShot]:
    by_id = {item["shot_id"]: item for item in package.shot_manifest}
    order = [
        shot_id
        for scene_id in package.timeline_manifest.get("scene_order", [])
        for shot_id in package.timeline_manifest.get("shot_order", {}).get(scene_id, [])
    ]
    if not order:
        order = [item["shot_id"] for item in package.shot_manifest]
    return [
        PlannedShot(
            item,
            str(by_id[item]["scene_id"]),
            float(by_id[item]["duration"]),
            index,
            dict(by_id[item]),
        )
        for index, item in enumerate(order)
    ]


def shot_plan_fingerprint(plan: list[PlannedShot]) -> str:
    """Return a deterministic identity for the complete planned shot timeline.

    Resume safety requires more than stable shot IDs: scene assignment, duration,
    ordering, and renderer-facing payload can all change the visual contract. The
    fingerprint therefore covers the canonicalized full plan and is persisted with
    film checkpoints before any output is reused.

    The function intentionally accepts PlannedShot-compatible objects so tests and
    downstream adapters can supply lightweight plan records without inheriting the
    concrete dataclass.
    """
    canonical = [
        {
            "shot_id": item.shot_id,
            "scene_id": item.scene_id,
            "duration": item.duration,
            "index": item.index,
            "payload": getattr(item, "payload", {}),
        }
        for item in plan
    ]
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
