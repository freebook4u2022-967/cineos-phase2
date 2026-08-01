"""Deterministic timeline planning."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlannedShot:
    shot_id: str
    scene_id: str
    duration: float
    index: int


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
            item, str(by_id[item]["scene_id"]), float(by_id[item]["duration"]), index
        )
        for index, item in enumerate(order)
    ]
