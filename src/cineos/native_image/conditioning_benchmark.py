"""Visual conditioning benchmark artifacts for CINEOS native image research."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .conditional_eval import ConditionalResponseEvaluator
from .latent_generation import ConditionalLatentGenerator


@dataclass(frozen=True, slots=True)
class ConditioningCell:
    character: str
    scene: str
    image_path: str


@dataclass(frozen=True, slots=True)
class ConditioningBenchmarkResult:
    cells: tuple[ConditioningCell, ...]
    same_character_scene_distance: float
    different_character_same_scene_distance: float
    identity_consistency_score: float
    report_path: str


def _save_pixels_ppm(pixels, path: Path, width: int, height: int) -> None:
    values = pixels.detach().float().reshape(-1).cpu()
    expected = width * height * 3
    if len(values) != expected:
        raise ValueError("generated pixel vector does not match benchmark dimensions")
    rgb = bytes(int(max(0.0, min(1.0, float(value))) * 255) for value in values)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb)


@dataclass(slots=True)
class ConditioningBenchmark:
    generator: ConditionalLatentGenerator
    width: int
    height: int

    def run(
        self,
        character_a: tuple[str | Path, ...],
        character_b: tuple[str | Path, ...],
        scene_a: tuple[str, str],
        scene_b: tuple[str, str],
        output_dir: str | Path,
        *,
        seed: int = 0,
    ) -> ConditioningBenchmarkResult:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        combinations = (
            ("A", "A", character_a, scene_a),
            ("A", "B", character_a, scene_b),
            ("B", "A", character_b, scene_a),
            ("B", "B", character_b, scene_b),
        )
        cells = []
        for character_name, scene_name, references, scene in combinations:
            result = self.generator.generate(
                references,
                scene[0],
                scene[1],
                seed=seed,
            )
            path = destination / f"character_{character_name}_scene_{scene_name}.ppm"
            _save_pixels_ppm(result.pixels, path, self.width, self.height)
            cells.append(ConditioningCell(character_name, scene_name, str(path)))

        metrics = ConditionalResponseEvaluator(self.generator).evaluate(
            character_a,
            character_b,
            scene_a=scene_a,
            scene_b=scene_b,
            seed=seed,
        )
        report_path = destination / "conditioning_benchmark.json"
        payload = {
            "cells": [asdict(cell) for cell in cells],
            "same_character_scene_distance": metrics.same_character_scene_distance,
            "different_character_same_scene_distance": (
                metrics.different_character_same_scene_distance
            ),
            "identity_consistency_score": metrics.identity_consistency_score,
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ConditioningBenchmarkResult(
            cells=tuple(cells),
            same_character_scene_distance=metrics.same_character_scene_distance,
            different_character_same_scene_distance=(
                metrics.different_character_same_scene_distance
            ),
            identity_consistency_score=metrics.identity_consistency_score,
            report_path=str(report_path),
        )
