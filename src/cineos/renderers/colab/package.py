from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cineos.mission_one.shot_package import DirectedShotPackage

from .config import ColabRenderConfig


@dataclass(slots=True)
class ColabRenderPackage:
    project_id: str
    scene_id: str
    shots: list[DirectedShotPackage]
    config: ColabRenderConfig
    package_version: str = "1.0"
    approved_reference_manifest: list[dict[str, Any]] = field(default_factory=list)
    dialogue_audio_manifest: list[dict[str, Any]] = field(default_factory=list)
    checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "package_version": self.package_version,
            "project_id": self.project_id,
            "scene_id": self.scene_id,
            "ordered_shot_packages": [s.to_dict() for s in self.shots],
            "prompts": [s.prompt for s in self.shots],
            "negative_prompts": [s.negative_prompt for s in self.shots],
            "seeds": [s.seed for s in self.shots],
            "frame_counts": [s.frame_count for s in self.shots],
            "fps": self.config.fps,
            "resolution": self.config.resolution,
            "model_id": self.config.model_id,
            "inference_steps": self.config.inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "approved_reference_manifest": self.approved_reference_manifest,
            "references_consumed_by_backend": False,
            "dialogue_audio_manifest": self.dialogue_audio_manifest,
            "expected_output_filenames": [s.expected_output for s in self.shots],
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, d):
        shots = [DirectedShotPackage(**x) for x in d["ordered_shot_packages"]]
        config = ColabRenderConfig(
            d["model_id"],
            d["fps"],
            d["resolution"],
            d["inference_steps"],
            d["guidance_scale"],
            d.get("seeds", [42])[0],
        )
        return cls(
            d["project_id"],
            d["scene_id"],
            shots,
            config,
            d.get("package_version", "1.0"),
            d.get("approved_reference_manifest", []),
            d.get("dialogue_audio_manifest", []),
            d.get("checksums", {}),
        )
