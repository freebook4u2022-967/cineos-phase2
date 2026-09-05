"""Deterministic CINEOS-native latent frame prototype.

This module produces real RGB pixel frames without third-party model APIs. It is
not a trained generative model. Its purpose is to establish the owned latent
state, encoder, sampler and decoder contracts that future learned weights will
replace without changing the surrounding CINEOS runtime.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .conditioning import NativeImageConditioningPlan


class TrainableLatentComponents(Protocol):
    """Future learned implementation boundary for CINEOS-owned weights."""

    def encode_identity(self, tokens: list[dict[str, Any]]) -> tuple[float, ...]: ...

    def encode_scene(self, plan: NativeImageConditioningPlan) -> tuple[float, ...]: ...

    def sample_latent(
        self,
        identity: tuple[float, ...],
        scene: tuple[float, ...],
        *,
        seed: int,
    ) -> tuple[float, ...]: ...

    def decode_rgb(
        self,
        latent: tuple[float, ...],
        *,
        width: int,
        height: int,
    ) -> bytes: ...


@dataclass(frozen=True, slots=True)
class NativePixelFrame:
    """A concrete RGB frame produced entirely inside the CINEOS native path."""

    width: int
    height: int
    rgb: bytes
    seed: int
    latent: tuple[float, ...]
    prototype: bool = True

    def __post_init__(self) -> None:
        expected = self.width * self.height * 3
        if len(self.rgb) != expected:
            raise ValueError(
                f"RGB payload has {len(self.rgb)} bytes; expected {expected}"
            )

    def save_ppm(self, path: str | Path) -> Path:
        """Persist the frame using the dependency-free binary PPM format."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        destination.write_bytes(header + self.rgb)
        return destination


class ProceduralLatentComponents:
    """Reference latent implementation used until learned CINEOS weights exist."""

    latent_dimensions = 16

    @staticmethod
    def _digest_vector(payload: str, dimensions: int) -> tuple[float, ...]:
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        values = []
        for index in range(dimensions):
            raw = digest[index % len(digest)] / 255.0
            values.append((raw * 2.0) - 1.0)
        return tuple(values)

    def encode_identity(self, tokens: list[dict[str, Any]]) -> tuple[float, ...]:
        if not tokens:
            raise ValueError("latent identity encoder requires identity tokens")
        payload = "|".join(
            f"{item.get('character_uuid')}:{item.get('primary_reference_id')}"
            for item in tokens
        )
        return self._digest_vector(payload, self.latent_dimensions)

    def encode_scene(self, plan: NativeImageConditioningPlan) -> tuple[float, ...]:
        payload = (
            f"{plan.shot_id}|{plan.scene_id}|{plan.composition_tokens}|"
            f"{plan.environment_tokens}|{plan.performance_tokens}"
        )
        return self._digest_vector(payload, self.latent_dimensions)

    def sample_latent(
        self,
        identity: tuple[float, ...],
        scene: tuple[float, ...],
        *,
        seed: int,
    ) -> tuple[float, ...]:
        if len(identity) != len(scene):
            raise ValueError("identity and scene latent dimensions must match")
        rng = random.Random(seed)
        values = []
        for index in range(len(identity)):
            mixed = (
                (identity[index] * 0.6) + (scene[index] * 0.4) + rng.uniform(-0.1, 0.1)
            )
            values.append(math.tanh(mixed))
        return tuple(values)

    def decode_rgb(
        self,
        latent: tuple[float, ...],
        *,
        width: int,
        height: int,
    ) -> bytes:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not latent:
            raise ValueError("decoder requires a latent vector")
        pixels = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                position = (y * width + x) * 3
                nx = x / max(1, width - 1)
                ny = y / max(1, height - 1)
                base = latent[(x + y) % len(latent)]
                pixels[position] = int(255 * ((math.sin(base + nx * 3.0) + 1.0) / 2.0))
                pixels[position + 1] = int(
                    255 * ((math.sin(base + ny * 4.0 + 1.2) + 1.0) / 2.0)
                )
                pixels[position + 2] = int(
                    255 * ((math.sin(base + (nx + ny) * 2.0 + 2.4) + 1.0) / 2.0)
                )
        return bytes(pixels)


class CineosLatentFrameModel:
    """NativeImageModel-compatible CINEOS frame prototype.

    Default output is capped for fast research execution. A future learned
    TrainableLatentComponents implementation can be injected without changing
    NativeFrameRuntime or the short-drama pipeline.
    """

    def __init__(
        self,
        components: TrainableLatentComponents | None = None,
        *,
        max_dimension: int = 128,
    ) -> None:
        if max_dimension < 16:
            raise ValueError("max_dimension must be at least 16")
        self.components = components or ProceduralLatentComponents()
        self.max_dimension = max_dimension
        self._plan: NativeImageConditioningPlan | None = None

    def encode_identity(self, tokens: list[dict[str, Any]]) -> tuple[float, ...]:
        return self.components.encode_identity(tokens)

    def encode_scene(self, plan: NativeImageConditioningPlan) -> tuple[float, ...]:
        self._plan = plan
        return self.components.encode_scene(plan)

    def generate(
        self,
        *,
        identity_state: tuple[float, ...],
        scene_state: tuple[float, ...],
        seed: int,
    ) -> NativePixelFrame:
        if self._plan is None:
            raise RuntimeError("encode_scene must run before generate")
        latent = self.components.sample_latent(
            identity_state,
            scene_state,
            seed=seed,
        )
        scale = min(
            1.0,
            self.max_dimension / max(self._plan.width, self._plan.height),
        )
        width = max(1, round(self._plan.width * scale))
        height = max(1, round(self._plan.height * scale))
        rgb = self.components.decode_rgb(latent, width=width, height=height)
        return NativePixelFrame(
            width=width,
            height=height,
            rgb=rgb,
            seed=seed,
            latent=latent,
        )
