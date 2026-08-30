"""Semantic video identity scoring backed by approved CINEOS identity anchors.

Low-level pixel metrics cannot establish that a rendered person is the approved
character. This module bridges the existing character identity embedding bank to
sampled rendered video frames through an injected semantic frame encoder. The
encoder may be a CINEOS-trained model or an explicitly declared external vision
foundation; this scorer never invents identity evidence from RGB statistics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from cineos.native_image.identity_bank import CharacterIdentityEmbeddingBank
from cineos.native_image.neural_decoder import DecodedRGBFrame


class VideoIdentityMetricError(RuntimeError):
    """Raised when semantic identity evidence is incomplete or malformed."""


class CharacterFrameEncoder(Protocol):
    """Encode one character observation from one sampled rendered frame."""

    def __call__(
        self,
        frame: DecodedRGBFrame,
        *,
        character_id: str,
        shot: Any,
        frame_index: int,
    ) -> Sequence[float] | None: ...


def _character_ids(shot: Any) -> tuple[str, ...]:
    characters = getattr(shot, "characters", None)
    if not isinstance(characters, list) or not characters:
        raise VideoIdentityMetricError(
            "video identity scoring requires shot.characters identity metadata"
        )

    result: list[str] = []
    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            raise VideoIdentityMetricError(
                f"shot character {index} must be a mapping with character_id"
            )
        raw = character.get("character_id")
        if not isinstance(raw, str) or not raw.strip():
            raise VideoIdentityMetricError(
                f"shot character {index} is missing character_id"
            )
        character_id = raw.strip()
        if character_id not in result:
            result.append(character_id)
    return tuple(result)


def _lower_tail_score(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise VideoIdentityMetricError("identity score aggregation received no values")
    index = min(len(ordered) - 1, max(0, math.floor((len(ordered) - 1) * quantile)))
    return ordered[index]


@dataclass(slots=True)
class EmbeddingBankVideoIdentitySource:
    """Measure rendered identity against approved multi-reference character anchors.

    Each character is scored independently across sampled frames. The per-character
    score uses a lower-tail quantile so a brief identity collapse cannot be hidden by
    otherwise strong frames. The shot score is the weakest character score, preventing
    a stable lead actor from masking drift in another visible character.
    """

    identity_bank: CharacterIdentityEmbeddingBank
    frame_encoder: CharacterFrameEncoder
    minimum_observations_per_character: int = 3
    lower_tail_quantile: float = 0.20

    def __post_init__(self) -> None:
        if not isinstance(self.identity_bank, CharacterIdentityEmbeddingBank):
            raise TypeError("identity_bank must be CharacterIdentityEmbeddingBank")
        if not callable(self.frame_encoder):
            raise TypeError("frame_encoder must be callable")
        if self.minimum_observations_per_character <= 0:
            raise ValueError("minimum_observations_per_character must be positive")
        if not 0.0 <= self.lower_tail_quantile <= 1.0:
            raise ValueError("lower_tail_quantile must be between 0 and 1")

    def __call__(
        self,
        output_path: str,
        *,
        shot: Any,
        frames: tuple[DecodedRGBFrame, ...],
        attempt_index: int,
    ) -> float:
        del output_path, attempt_index
        if not frames:
            raise VideoIdentityMetricError(
                "video identity scoring requires sampled frames"
            )

        scores_by_character: dict[str, list[float]] = {}
        for character_id in _character_ids(shot):
            try:
                self.identity_bank.get(character_id)
            except KeyError as exc:
                raise VideoIdentityMetricError(
                    f"no approved identity anchor exists for character {character_id!r}"
                ) from exc

            observations: list[float] = []
            for frame_index, frame in enumerate(frames):
                vector = self.frame_encoder(
                    frame,
                    character_id=character_id,
                    shot=shot,
                    frame_index=frame_index,
                )
                if vector is None:
                    continue
                try:
                    similarity = self.identity_bank.similarity(character_id, vector)
                except (TypeError, ValueError) as exc:
                    raise VideoIdentityMetricError(
                        f"invalid identity embedding for character {character_id!r}"
                    ) from exc
                observations.append(max(0.0, min(1.0, float(similarity))))

            if len(observations) < self.minimum_observations_per_character:
                raise VideoIdentityMetricError(
                    f"character {character_id!r} produced {len(observations)} semantic "
                    "identity observations; "
                    f"{self.minimum_observations_per_character} required"
                )
            scores_by_character[character_id] = observations

        if not scores_by_character:
            raise VideoIdentityMetricError("video identity scoring found no characters")

        character_scores = [
            _lower_tail_score(values, self.lower_tail_quantile)
            for values in scores_by_character.values()
        ]
        return min(character_scores)


__all__ = [
    "CharacterFrameEncoder",
    "EmbeddingBankVideoIdentitySource",
    "VideoIdentityMetricError",
]
