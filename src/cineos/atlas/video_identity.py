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


def _temporal_bin(frame_index: int, frame_count: int, bin_count: int) -> int:
    """Map a sampled frame index into an evenly spaced temporal evidence bin."""

    return min(bin_count - 1, (frame_index * bin_count) // frame_count)


@dataclass(slots=True)
class EmbeddingBankVideoIdentitySource:
    """Measure rendered identity against approved multi-reference character anchors.

    Each character is scored independently across sampled frames. The per-character
    score uses a lower-tail quantile so a brief identity collapse cannot be hidden by
    otherwise strong frames. The shot score is the weakest character score, preventing
    a stable lead actor from masking drift in another visible character.

    Identity evidence must also cover a meaningful fraction of sampled frames and be
    distributed across the shot timeline. This prevents a long shot from passing on a
    cluster of clean detections near only the beginning or end while the character
    disappears, becomes untrackable, or collapses elsewhere. Integrations that
    intentionally preserve the legacy absolute-count-only contract can set
    ``minimum_observation_fraction`` to ``0.0`` explicitly; that also disables temporal
    distribution enforcement.

    For multi-character shots, every semantic observation must also discriminate its
    assigned identity from the other conditioned cast identities. An embedding that is
    closer (or nearly as close) to another cast member is scored as identity failure,
    preventing character swaps or identity collapse from receiving a strong aggregate
    identity score merely because both people resemble some approved cast anchor.
    """

    identity_bank: CharacterIdentityEmbeddingBank
    frame_encoder: CharacterFrameEncoder
    minimum_observations_per_character: int = 3
    lower_tail_quantile: float = 0.20
    minimum_cross_character_margin: float = 0.05
    minimum_observation_fraction: float = 0.40
    minimum_temporal_bins: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.identity_bank, CharacterIdentityEmbeddingBank):
            raise TypeError("identity_bank must be CharacterIdentityEmbeddingBank")
        if not callable(self.frame_encoder):
            raise TypeError("frame_encoder must be callable")
        if self.minimum_observations_per_character <= 0:
            raise ValueError("minimum_observations_per_character must be positive")
        if not 0.0 <= self.lower_tail_quantile <= 1.0:
            raise ValueError("lower_tail_quantile must be between 0 and 1")
        if not 0.0 <= self.minimum_cross_character_margin <= 2.0:
            raise ValueError("minimum_cross_character_margin must be between 0 and 2")
        if not 0.0 <= self.minimum_observation_fraction <= 1.0:
            raise ValueError("minimum_observation_fraction must be between 0 and 1")
        if not isinstance(self.minimum_temporal_bins, int) or isinstance(
            self.minimum_temporal_bins, bool
        ):
            raise TypeError("minimum_temporal_bins must be an integer")
        if self.minimum_temporal_bins <= 0:
            raise ValueError("minimum_temporal_bins must be positive")

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

        character_ids = _character_ids(shot)
        for character_id in character_ids:
            try:
                self.identity_bank.get(character_id)
            except KeyError as exc:
                raise VideoIdentityMetricError(
                    f"no approved identity anchor exists for character {character_id!r}"
                ) from exc

        required_observations = max(
            self.minimum_observations_per_character,
            math.ceil(len(frames) * self.minimum_observation_fraction),
        )
        required_temporal_bins = min(len(frames), self.minimum_temporal_bins)
        enforce_temporal_distribution = self.minimum_observation_fraction > 0.0
        scores_by_character: dict[str, list[float]] = {}
        for character_id in character_ids:
            observations: list[float] = []
            observed_temporal_bins: set[int] = set()
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
                    if len(character_ids) > 1:
                        impostor_similarity = max(
                            self.identity_bank.similarity(other_character_id, vector)
                            for other_character_id in character_ids
                            if other_character_id != character_id
                        )
                    else:
                        impostor_similarity = None
                except (KeyError, TypeError, ValueError) as exc:
                    raise VideoIdentityMetricError(
                        f"invalid identity embedding for character {character_id!r}"
                    ) from exc

                score = max(0.0, min(1.0, float(similarity)))
                if (
                    impostor_similarity is not None
                    and float(similarity) - float(impostor_similarity)
                    < self.minimum_cross_character_margin
                ):
                    score = 0.0
                observations.append(score)
                if enforce_temporal_distribution:
                    observed_temporal_bins.add(
                        _temporal_bin(
                            frame_index,
                            len(frames),
                            required_temporal_bins,
                        )
                    )

            if len(observations) < required_observations:
                raise VideoIdentityMetricError(
                    f"character {character_id!r} produced {len(observations)} semantic "
                    "identity observations; "
                    f"{required_observations} required across {len(frames)} sampled frames"
                )
            if enforce_temporal_distribution and len(observed_temporal_bins) < required_temporal_bins:
                raise VideoIdentityMetricError(
                    f"character {character_id!r} identity observations covered "
                    f"{len(observed_temporal_bins)} of {required_temporal_bins} required "
                    "temporal bins"
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
