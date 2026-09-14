import pytest

from cineos.atlas.video_identity import (
    EmbeddingBankVideoIdentitySource,
    VideoIdentityMetricError,
)
from cineos.native_image.identity_bank import CharacterIdentityEmbeddingBank
from cineos.native_image.neural_decoder import DecodedRGBFrame


class Shot:
    def __init__(self, *, challenge: bool):
        self.characters = [
            {"character_id": "lead"},
            {"character_id": "partner"},
        ]
        self.metadata = (
            {"benchmark_challenges": ["multi_character_interaction"]}
            if challenge
            else {}
        )


def _frames(count: int = 6) -> tuple[DecodedRGBFrame, ...]:
    return tuple(
        DecodedRGBFrame(width=1, height=1, rgb=bytes((index, index, index)))
        for index in range(count)
    )


def _bank() -> CharacterIdentityEmbeddingBank:
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("lead", [(1.0, 0.0)])
    bank.build_character("partner", [(0.0, 1.0)])
    return bank


def _source(observed_by_character: dict[str, set[int]]):
    def encode(_frame, *, character_id, frame_index, **_kwargs):
        if frame_index not in observed_by_character[character_id]:
            return None
        return (1.0, 0.0) if character_id == "lead" else (0.0, 1.0)

    return EmbeddingBankVideoIdentitySource(
        identity_bank=_bank(),
        frame_encoder=encode,
        minimum_observations_per_character=3,
        minimum_observation_fraction=0.0,
    )


def test_competitive_multi_character_interaction_rejects_disjoint_solo_appearances():
    source = _source(
        {
            "lead": {0, 1, 2},
            "partner": {3, 4, 5},
        }
    )

    with pytest.raises(
        VideoIdentityMetricError,
        match="multi_character_interaction requires at least two distinct cast identities",
    ):
        source(
            "candidate.mp4",
            shot=Shot(challenge=True),
            frames=_frames(),
            attempt_index=0,
        )


def test_competitive_multi_character_interaction_accepts_same_frame_copresence():
    source = _source(
        {
            "lead": {0, 1, 2},
            "partner": {2, 3, 4},
        }
    )

    assert (
        source(
            "candidate.mp4",
            shot=Shot(challenge=True),
            frames=_frames(),
            attempt_index=0,
        )
        > 0.99
    )


def test_generic_multi_character_shot_keeps_shot_reverse_shot_compatibility():
    source = _source(
        {
            "lead": {0, 1, 2},
            "partner": {3, 4, 5},
        }
    )

    assert (
        source(
            "candidate.mp4",
            shot=Shot(challenge=False),
            frames=_frames(),
            attempt_index=0,
        )
        > 0.99
    )
