import pytest

from cineos.atlas.video_identity import (
    EmbeddingBankVideoIdentitySource,
    VideoIdentityMetricError,
)
from cineos.native_image.identity_bank import CharacterIdentityEmbeddingBank
from cineos.native_image.neural_decoder import DecodedRGBFrame


class Shot:
    def __init__(self, character_ids):
        self.characters = [{"character_id": item} for item in character_ids]


def _frames(count=5):
    return tuple(
        DecodedRGBFrame(width=1, height=1, rgb=bytes((index, index, index)))
        for index in range(count)
    )


def test_identity_source_scores_against_approved_bank_anchor():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("lead", [(1.0, 0.0), (0.99, 0.01)])

    source = EmbeddingBankVideoIdentitySource(
        identity_bank=bank,
        frame_encoder=lambda _frame, **_kwargs: (1.0, 0.0),
        minimum_observations_per_character=3,
    )

    score = source(
        "candidate.mp4",
        shot=Shot(["lead"]),
        frames=_frames(),
        attempt_index=0,
    )

    assert score > 0.99


def test_lower_tail_identity_catches_brief_character_drift():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("lead", [(1.0, 0.0), (1.0, 0.01)])
    vectors = {
        0: (1.0, 0.0),
        1: (1.0, 0.0),
        2: (0.1, 1.0),
        3: (1.0, 0.0),
        4: (1.0, 0.0),
    }

    source = EmbeddingBankVideoIdentitySource(
        identity_bank=bank,
        frame_encoder=lambda _frame, *, frame_index, **_kwargs: vectors[frame_index],
        minimum_observations_per_character=3,
        lower_tail_quantile=0.20,
    )

    score = source(
        "candidate.mp4",
        shot=Shot(["lead"]),
        frames=_frames(),
        attempt_index=0,
    )

    assert score < 0.20


def test_multi_character_score_is_weakest_character_not_average():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("lead", [(1.0, 0.0), (0.99, 0.01)])
    bank.build_character("partner", [(0.0, 1.0), (0.01, 0.99)])

    def encode(_frame, *, character_id, **_kwargs):
        if character_id == "lead":
            return (1.0, 0.0)
        return (0.8, 0.2)

    source = EmbeddingBankVideoIdentitySource(
        identity_bank=bank,
        frame_encoder=encode,
        minimum_observations_per_character=3,
    )

    score = source(
        "candidate.mp4",
        shot=Shot(["lead", "partner"]),
        frames=_frames(),
        attempt_index=1,
    )

    partner_score = bank.similarity("partner", (0.8, 0.2))
    assert score == pytest.approx(max(0.0, partner_score))
    assert score < 0.30


def test_missing_anchor_fails_closed():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("lead", [(1.0, 0.0)])
    source = EmbeddingBankVideoIdentitySource(
        identity_bank=bank,
        frame_encoder=lambda _frame, **_kwargs: (1.0, 0.0),
    )

    with pytest.raises(VideoIdentityMetricError, match="no approved identity anchor"):
        source(
            "candidate.mp4",
            shot=Shot(["missing"]),
            frames=_frames(),
            attempt_index=0,
        )


def test_insufficient_detected_character_observations_fail_closed():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("lead", [(1.0, 0.0)])

    def encode(_frame, *, frame_index, **_kwargs):
        return (1.0, 0.0) if frame_index == 0 else None

    source = EmbeddingBankVideoIdentitySource(
        identity_bank=bank,
        frame_encoder=encode,
        minimum_observations_per_character=2,
    )

    with pytest.raises(VideoIdentityMetricError, match="2 required"):
        source(
            "candidate.mp4",
            shot=Shot(["lead"]),
            frames=_frames(3),
            attempt_index=0,
        )


def test_character_metadata_is_required():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("lead", [(1.0, 0.0)])
    source = EmbeddingBankVideoIdentitySource(
        identity_bank=bank,
        frame_encoder=lambda _frame, **_kwargs: (1.0, 0.0),
    )

    class EmptyShot:
        characters = []

    with pytest.raises(VideoIdentityMetricError, match="shot.characters"):
        source(
            "candidate.mp4",
            shot=EmptyShot(),
            frames=_frames(),
            attempt_index=0,
        )
