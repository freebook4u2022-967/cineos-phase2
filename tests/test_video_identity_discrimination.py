from types import SimpleNamespace

import pytest

from cineos.atlas.video_identity import EmbeddingBankVideoIdentitySource
from cineos.native_image.identity_bank import CharacterIdentityEmbeddingBank
from cineos.native_image.neural_decoder import DecodedRGBFrame

_FRAME = DecodedRGBFrame(width=1, height=1, rgb=b"\x00\x00\x00")
_FRAMES = (_FRAME, _FRAME, _FRAME)


def _bank() -> CharacterIdentityEmbeddingBank:
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("alice", [(1.0, 0.0)])
    bank.build_character("bob", [(0.0, 1.0)])
    return bank


def _shot(*character_ids: str):
    return SimpleNamespace(
        characters=[{"character_id": character_id} for character_id in character_ids]
    )


def _score(vectors, *character_ids: str, margin: float = 0.05) -> float:
    def encoder(frame, *, character_id, shot, frame_index):
        del frame, shot, frame_index
        return vectors[character_id]

    source = EmbeddingBankVideoIdentitySource(
        identity_bank=_bank(),
        frame_encoder=encoder,
        minimum_cross_character_margin=margin,
    )
    return source(
        "unused.mp4",
        shot=_shot(*character_ids),
        frames=_FRAMES,
        attempt_index=0,
    )


def test_single_character_score_preserves_legacy_semantics():
    assert _score({"alice": (1.0, 0.0)}, "alice") == pytest.approx(1.0)


def test_distinct_multi_character_identities_pass():
    assert _score(
        {"alice": (1.0, 0.0), "bob": (0.0, 1.0)}, "alice", "bob"
    ) == pytest.approx(1.0)


def test_multi_character_identity_collapse_is_rejected():
    collapsed = (1.0, 1.0)
    assert _score({"alice": collapsed, "bob": collapsed}, "alice", "bob") == 0.0


def test_multi_character_identity_swap_is_rejected():
    assert _score({"alice": (0.0, 1.0), "bob": (1.0, 0.0)}, "alice", "bob") == 0.0


def test_near_ambiguous_identity_respects_configured_margin():
    vectors = {"alice": (1.0, 0.98), "bob": (0.0, 1.0)}
    assert _score(vectors, "alice", "bob", margin=0.05) == 0.0


def test_cross_character_margin_is_validated():
    with pytest.raises(ValueError, match="minimum_cross_character_margin"):
        EmbeddingBankVideoIdentitySource(
            identity_bank=_bank(),
            frame_encoder=lambda *args, **kwargs: (1.0, 0.0),
            minimum_cross_character_margin=-0.01,
        )
