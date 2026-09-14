from types import SimpleNamespace

import pytest

from cineos.atlas.production_semantic_qc import (
    ProductionSemanticQCError,
    validate_dialogue_speaker_bindings,
)


def _shot(dialogue_timing, *, characters=None):
    return SimpleNamespace(
        metadata={"benchmark_challenges": ["dialogue"]},
        performance={"dialogue_timing": dialogue_timing},
        characters=[] if characters is None else characters,
    )


def test_multi_speaker_bindings_accept_distinct_stable_non_overlapping_tracks() -> None:
    shot = _shot(
        [
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            },
            {
                "speaker_id": "bob",
                "speaker_face_track_index": 1,
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 0,
                "start_seconds": 2.2,
                "end_seconds": 3.0,
            },
        ]
    )

    validate_dialogue_speaker_bindings([shot])


def test_multi_speaker_bindings_reject_two_speakers_on_one_face_track() -> None:
    shot = _shot(
        [
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            },
            {
                "speaker_id": "bob",
                "speaker_face_track_index": 0,
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
        ]
    )

    with pytest.raises(ProductionSemanticQCError, match="distinct speakers"):
        validate_dialogue_speaker_bindings([shot])


def test_multi_speaker_bindings_reject_one_speaker_drifting_between_tracks() -> None:
    shot = _shot(
        [
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 0.8,
            },
            {
                "speaker_id": "bob",
                "speaker_face_track_index": 1,
                "start_seconds": 0.8,
                "end_seconds": 1.5,
            },
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 2,
                "start_seconds": 1.5,
                "end_seconds": 2.2,
            },
        ]
    )

    with pytest.raises(ProductionSemanticQCError, match="stable speaker-to-face"):
        validate_dialogue_speaker_bindings([shot])


def test_multi_speaker_bindings_reject_overlapping_mixed_audio_windows() -> None:
    shot = _shot(
        [
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 1.2,
            },
            {
                "speaker_id": "bob",
                "speaker_face_track_index": 1,
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
        ]
    )

    with pytest.raises(ProductionSemanticQCError, match="must not overlap"):
        validate_dialogue_speaker_bindings([shot])


def test_dialogue_timing_rejects_malformed_cue_before_speaker_counting() -> None:
    shot = _shot(
        [
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            },
            "not-a-cue",
        ]
    )

    with pytest.raises(ProductionSemanticQCError, match="must be a mapping"):
        validate_dialogue_speaker_bindings([shot])


def test_dialogue_timing_rejects_missing_speaker_before_speaker_counting() -> None:
    shot = _shot(
        [
            {
                "speaker_id": "alice",
                "speaker_face_track_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            },
            {
                "speaker_face_track_index": 1,
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
        ]
    )

    with pytest.raises(ProductionSemanticQCError, match="speaker_id must be non-empty"):
        validate_dialogue_speaker_bindings([shot])


def test_multi_character_dialogue_requires_explicit_timing_before_render() -> None:
    shot = _shot(
        None, characters=[{"character_uuid": "alice"}, {"character_uuid": "bob"}]
    )

    with pytest.raises(ProductionSemanticQCError, match="multi-character dialogue"):
        validate_dialogue_speaker_bindings([shot])


def test_multi_character_dialogue_rejects_unconditioned_speaker_identity() -> None:
    shot = _shot(
        [{"speaker_id": "mallory", "speaker_face_track_index": 0}],
        characters=[{"character_uuid": "alice"}, {"character_uuid": "bob"}],
    )

    with pytest.raises(ProductionSemanticQCError, match="conditioned character_uuid"):
        validate_dialogue_speaker_bindings([shot])


def test_multi_character_dialogue_rejects_duplicate_conditioned_identity() -> None:
    shot = _shot(
        [{"speaker_id": "alice", "speaker_face_track_index": 0}],
        characters=[{"character_uuid": "alice"}, {"character_uuid": "alice"}],
    )

    with pytest.raises(
        ProductionSemanticQCError, match="distinct conditioned character_uuid"
    ):
        validate_dialogue_speaker_bindings([shot])


def test_multi_character_dialogue_accepts_speaker_bound_to_conditioned_identity() -> (
    None
):
    shot = _shot(
        [{"speaker_id": " alice ", "speaker_face_track_index": 0}],
        characters=[{"character_uuid": "alice"}, {"character_uuid": "bob"}],
    )

    validate_dialogue_speaker_bindings([shot])


def test_multi_character_single_speaker_requires_face_track_binding() -> None:
    shot = _shot(
        [{"speaker_id": "alice"}],
        characters=[{"character_uuid": "alice"}, {"character_uuid": "bob"}],
    )

    with pytest.raises(ProductionSemanticQCError, match="speaker_face_track_index"):
        validate_dialogue_speaker_bindings([shot])


def test_multi_character_single_speaker_accepts_stable_face_track_binding() -> None:
    shot = _shot(
        [
            {"speaker_id": "alice", "speaker_face_track_index": 1},
            {"speaker_id": "alice", "speaker_face_track_index": 1},
        ],
        characters=[{"character_uuid": "alice"}, {"character_uuid": "bob"}],
    )

    validate_dialogue_speaker_bindings([shot])


def test_multi_character_single_speaker_rejects_face_track_drift() -> None:
    shot = _shot(
        [
            {"speaker_id": "alice", "speaker_face_track_index": 0},
            {"speaker_id": "alice", "speaker_face_track_index": 1},
        ],
        characters=[{"character_uuid": "alice"}, {"character_uuid": "bob"}],
    )

    with pytest.raises(ProductionSemanticQCError, match="stable speaker-to-face"):
        validate_dialogue_speaker_bindings([shot])


def test_single_speaker_dialogue_keeps_runtime_compatibility() -> None:
    shot = _shot(
        [
            {
                "speaker_id": "alice",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            }
        ]
    )

    validate_dialogue_speaker_bindings([shot])


def test_single_speaker_dialogue_does_not_require_track_or_window_metadata() -> None:
    shot = _shot([{"speaker_id": "alice"}])

    validate_dialogue_speaker_bindings([shot])
