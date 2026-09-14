from types import SimpleNamespace

import pytest

from cineos.atlas.production_semantic_qc import (
    ProductionSemanticQCError,
    validate_dialogue_speaker_bindings,
)


def _shot(dialogue_timing):
    return SimpleNamespace(
        metadata={"benchmark_challenges": ["dialogue"]},
        performance={"dialogue_timing": dialogue_timing},
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
