from __future__ import annotations

import json
from pathlib import Path

import pytest

from cineos.audio.lipsync_qc import (
    ExternalLipSyncAnalyzer,
    LipSyncAnalyzerProvenance,
    LipSyncQCError,
)


def _provenance() -> LipSyncAnalyzerProvenance:
    return LipSyncAnalyzerProvenance(
        analyzer_id="learned-av-sync-evaluator",
        analyzer_revision="immutable-revision-1",
        analyzer_license_id="declared-license",
        analyzer_source_url="https://example.invalid/model-card",
    )


def _completed() -> object:
    class Completed:
        stdout = json.dumps(
            {
                "sync_confidence": 0.92,
                "av_offset_ms": -16.0,
                "speaking_frame_coverage": 0.82,
                "face_track_coverage": 0.96,
            }
        )

    return Completed()


def _analyzer() -> ExternalLipSyncAnalyzer:
    return ExternalLipSyncAnalyzer(
        command=("sync-eval", "--video", "{video}", "--audio", "{audio}"),
        provenance=_provenance(),
    )


def test_external_analyzer_rejects_video_mutation_during_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "shot.mp4"
    audio = tmp_path / "dialogue.wav"
    video.write_bytes(b"approved-video")
    audio.write_bytes(b"approved-audio")

    def mutate_video(*args: object, **kwargs: object) -> object:
        video.write_bytes(b"substituted-video")
        return _completed()

    monkeypatch.setattr("cineos.audio.lipsync_qc.subprocess.run", mutate_video)

    with pytest.raises(LipSyncQCError, match="video artifact changed"):
        _analyzer().measure(shot_id="shot-01", video_path=video, audio_path=audio)


def test_external_analyzer_rejects_audio_mutation_during_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "shot.mp4"
    audio = tmp_path / "dialogue.wav"
    video.write_bytes(b"approved-video")
    audio.write_bytes(b"approved-audio")

    def mutate_audio(*args: object, **kwargs: object) -> object:
        audio.write_bytes(b"substituted-audio")
        return _completed()

    monkeypatch.setattr("cineos.audio.lipsync_qc.subprocess.run", mutate_audio)

    with pytest.raises(LipSyncQCError, match="audio artifact changed"):
        _analyzer().measure(shot_id="shot-01", video_path=video, audio_path=audio)
