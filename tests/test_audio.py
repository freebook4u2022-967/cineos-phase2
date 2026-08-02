from pathlib import Path

import pytest

from cineos.audio import (
    AudioProject,
    AudioTimeline,
    DialogueCue,
    FakeProvider,
    LipSyncMetadata,
    Mixer,
    MixInput,
    ProviderCapabilityError,
    VoiceCasting,
    VoiceProfile,
)
from cineos.audio.serializer import dumps, loads


def approved_voice(character: str = "character") -> VoiceProfile:
    return VoiceProfile(
        character,
        "en",
        approved_reference_ids=["rights-approved-reference"],
        rights_approved=True,
    )


def test_project_and_serialization_are_deterministic() -> None:
    first = AudioProject("package", language="en")
    second = AudioProject("package", language="en")
    assert first.project_id == second.project_id
    assert first.content_hash == second.content_hash
    assert loads(dumps(first)).content_hash == first.content_hash


def test_casting_is_stable_and_override_requires_approval() -> None:
    voice = approved_voice()
    casting = VoiceCasting()
    casting.add_profile(voice)
    casting.assign("character", voice.voice_id)
    casting.assign("character", voice.voice_id)
    with pytest.raises(Exception, match="require approval"):
        casting.assign("character", voice.voice_id, scene_id="scene")


def test_overlap_and_alignment() -> None:
    timeline = AudioTimeline()
    timeline.align_scene("scene", 0, 2)
    timeline.align_shot("scene", "shot", 0, 2)
    left = DialogueCue("scene", "shot", "a", "one", "en", 0, 1)
    right = DialogueCue("scene", "shot", "b", "two", "en", 0.5, 1)
    timeline.add_cue(left)
    timeline.add_cue(right)
    assert timeline.overlap_conflicts() == [(left.cue_id, right.cue_id)]
    with pytest.raises(ValueError, match="aligned"):
        timeline.add_cue(DialogueCue("scene", "shot", "a", "late", "en", 1.5, 1))


def test_fake_synthesis_mix_silence_and_lipsync(tmp_path: Path) -> None:
    voice = approved_voice()
    cue = DialogueCue(
        "scene",
        "shot",
        "character",
        "hello world",
        "en",
        0,
        0.1,
        approved_voice_profile_id=voice.voice_id,
    )
    result = FakeProvider(8_000).synthesize(cue, voice, tmp_path / "cue.wav")
    mixed = Mixer(8_000).mix([MixInput(result.audio_path)], tmp_path / "mixed.wav")
    silent = Mixer(8_000).mix([], tmp_path / "silent.wav", duration=0.1)
    metadata = LipSyncMetadata(
        "shot", "character", cue.cue_id, result.phonemes, result.words
    )
    assert mixed.stat().st_size > 44
    assert silent.stat().st_size > 44
    assert len(metadata.content_hash) == 64


def test_provider_rejects_unsupported_language(tmp_path: Path) -> None:
    voice = approved_voice()
    cue = DialogueCue("scene", "shot", "character", "bonjour", "fr", 0, 1)
    with pytest.raises(ProviderCapabilityError):
        FakeProvider().synthesize(cue, voice, tmp_path / "cue.wav")
