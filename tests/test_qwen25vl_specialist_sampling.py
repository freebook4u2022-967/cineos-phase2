"""Regression coverage for high-resolution specialist semantic sampling."""

from pathlib import Path

import pytest

from cineos.atlas.artifact_video_observer import FFmpegRGBSampler, RGBVideoSample
from cineos.atlas.qwen25vl_semantic_judge import (
    Qwen25VLSemanticJudge,
    Qwen25VLSemanticJudgeError,
)


def test_real_qwen_path_uses_independent_high_resolution_artifact_decode():
    judge = Qwen25VLSemanticJudge()

    sampler = judge.artifact_sampler
    assert isinstance(sampler, FFmpegRGBSampler)
    assert sampler.width == 384
    assert sampler.height == 384
    assert sampler.sample_fps == 4.0
    assert sampler.max_frames == 24

    provenance = judge.runtime_provenance()
    assert provenance["sampling"] == {
        "mode": "independent_artifact_decode",
        "width": 384,
        "height": 384,
        "sample_fps": 4.0,
        "max_frames": 24,
    }
    assert provenance["origin"] == "external_pretrained"
    assert provenance["production_measurement_evidence"] is True


def test_injected_model_preserves_legacy_upstream_sample_without_sampler():
    judge = Qwen25VLSemanticJudge(model=object(), processor=object())

    assert judge.artifact_sampler is None
    assert judge.runtime_provenance()["sampling"] == {
        "mode": "upstream_observer_sample"
    }


def test_explicit_specialist_sampler_must_return_rgb_video_sample(tmp_path):
    class _BadSampler:
        def __call__(self, artifact: Path):
            return object()

    judge = Qwen25VLSemanticJudge(
        artifact_sampler=_BadSampler(),
        model=object(),
        processor=object(),
    )
    upstream = RGBVideoSample(
        width=1,
        height=1,
        frames=(b"\x00\x00\x00", b"\x01\x01\x01"),
    )

    with pytest.raises(
        Qwen25VLSemanticJudgeError,
        match="specialist artifact sampler must return RGBVideoSample evidence",
    ):
        judge(
            upstream,
            artifact=tmp_path / "shot.mp4",
            shot=object(),
            attempt_index=0,
        )
