from pathlib import Path

import pytest

from cineos.atlas.wan22_execution import Wan22ExecutionConfig, Wan22ExecutionError
from cineos.atlas.wan22_sequence import (
    Wan22SequenceShot,
    run_wan22_gpu_sequence_validation,
)


class FakeOutput:
    def __init__(self, frame_count):
        self.frames = [[f"frame-{index}" for index in range(frame_count)]]


class PersistentPipeline:
    def __init__(self):
        self.calls = []
        self.references = []

    def to(self, _device):
        return self

    def enable_vae_tiling(self):
        return None

    def __call__(
        self,
        prompt,
        width,
        height,
        num_frames,
        generator=None,
        image=None,
        negative_prompt=None,
        num_inference_steps=None,
        guidance_scale=None,
    ):
        del (
            prompt,
            width,
            height,
            generator,
            negative_prompt,
            num_inference_steps,
            guidance_scale,
        )
        self.calls.append(num_frames)
        self.references.append(image)
        return FakeOutput(num_frames)


def _shots(reference_id="hero-approved-front", count=5):
    return [
        Wan22SequenceShot(
            shot_id=f"shot-{index + 1:02d}",
            scene_id="scene-connected",
            config=Wan22ExecutionConfig(
                prompt=f"same actor performs connected action {index + 1}",
                requested_duration_seconds=1.0,
                seed=100 + index,
                approved_reference_id=reference_id,
            ),
            continuity_note=f"continue directly from shot {index}",
        )
        for index in range(count)
    ]


def test_sequence_reuses_one_loaded_pipeline_across_five_shots(tmp_path):
    pipeline = PersistentPipeline()
    factory_calls = []

    def factory(model_id, **kwargs):
        factory_calls.append((model_id, kwargs))
        return pipeline

    def exporter(frames, output_path, *, fps):
        del frames, fps
        Path(output_path).write_bytes(b"connected-video")

    receipt = run_wan22_gpu_sequence_validation(
        _shots(),
        output_dir=tmp_path,
        device="cpu",
        dtype="float32",
        memory_strategy="resident",
        reference_loader=lambda reference_id: f"loaded:{reference_id}",
        pipeline_factory=factory,
        video_exporter=exporter,
    )

    assert len(factory_calls) == 1
    assert len(pipeline.calls) == 5
    assert pipeline.references == ["loaded:hero-approved-front"] + ["frame-24"] * 4
    assert receipt["shot_count"] == 5
    assert receipt["runtime"]["persistent_model_session"] is True
    assert receipt["conditioning"] == {
        "require_shared_reference": True,
        "shared_approved_reference_id": "hero-approved-front",
    }
    assert receipt["quality_control"] == {
        "enabled": False,
        "max_rerender_attempts": 2,
        "rejected_candidate_count": 0,
        "accepted_shot_count": 5,
    }
    assert [item["previous_shot_id"] for item in receipt["shots"]] == [
        None,
        "shot-01",
        "shot-02",
        "shot-03",
        "shot-04",
    ]
    assert len({item["artifact"]["sha256"] for item in receipt["shots"]}) == 1
    assert len(receipt["sequence_sha256"]) == 64
    assert Path(receipt["manifest_path"]).exists()


def test_sequence_qc_rejects_preserves_and_rerenders_in_same_session(tmp_path):
    pipeline = PersistentPipeline()
    export_count = 0

    def factory(_model_id, **_kwargs):
        return pipeline

    def exporter(frames, output_path, *, fps):
        nonlocal export_count
        del frames, fps
        export_count += 1
        Path(output_path).write_bytes(f"candidate-{export_count}".encode())

    def evaluator(output_path, *, shot, attempt_index):
        del output_path
        if shot.shot_id == "shot-01" and attempt_index == 0:
            return {
                "accepted": False,
                "score": 0.41,
                "directives": ["preserve identity", "reduce temporal drift"],
            }
        return {"accepted": True, "score": 0.93}

    receipt = run_wan22_gpu_sequence_validation(
        _shots(),
        output_dir=tmp_path,
        device="cpu",
        dtype="float32",
        memory_strategy="resident",
        reference_loader=lambda reference_id: f"loaded:{reference_id}",
        pipeline_factory=factory,
        video_exporter=exporter,
        shot_quality_evaluator=evaluator,
        max_rerender_attempts=2,
    )

    assert len(pipeline.calls) == 6
    assert pipeline.references == ["loaded:hero-approved-front"] * 2 + ["frame-24"] * 4
    assert receipt["quality_control"]["enabled"] is True
    assert receipt["quality_control"]["rejected_candidate_count"] == 1
    first = receipt["shots"][0]
    assert first["rerender_count"] == 1
    assert [attempt["status"] for attempt in first["attempts"]] == [
        "rejected",
        "accepted",
    ]
    assert [attempt["seed"] for attempt in first["attempts"]] == [100, 101]
    rejected_path = Path(first["attempts"][0]["output_path"])
    assert rejected_path.name.endswith(".rejected-01.mp4")
    assert rejected_path.read_bytes() == b"candidate-1"
    assert Path(first["output_path"]).read_bytes() == b"candidate-2"


def test_sequence_qc_fails_closed_after_retry_budget(tmp_path):
    pipeline = PersistentPipeline()

    def exporter(frames, output_path, *, fps):
        del frames, fps
        Path(output_path).write_bytes(b"rejected-candidate")

    with pytest.raises(Wan22ExecutionError, match="failed quality gate after 2"):
        run_wan22_gpu_sequence_validation(
            _shots(),
            output_dir=tmp_path,
            device="cpu",
            dtype="float32",
            memory_strategy="resident",
            reference_loader=lambda reference_id: f"loaded:{reference_id}",
            pipeline_factory=lambda *_args, **_kwargs: pipeline,
            video_exporter=exporter,
            shot_quality_evaluator=lambda *_args, **_kwargs: {"accepted": False},
            max_rerender_attempts=1,
        )

    assert len(pipeline.calls) == 2
    assert (tmp_path / "scene-connected-shot-01.rejected-01.mp4").exists()
    assert (tmp_path / "scene-connected-shot-01.rejected-02.mp4").exists()


def test_identity_sequence_fails_closed_on_reference_drift(tmp_path):
    shots = _shots()
    shots[-1] = Wan22SequenceShot(
        shot_id="shot-05",
        scene_id="scene-connected",
        config=Wan22ExecutionConfig(
            prompt="different identity must not enter shared-reference benchmark",
            requested_duration_seconds=1.0,
            approved_reference_id="different-approved-reference",
        ),
    )

    with pytest.raises(Wan22ExecutionError, match="one shared approved reference"):
        run_wan22_gpu_sequence_validation(shots, output_dir=tmp_path)


def test_production_sequence_requires_five_to_ten_shots(tmp_path):
    with pytest.raises(Wan22ExecutionError, match="between 5 and 10"):
        run_wan22_gpu_sequence_validation(_shots(count=4), output_dir=tmp_path)


def test_sequence_with_references_requires_loader(tmp_path):
    with pytest.raises(Wan22ExecutionError, match="reference_loader"):
        run_wan22_gpu_sequence_validation(_shots(), output_dir=tmp_path)
