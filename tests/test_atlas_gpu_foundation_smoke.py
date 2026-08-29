from pathlib import Path

import pytest

from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_foundation_smoke import (
    GPUFoundationExecutionError,
    execute_foundation_gpu_shot,
)
from cineos.atlas.native_request import NativeShotRequest


class _Properties:
    name = "Test GPU"
    total_memory = 48 * 1024**3


class _Cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_properties(_index):
        return _Properties()

    @staticmethod
    def get_device_capability(_index):
        return (9, 0)

    @staticmethod
    def is_bf16_supported():
        return True

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def mem_get_info(_index=0):
        return (40 * 1024**3, 48 * 1024**3)


class _Torch:
    cuda = _Cuda()


class _Output:
    def __init__(self, frames):
        self.frames = [frames]


class _Pipeline:
    def __init__(self):
        self.device = None
        self.progress_disabled = False

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, *, disable):
        self.progress_disabled = disable

    def __call__(self, prompt, width, height, num_frames, generator=None):
        del prompt, width, height, generator
        return _Output([object() for _ in range(num_frames)])


def _request():
    request = NativeShotRequest(
        shot_id="shot-001",
        scene_id="scene-001",
        camera={"resolution": (1280, 704), "fps": 24.0, "duration": 1.0},
        characters=[{"identity_invariants": ["same protagonist"]}],
        environment={"description": "rainy night street"},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=[],
        deterministic_seed=1234,
        renderer_requirements={},
        metadata={"prompt": "protagonist walks toward camera"},
    )
    request.refresh_hash()
    return request


def test_gpu_foundation_shot_binds_preflight_render_and_artifact_evidence(tmp_path):
    pipeline = _Pipeline()

    def exporter(_frames, output_path, *, fps):
        assert fps == 24.0
        Path(output_path).write_bytes(b"real-video-artifact")

    receipt = execute_foundation_gpu_shot(
        _request(),
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        torch_module=_Torch(),
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=exporter,
    )

    assert pipeline.device == "cuda:0"
    assert pipeline.progress_disabled is True
    assert receipt.result.frame_count == 24
    assert receipt.execution_plan.memory_strategy == "resident"
    assert receipt.output_bytes == len(b"real-video-artifact")
    assert len(receipt.output_sha256) == 64
    assert receipt.profile_id == WAN22_TI2V_5B_PROFILE.profile_id
    assert receipt.origin == "external_pretrained_foundation"
    assert receipt.to_dict()["foundation"]["model_id"].startswith("Wan-AI/")


def test_gpu_foundation_shot_fails_closed_without_written_video_artifact(tmp_path):
    with pytest.raises(GPUFoundationExecutionError, match="no readable video artifact"):
        execute_foundation_gpu_shot(
            _request(),
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            torch_module=_Torch(),
            pipeline_factory=lambda *_args, **_kwargs: _Pipeline(),
            video_exporter=lambda *_args, **_kwargs: None,
        )
