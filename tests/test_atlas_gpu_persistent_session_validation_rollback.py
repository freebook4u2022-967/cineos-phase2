from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionError
from cineos.atlas.gpu_persistent_session import PersistentGPUFoundationExecutor
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

    @staticmethod
    def synchronize(_index=0):
        return None


class _Torch:
    cuda = _Cuda()


class _Output:
    def __init__(self, frames):
        self.frames = [frames]


class _Pipeline:
    def __init__(self):
        self.calls = 0

    def to(self, _device):
        return self

    def set_progress_bar_config(self, *, disable):
        del disable

    def __call__(self, prompt, width, height, num_frames, generator=None):
        del prompt, width, height, generator
        self.calls += 1
        return _Output([object() for _ in range(num_frames)])


def _request(
    shot_id: str,
    *,
    seed: int,
    previous_shot: str | None = None,
) -> NativeShotRequest:
    continuity = {}
    if previous_shot is not None:
        continuity["previous_shot"] = previous_shot
    request = NativeShotRequest(
        shot_id=shot_id,
        scene_id="scene-rollback",
        camera={"resolution": (1280, 704), "fps": 24.0, "duration": 1.0},
        characters=[{"identity_invariants": ["same protagonist"]}],
        environment={"description": "rainy night street"},
        wardrobe=[],
        props=[],
        continuity=continuity,
        performance={},
        approved_reference_ids=[],
        deterministic_seed=seed,
        renderer_requirements={},
        metadata={"prompt": f"connected shot {shot_id}"},
    )
    request.refresh_hash()
    return request


def test_invalid_mp4_cannot_remain_as_successor_continuity_anchor(tmp_path):
    pipeline = _Pipeline()

    def exporter(_frames, output_path, *, fps):
        del fps
        # The renderer can hash/cache these fresh non-empty bytes, but the outer GPU
        # execution boundary must reject them as an invalid MP4 container.
        Path(output_path).write_bytes(b"not-an-mp4")

    with PersistentGPUFoundationExecutor(
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        torch_module=_Torch(),
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=exporter,
    ) as executor:
        with pytest.raises(GPUFoundationExecutionError):
            executor.render(_request("shot-invalid", seed=901))

        assert executor.is_open is True

        with pytest.raises(DiffusersVideoError, match="predecessor frame is unavailable"):
            executor.render(
                _request(
                    "shot-successor",
                    seed=902,
                    previous_shot="shot-invalid",
                )
            )

    # The invalid root reached inference exactly once. Its successor failed before
    # inference because validation rollback removed the rejected continuity anchor.
    assert pipeline.calls == 1
