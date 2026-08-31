from pathlib import Path

import pytest

from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_persistent_session import (
    PersistentGPUFoundationExecutor,
    PersistentGPUSessionError,
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
        self.calls = 0

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, *, disable):
        del disable

    def __call__(self, prompt, width, height, num_frames, generator=None):
        del prompt, width, height, generator
        self.calls += 1
        return _Output([object() for _ in range(num_frames)])


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + box_type + payload


def _minimal_mp4_bytes(payload: bytes) -> bytes:
    return b"".join(
        (
            _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2"),
            _box(b"moov"),
            _box(b"mdat", payload),
        )
    )


def _request(shot_id: str, seed: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=shot_id,
        scene_id="scene-001",
        camera={"resolution": (1280, 704), "fps": 24.0, "duration": 1.0},
        characters=[{"identity_invariants": ["same protagonist"]}],
        environment={"description": "rainy night street"},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=[],
        deterministic_seed=seed,
        renderer_requirements={},
        metadata={"prompt": f"connected shot {shot_id}"},
    )
    request.refresh_hash()
    return request


def test_persistent_session_loads_pipeline_once_for_multiple_shots(tmp_path):
    pipeline = _Pipeline()
    factory_calls = 0

    def factory(*_args, **_kwargs):
        nonlocal factory_calls
        factory_calls += 1
        return pipeline

    def exporter(_frames, output_path, *, fps):
        del fps
        payload = Path(output_path).stem.encode("utf-8")
        Path(output_path).write_bytes(_minimal_mp4_bytes(payload))

    session = PersistentGPUFoundationExecutor(
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        torch_module=_Torch(),
        pipeline_factory=factory,
        video_exporter=exporter,
    )

    with session as executor:
        first = executor.render(_request("shot-001", 101))
        second = executor.render(_request("shot-002", 102))
        assert executor.is_open is True

    assert session.is_open is False
    assert factory_calls == 1
    assert pipeline.calls == 2
    assert first.output_sha256 != second.output_sha256
    assert first.execution_plan == second.execution_plan
    assert first.runtime_provenance["persistent_model_session"] is True
    assert first.runtime_provenance["production_default_runtime"] is False


def test_persistent_session_callable_matches_benchmark_executor_contract(tmp_path):
    pipeline = _Pipeline()

    def exporter(_frames, output_path, *, fps):
        del fps
        Path(output_path).write_bytes(_minimal_mp4_bytes(b"connected-shot"))

    with PersistentGPUFoundationExecutor(
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        torch_module=_Torch(),
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=exporter,
    ) as executor:
        receipt = executor(
            _request("shot-003", 103),
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
        )

    assert receipt.result.shot_id == "shot-003"
    assert pipeline.calls == 1


def test_persistent_session_fails_closed_when_rendered_while_closed(tmp_path):
    session = PersistentGPUFoundationExecutor(
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        torch_module=_Torch(),
    )

    with pytest.raises(PersistentGPUSessionError, match="must be opened"):
        session.render(_request("shot-004", 104))
