import pytest

from cineos.atlas.diffusers_video import (
    DiffusersVideoError,
    DiffusersVideoRenderer,
    FoundationProvenance,
)


class DeviceAwareOffloadPipeline:
    def __init__(self):
        self.calls = []

    def enable_model_cpu_offload(self, gpu_id=None):
        self.calls.append(("model", gpu_id))

    def enable_sequential_cpu_offload(self, gpu_id=None):
        self.calls.append(("sequential", gpu_id))


class LegacyOffloadPipeline:
    def __init__(self):
        self.calls = []

    def enable_model_cpu_offload(self):
        self.calls.append("model")


def _renderer(tmp_path, pipeline):
    return DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=lambda *_args, **_kwargs: None,
    )


def test_model_cpu_offload_targets_selected_nonzero_gpu(tmp_path):
    pipeline = DeviceAwareOffloadPipeline()
    renderer = _renderer(tmp_path, pipeline)

    renderer.initialize()
    renderer.load_model(device="cuda:1", memory_strategy="model_cpu_offload")

    assert pipeline.calls == [("model", 1)]


def test_sequential_cpu_offload_targets_selected_nonzero_gpu(tmp_path):
    pipeline = DeviceAwareOffloadPipeline()
    renderer = _renderer(tmp_path, pipeline)

    renderer.initialize()
    renderer.load_model(device="cuda:2", memory_strategy="sequential_cpu_offload")

    assert pipeline.calls == [("sequential", 2)]


def test_nonzero_gpu_fails_closed_when_pipeline_cannot_accept_gpu_id(tmp_path):
    pipeline = LegacyOffloadPipeline()
    renderer = _renderer(tmp_path, pipeline)

    renderer.initialize()
    with pytest.raises(DiffusersVideoError, match="cannot target cuda:1"):
        renderer.load_model(device="cuda:1", memory_strategy="model_cpu_offload")

    assert pipeline.calls == []


def test_gpu_zero_keeps_backward_compatibility_with_legacy_offload_pipeline(tmp_path):
    pipeline = LegacyOffloadPipeline()
    renderer = _renderer(tmp_path, pipeline)

    renderer.initialize()
    renderer.load_model(device="cuda", memory_strategy="model_cpu_offload")

    assert pipeline.calls == ["model"]
