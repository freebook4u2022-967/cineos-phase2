from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video import (
    NativeFilmContinuityBridge,
    NativeFilmRendererBinding,
    NativeTemporalModel,
)


class RecordingNativeRenderer:
    def __init__(self) -> None:
        self.states = []
        self.cancelled = False

    def render(self, planned, target, *, temporal_state):
        self.states.append(temporal_state)
        temporal_state.hidden = Tensor(
            (1.0,) * len(temporal_state.hidden.values),
            temporal_state.hidden.shape,
            temporal_state.hidden.device,
        )
        temporal_state.last_latent = Tensor(
            (2.0,) * 8,
            (8,),
            temporal_state.hidden.device,
        )
        temporal_state.last_frame_index = 0
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native-shot")
        return path

    def cancel_pending(self) -> None:
        self.cancelled = True


def _planned(shot_id: str):
    return SimpleNamespace(shot_id=shot_id, payload={})


def test_binding_passes_exact_active_temporal_state_to_renderer(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    planned = _planned("shot-1")
    bridge.start_attempt(planned, 0, 1)
    active = bridge.state_for("shot-1")

    renderer = RecordingNativeRenderer()
    binding = NativeFilmRendererBinding(renderer=renderer, continuity=bridge)
    output = binding.render(planned, tmp_path / "shot.mp4")

    assert renderer.states == [active]
    assert output.read_bytes() == b"native-shot"

    bridge.accept_attempt(planned, 0, 1)
    assert bridge.memory.latest() is not None
    assert bridge.memory.latest().shot_id == "shot-1"
    assert bridge.memory.latest().latent.values == (2.0,) * bridge.model.latent_dim


def test_binding_fails_closed_without_active_attempt(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    binding = NativeFilmRendererBinding(
        renderer=RecordingNativeRenderer(),
        continuity=bridge,
    )

    with pytest.raises(KeyError, match="no active temporal state"):
        binding.render(_planned("shot-missing"), tmp_path / "missing.mp4")


def test_binding_rejects_missing_shot_id(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    binding = NativeFilmRendererBinding(
        renderer=RecordingNativeRenderer(),
        continuity=bridge,
    )

    with pytest.raises(ValueError, match="shot_id"):
        binding.render(SimpleNamespace(shot_id=""), tmp_path / "missing.mp4")


def test_binding_forwards_cancellation():
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    renderer = RecordingNativeRenderer()
    binding = NativeFilmRendererBinding(renderer=renderer, continuity=bridge)

    binding.cancel_pending()

    assert renderer.cancelled is True
