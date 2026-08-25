from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.film.build import FilmBuild
from cineos.film.orchestrator import FilmOrchestrator
from cineos.film.shot_state import ShotState
from cineos.native_image.tensor_model import Tensor
from cineos.native_video import (
    NativeFilmContinuityBridge,
    NativeFilmRendererBinding,
    NativeTemporalModel,
)


class RecordingNativeRenderer:
    def __init__(self, *, latent_dim: int = 16) -> None:
        self.latent_dim = latent_dim
        self.states = []
        self.initial_hidden = []
        self.cancelled = False

    def render(self, planned, target, *, temporal_state):
        self.states.append(temporal_state)
        self.initial_hidden.append(temporal_state.hidden.values)
        temporal_state.hidden = Tensor(
            (1.0,) * len(temporal_state.hidden.values),
            temporal_state.hidden.shape,
            temporal_state.hidden.device,
        )
        temporal_state.last_latent = Tensor(
            (2.0,) * self.latent_dim,
            (self.latent_dim,),
            temporal_state.hidden.device,
        )
        temporal_state.last_frame_index = 0
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native-shot")
        return path

    def cancel_pending(self) -> None:
        self.cancelled = True


class MissingArtifactRenderer(RecordingNativeRenderer):
    def render(self, planned, target, *, temporal_state):
        del planned, temporal_state
        return Path(target)


class EmptyArtifactRenderer(RecordingNativeRenderer):
    def render(self, planned, target, *, temporal_state):
        del planned, temporal_state
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path


class RejectThenApprove:
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, path, planned):
        self.calls += 1
        return {"approved": self.calls >= 2}


def _planned(shot_id: str, *, scene_id: str = "scene-a", index: int = 0):
    return SimpleNamespace(
        shot_id=shot_id,
        scene_id=scene_id,
        index=index,
        duration=1.0,
        payload={},
    )


def _accept_anchor(bridge: NativeFilmContinuityBridge, planned) -> None:
    bridge.start_attempt(planned, 0, 1)
    state = bridge.state_for(planned.shot_id)
    state.hidden = Tensor(
        (1.0,) * bridge.model.hidden_dim,
        (bridge.model.hidden_dim,),
        state.hidden.device,
    )
    state.last_latent = Tensor(
        (2.0,) * bridge.model.latent_dim,
        (bridge.model.latent_dim,),
        state.hidden.device,
    )
    state.last_frame_index = 0
    bridge.accept_attempt(planned, 0, 1)


def test_binding_passes_exact_active_temporal_state_to_renderer(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    planned = _planned("shot-1")
    bridge.start_attempt(planned, 0, 1)
    active = bridge.state_for("shot-1")

    renderer = RecordingNativeRenderer(latent_dim=bridge.model.latent_dim)
    binding = NativeFilmRendererBinding(renderer=renderer, continuity=bridge)
    output = binding.render(planned, tmp_path / "shot.mp4")

    assert renderer.states == [active]
    assert output.read_bytes() == b"native-shot"
    assert active.metadata["native_artifact_bytes"] == len(b"native-shot")
    assert (
        active.metadata["native_artifact_sha256"]
        == "1c7b54bea15174d5fa93a3689755184dfc8464cea7e73a4b6b36e5896863fa24"
    )

    bridge.accept_attempt(planned, 0, 1)
    assert bridge.memory.latest() is not None
    assert bridge.memory.latest().shot_id == "shot-1"
    assert bridge.memory.latest().latent.values == (2.0,) * bridge.model.latent_dim


def test_binding_fails_closed_when_renderer_returns_missing_artifact(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    planned = _planned("shot-missing-artifact")
    bridge.start_attempt(planned, 0, 1)
    binding = NativeFilmRendererBinding(
        renderer=MissingArtifactRenderer(latent_dim=bridge.model.latent_dim),
        continuity=bridge,
    )

    with pytest.raises(RuntimeError, match="missing artifact"):
        binding.render(planned, tmp_path / "missing.mp4")

    state = bridge.state_for(planned.shot_id)
    assert "native_artifact_sha256" not in state.metadata
    assert "native_artifact_bytes" not in state.metadata


def test_binding_fails_closed_when_renderer_returns_empty_artifact(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    planned = _planned("shot-empty-artifact")
    bridge.start_attempt(planned, 0, 1)
    binding = NativeFilmRendererBinding(
        renderer=EmptyArtifactRenderer(latent_dim=bridge.model.latent_dim),
        continuity=bridge,
    )

    with pytest.raises(RuntimeError, match="empty artifact"):
        binding.render(planned, tmp_path / "empty.mp4")

    state = bridge.state_for(planned.shot_id)
    assert "native_artifact_sha256" not in state.metadata
    assert "native_artifact_bytes" not in state.metadata


def test_orchestrator_retry_rebinds_last_durable_continuity_state(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    _accept_anchor(bridge, _planned("shot-anchor"))

    renderer = RecordingNativeRenderer(latent_dim=bridge.model.latent_dim)
    binding = NativeFilmRendererBinding(renderer=renderer, continuity=bridge)
    orchestrator = FilmOrchestrator(
        binding,
        validator=RejectThenApprove(),
        max_recovery_attempts=1,
        **bridge.orchestrator_kwargs(),
    )
    planned = _planned("shot-retry", index=1)
    state = ShotState(planned.shot_id)
    build = FilmBuild("project", "package", "native")

    orchestrator._render_shot(
        planned,
        state,
        tmp_path,
        build,
        scene_index=0,
    )

    assert state.approved
    assert state.attempt_count == 2
    assert len(renderer.initial_hidden) == 2
    assert renderer.initial_hidden[0] == pytest.approx(
        (0.65,) * bridge.model.hidden_dim
    )
    assert renderer.initial_hidden[1] == pytest.approx(
        (0.65,) * bridge.model.hidden_dim
    )
    assert len(bridge.memory.anchors) == 2
    assert bridge.memory.latest() is not None
    assert bridge.memory.latest().shot_id == "shot-retry"


def test_binding_fails_closed_without_active_attempt(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    binding = NativeFilmRendererBinding(
        renderer=RecordingNativeRenderer(latent_dim=bridge.model.latent_dim),
        continuity=bridge,
    )

    with pytest.raises(KeyError, match="no active temporal state"):
        binding.render(_planned("shot-missing"), tmp_path / "missing.mp4")


def test_binding_rejects_missing_shot_id(tmp_path):
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    binding = NativeFilmRendererBinding(
        renderer=RecordingNativeRenderer(latent_dim=bridge.model.latent_dim),
        continuity=bridge,
    )

    with pytest.raises(ValueError, match="shot_id"):
        binding.render(SimpleNamespace(shot_id=""), tmp_path / "missing.mp4")


def test_binding_forwards_cancellation():
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    renderer = RecordingNativeRenderer(latent_dim=bridge.model.latent_dim)
    binding = NativeFilmRendererBinding(renderer=renderer, continuity=bridge)

    binding.cancel_pending()

    assert renderer.cancelled is True
