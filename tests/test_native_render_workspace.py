from __future__ import annotations

import json

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video.render_workspace import (
    NativeRenderWorkspace,
    RenderContract,
    RenderWorkspaceError,
)
from cineos.native_video.temporal_model import NativeTemporalModel, TemporalFrameInput


def _contract(**overrides: object) -> RenderContract:
    data = {
        "shot_id": "shot-001",
        "base_state_hash": "a" * 64,
        "width": 320,
        "height": 180,
        "fps": 8,
        "frame_count": 3,
        "decoder_id": "decoder/1",
        "renderer_id": "renderer/1",
        "conditioning_hash": "b" * 64,
    }
    data.update(overrides)
    return RenderContract(**data)


def _advance(model: NativeTemporalModel, state, index: int) -> None:
    model.step(
        TemporalFrameInput(
            shot_id="shot-001",
            frame_index=index,
            identity=Tensor((0.1,) * model.identity_dim, (model.identity_dim,)),
            scene=Tensor((0.2,) * model.scene_dim, (model.scene_dim,)),
            motion=Tensor((0.05,) * model.motion_dim, (model.motion_dim,)),
        ),
        state,
    )


def test_workspace_round_trip_resumes_after_committed_frame(tmp_path) -> None:
    workspace = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    _advance(model, state, 0)

    workspace.commit_frame(0, b"P6\n1 1\n255\n\x00\x00\x00", state)

    reopened = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    restored = reopened.load_state()
    assert restored is not None
    assert restored.last_frame_index == 0
    assert reopened.next_frame_index() == 1


def test_workspace_rejects_contract_drift(tmp_path) -> None:
    NativeRenderWorkspace.open(tmp_path / "shot", _contract())

    with pytest.raises(RenderWorkspaceError, match="different render contract"):
        NativeRenderWorkspace.open(tmp_path / "shot", _contract(fps=12))


def test_workspace_rejects_manifest_tampering(tmp_path) -> None:
    workspace = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    document = json.loads(workspace.manifest_path.read_text(encoding="utf-8"))
    document["contract"]["fps"] = 99
    workspace.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RenderWorkspaceError, match="contract hash mismatch"):
        NativeRenderWorkspace.open(tmp_path / "shot", _contract())


def test_workspace_rejects_frames_without_checkpoint(tmp_path) -> None:
    workspace = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    workspace.frame_path(0).write_bytes(b"not-empty")

    with pytest.raises(RenderWorkspaceError, match="frames but no temporal checkpoint"):
        workspace.next_frame_index()


def test_workspace_rejects_missing_committed_frame(tmp_path) -> None:
    workspace = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    _advance(model, state, 0)
    workspace.save_state(state)

    with pytest.raises(RenderWorkspaceError, match="missing committed frame 0"):
        workspace.next_frame_index()


def test_workspace_rejects_uncommitted_future_frame(tmp_path) -> None:
    workspace = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    _advance(model, state, 0)
    workspace.commit_frame(0, b"frame-zero", state)
    workspace.frame_path(1).write_bytes(b"future")

    with pytest.raises(RenderWorkspaceError, match="uncommitted future frame"):
        workspace.next_frame_index()


def test_workspace_rejects_temporal_checkpoint_tampering(tmp_path) -> None:
    workspace = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    workspace.save_state(state)
    document = json.loads(workspace.checkpoint_path.read_text(encoding="utf-8"))
    document["state"]["shot_id"] = "shot-evil"
    workspace.checkpoint_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RenderWorkspaceError, match="hash mismatch"):
        workspace.load_state()


def test_workspace_clear_removes_attempt_state(tmp_path) -> None:
    workspace = NativeRenderWorkspace.open(tmp_path / "shot", _contract())
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    _advance(model, state, 0)
    workspace.commit_frame(0, b"frame-zero", state)

    workspace.clear()

    assert not workspace.manifest_path.exists()
    assert not workspace.checkpoint_path.exists()
    assert not workspace.frame_path(0).exists()
