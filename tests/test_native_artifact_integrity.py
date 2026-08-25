from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video import (
    ArtifactIntegrityError,
    NativeFilmContinuityBridge,
    NativeTemporalModel,
    provenance_for,
    verify_provenance,
)


def _planned(shot_id: str):
    return SimpleNamespace(shot_id=shot_id, payload={})


def _durable_anchor_with_provenance(bridge: NativeFilmContinuityBridge, artifact):
    planned = _planned("shot-integrity")
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
    provenance = provenance_for(artifact)
    state.metadata["native_artifact_sha256"] = provenance.sha256
    state.metadata["native_artifact_bytes"] = provenance.byte_size
    bridge.accept_attempt(planned, 0, 1)
    return provenance


def test_provenance_roundtrip_and_mismatch_detection(tmp_path):
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"native-shot-v1")
    provenance = provenance_for(artifact)

    verified = verify_provenance(
        artifact,
        sha256=provenance.sha256,
        byte_size=provenance.byte_size,
    )
    assert verified == provenance

    artifact.write_bytes(b"native-shot-v2")
    with pytest.raises(ArtifactIntegrityError, match="sha256"):
        verify_provenance(
            artifact,
            sha256=provenance.sha256,
            byte_size=provenance.byte_size,
        )


def test_provenance_rejects_missing_and_empty_artifacts(tmp_path):
    with pytest.raises(ArtifactIntegrityError, match="missing artifact"):
        provenance_for(tmp_path / "missing.mp4")

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ArtifactIntegrityError, match="empty artifact"):
        provenance_for(empty)


def test_bridge_verifies_resumed_artifact_against_durable_anchor(tmp_path):
    artifact = tmp_path / "accepted.mp4"
    artifact.write_bytes(b"accepted-native-render")
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    provenance = _durable_anchor_with_provenance(bridge, artifact)

    snapshot = bridge.snapshot()
    resumed = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    resumed.restore(snapshot)

    assert resumed.verify_latest_artifact(artifact) == provenance

    artifact.write_bytes(b"tampered-native-render")
    with pytest.raises(ArtifactIntegrityError, match="continuity provenance"):
        resumed.verify_latest_artifact(artifact)


def test_bridge_fails_closed_for_legacy_anchor_without_provenance(tmp_path):
    artifact = tmp_path / "legacy.mp4"
    artifact.write_bytes(b"legacy-native-render")
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    planned = _planned("legacy-shot")
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

    with pytest.raises(ArtifactIntegrityError, match="no native artifact provenance"):
        bridge.verify_latest_artifact(artifact)

    assert bridge.verify_latest_artifact(
        artifact,
        require_provenance=False,
    ) is None
