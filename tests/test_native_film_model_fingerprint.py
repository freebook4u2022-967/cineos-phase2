from __future__ import annotations

import pytest

from cineos.native_video.film_bridge import (
    NativeFilmContinuityBridge,
    temporal_model_fingerprint,
)
from cineos.native_video.temporal_model import NativeTemporalModel


def test_temporal_model_fingerprint_is_stable_for_identical_models() -> None:
    left = NativeTemporalModel.initialized()
    right = NativeTemporalModel.initialized()

    assert temporal_model_fingerprint(left) == temporal_model_fingerprint(right)


def test_temporal_model_fingerprint_changes_when_weights_change() -> None:
    baseline = NativeTemporalModel.initialized()
    changed = NativeTemporalModel.initialized()
    changed.recurrent.weights[0] += 0.001

    assert temporal_model_fingerprint(baseline) != temporal_model_fingerprint(changed)


def test_snapshot_binds_continuity_to_active_temporal_model() -> None:
    bridge = NativeFilmContinuityBridge.default()

    snapshot = bridge.snapshot()

    assert snapshot["temporal_model_fingerprint"] == temporal_model_fingerprint(
        bridge.model
    )


def test_restore_rejects_checkpoint_from_different_temporal_weights() -> None:
    source = NativeFilmContinuityBridge.default()
    snapshot = source.snapshot()

    upgraded_model = NativeTemporalModel.initialized()
    upgraded_model.decoder.bias[0] = 0.125
    destination = NativeFilmContinuityBridge(model=upgraded_model)

    with pytest.raises(ValueError, match="fingerprint"):
        destination.restore(snapshot)


def test_restore_accepts_legacy_checkpoint_without_model_fingerprint() -> None:
    source = NativeFilmContinuityBridge.default()
    snapshot = source.snapshot()
    snapshot.pop("temporal_model_fingerprint")

    destination = NativeFilmContinuityBridge.default()
    destination.restore(snapshot)

    migrated = destination.snapshot()
    assert migrated["temporal_model_fingerprint"] == temporal_model_fingerprint(
        destination.model
    )
