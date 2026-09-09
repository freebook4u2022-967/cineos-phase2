from __future__ import annotations

import pytest

from cineos.atlas.model_snapshot_attestation import (
    ModelSnapshotAttestationError,
    attest_model_snapshot,
    verify_model_snapshot,
)


def _snapshot(tmp_path, name="snapshot"):
    root = tmp_path / name
    (root / "unet").mkdir(parents=True)
    (root / "config.json").write_text('{"kind":"wan"}\n', encoding="utf-8")
    (root / "unet" / "weights.bin").write_bytes(b"pinned-weights")
    return root


def test_snapshot_attestation_is_deterministic_and_revision_bound(tmp_path):
    root = _snapshot(tmp_path)

    first = attest_model_snapshot(
        root, model_id="Wan-AI/Wan2.2-I2V-A14B", revision="abc123"
    )
    second = attest_model_snapshot(
        root, model_id="Wan-AI/Wan2.2-I2V-A14B", revision="abc123"
    )

    assert first == second
    assert [item.path for item in first.files] == ["config.json", "unet/weights.bin"]
    assert len(first.snapshot_sha256) == 64
    assert first.to_dict()["schema"] == "cineos-model-snapshot-attestation/0.1"

    other_revision = attest_model_snapshot(
        root, model_id="Wan-AI/Wan2.2-I2V-A14B", revision="def456"
    )
    assert other_revision.snapshot_sha256 != first.snapshot_sha256


def test_verify_rejects_mutated_weight_bytes(tmp_path):
    root = _snapshot(tmp_path)
    approved = attest_model_snapshot(root, model_id="model", revision="rev")
    (root / "unet" / "weights.bin").write_bytes(b"substituted-weights")

    with pytest.raises(ModelSnapshotAttestationError, match="do not match"):
        verify_model_snapshot(root, approved)


def test_verify_rejects_missing_or_added_files(tmp_path):
    missing_root = _snapshot(tmp_path, "missing")
    missing_approved = attest_model_snapshot(
        missing_root, model_id="model", revision="rev"
    )
    (missing_root / "config.json").unlink()

    with pytest.raises(ModelSnapshotAttestationError, match="do not match"):
        verify_model_snapshot(missing_root, missing_approved)

    added_root = _snapshot(tmp_path, "added")
    added_approved = attest_model_snapshot(added_root, model_id="model", revision="rev")
    (added_root / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ModelSnapshotAttestationError, match="do not match"):
        verify_model_snapshot(added_root, added_approved)


def test_rejects_empty_snapshot_and_blank_provenance(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ModelSnapshotAttestationError, match="contains no files"):
        attest_model_snapshot(root, model_id="model", revision="rev")
    with pytest.raises(ModelSnapshotAttestationError, match="must be non-empty"):
        attest_model_snapshot(root, model_id=" ", revision="rev")


def test_rejects_symlinked_model_content(tmp_path):
    root = _snapshot(tmp_path)
    target = tmp_path / "outside.bin"
    target.write_bytes(b"mutable")
    link = root / "external.bin"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ModelSnapshotAttestationError, match="unsupported symlink"):
        attest_model_snapshot(root, model_id="model", revision="rev")
