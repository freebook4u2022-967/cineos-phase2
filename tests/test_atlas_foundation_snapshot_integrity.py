"""Regression coverage for production foundation snapshot byte integrity."""

import json

import pytest

from cineos.atlas.foundation_snapshot_integrity import (
    FoundationSnapshotIntegrityError,
    inspect_foundation_snapshot,
    verify_foundation_snapshot_manifest,
    write_foundation_snapshot_manifest,
)


def _snapshot(tmp_path):
    root = tmp_path / "snapshot"
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text('{"_class_name":"Pipeline"}\n')
    (root / "transformer" / "weights.bin").write_bytes(b"foundation-weights")
    return root


def test_snapshot_integrity_is_deterministic_and_path_portable(tmp_path):
    first_root = _snapshot(tmp_path / "first")
    second_root = _snapshot(tmp_path / "second")

    first = inspect_foundation_snapshot(
        first_root,
        profile_id="wan-a14b",
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision="a" * 40,
    )
    second = inspect_foundation_snapshot(
        second_root,
        profile_id="wan-a14b",
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision="a" * 40,
    )

    assert first.tree_sha256 == second.tree_sha256
    assert first.file_count == 2
    assert first.total_bytes > 0
    assert all(not item["path"].startswith("/") for item in first.files)


def test_snapshot_integrity_changes_when_model_bytes_change(tmp_path):
    root = _snapshot(tmp_path)
    original = inspect_foundation_snapshot(
        root,
        profile_id="wan-5b",
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        revision="b" * 40,
    )

    (root / "transformer" / "weights.bin").write_bytes(b"substituted-weights")
    changed = inspect_foundation_snapshot(
        root,
        profile_id="wan-5b",
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        revision="b" * 40,
    )

    assert changed.tree_sha256 != original.tree_sha256


def test_manifest_verification_fails_closed_after_snapshot_mutation(tmp_path):
    root = _snapshot(tmp_path)
    manifest = tmp_path / "foundation-snapshot-integrity.json"
    written = write_foundation_snapshot_manifest(
        root,
        manifest,
        profile_id="wan-a14b",
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision="c" * 40,
    )

    verified = verify_foundation_snapshot_manifest(root, manifest)
    assert verified.tree_sha256 == written.tree_sha256

    (root / "transformer" / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(
        FoundationSnapshotIntegrityError,
        match="bytes or provenance changed",
    ):
        verify_foundation_snapshot_manifest(root, manifest)


def test_manifest_verification_rejects_rewritten_provenance(tmp_path):
    root = _snapshot(tmp_path)
    manifest = tmp_path / "foundation-snapshot-integrity.json"
    write_foundation_snapshot_manifest(
        root,
        manifest,
        profile_id="wan-a14b",
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision="d" * 40,
    )
    payload = json.loads(manifest.read_text())
    payload["model_id"] = "borrowed/substitute"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(
        FoundationSnapshotIntegrityError,
        match="bytes or provenance changed",
    ):
        verify_foundation_snapshot_manifest(root, manifest)


def test_snapshot_integrity_hashes_symlink_target_bytes(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"original-blob")
    root = tmp_path / "snapshot"
    root.mkdir()
    link = root / "weights.bin"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    original = inspect_foundation_snapshot(
        root,
        profile_id="wan-5b",
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        revision="e" * 40,
    )
    assert original.files[0]["symlink"] is True

    blob.write_bytes(b"changed-blob")
    changed = inspect_foundation_snapshot(
        root,
        profile_id="wan-5b",
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        revision="e" * 40,
    )
    assert changed.tree_sha256 != original.tree_sha256
