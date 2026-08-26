import pytest

from cineos.native_video import release_registry
from cineos.native_video.release_chain import append_release
from cineos.native_video.release_registry import (
    ACTIVE_SNAPSHOT_FILE,
    ReleaseRegistryError,
    commit_release_snapshot,
    load_verified_release_snapshot,
)
from cineos.native_video.release_seal import ReleaseSealError


def _key(byte: int = 7) -> bytes:
    return bytes([byte]) * 32


def _sha(char: str) -> str:
    return char * 64


def _entries(release_id: str = "film-v1"):
    return append_release(
        (),
        release_id=release_id,
        receipt_sha256=_sha("a"),
        native_model_manifest_sha256=_sha("b"),
    )


def test_release_registry_round_trip_authenticates_active_snapshot(tmp_path) -> None:
    committed = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )

    loaded = load_verified_release_snapshot(
        tmp_path,
        key=_key(),
        expected_key_id="kms-prod-v1",
    )
    assert loaded == committed
    assert loaded.entries == _entries()
    assert (tmp_path / ACTIVE_SNAPSHOT_FILE).read_text(encoding="utf-8").strip() == (
        committed.generation_id
    )


def test_failed_snapshot_commit_keeps_previous_active_release(
    tmp_path, monkeypatch
) -> None:
    first = commit_release_snapshot(
        _entries("film-v1"),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )

    def fail_sealing(*args, **kwargs):
        raise ReleaseSealError("injected sealing failure")

    monkeypatch.setattr(release_registry, "seal_release_chain_file", fail_sealing)
    with pytest.raises(ReleaseSealError, match="injected sealing failure"):
        commit_release_snapshot(
            _entries("film-v2"),
            tmp_path,
            key=_key(),
            key_id="kms-prod-v1",
        )

    assert (tmp_path / ACTIVE_SNAPSHOT_FILE).read_text(encoding="utf-8").strip() == (
        first.generation_id
    )
    loaded = load_verified_release_snapshot(
        tmp_path,
        key=_key(),
        expected_key_id="kms-prod-v1",
    )
    assert loaded.generation_id == first.generation_id
    assert loaded.entries == first.entries


def test_release_registry_rejects_pointer_tampering(tmp_path) -> None:
    commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )
    (tmp_path / ACTIVE_SNAPSHOT_FILE).write_text("../escape\n", encoding="utf-8")

    with pytest.raises(ReleaseRegistryError, match="SHA-256 generation id"):
        load_verified_release_snapshot(tmp_path, key=_key())


def test_release_registry_rejects_wrong_authentication_key(tmp_path) -> None:
    commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(1),
        key_id="kms-prod-v1",
    )

    with pytest.raises(ReleaseRegistryError, match="failed authentication"):
        load_verified_release_snapshot(tmp_path, key=_key(2))


def test_key_rotation_creates_distinct_authenticated_generation(tmp_path) -> None:
    first = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(1),
        key_id="kms-prod-v1",
    )
    second = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(2),
        key_id="kms-prod-v2",
    )

    assert first.generation_id != second.generation_id
    assert second.seal.key_id == "kms-prod-v2"
    with pytest.raises(ReleaseRegistryError, match="failed authentication"):
        load_verified_release_snapshot(
            tmp_path,
            key=_key(1),
            expected_key_id="kms-prod-v1",
        )
