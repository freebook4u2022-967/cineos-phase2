import pytest

from cineos.native_video import release_registry
from cineos.native_video.release_chain import append_release
from cineos.native_video.release_registry import (
    ACTIVATION_LOCK_FILE,
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
    assert not (tmp_path / ACTIVATION_LOCK_FILE).exists()


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
    assert not (tmp_path / ACTIVATION_LOCK_FILE).exists()
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

    with pytest.raises(ReleaseRegistryError, match="SHA-256"):
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
        expected_active_generation_id=first.generation_id,
    )

    assert first.generation_id != second.generation_id
    assert second.seal.key_id == "kms-prod-v2"
    with pytest.raises(ReleaseRegistryError, match="failed authentication"):
        load_verified_release_snapshot(
            tmp_path,
            key=_key(1),
            expected_key_id="kms-prod-v1",
        )


def test_trusted_generation_guard_rejects_valid_snapshot_rollback(tmp_path) -> None:
    first = commit_release_snapshot(
        _entries("film-v1"),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )
    second_entries = append_release(
        first.entries,
        release_id="film-v2",
        receipt_sha256=_sha("c"),
        native_model_manifest_sha256=_sha("d"),
    )
    second = commit_release_snapshot(
        second_entries,
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
        expected_active_generation_id=first.generation_id,
    )

    # Simulate an attacker rolling CURRENT back to a historical snapshot whose
    # chain and HMAC seal are both still completely valid.
    (tmp_path / ACTIVE_SNAPSHOT_FILE).write_text(
        first.generation_id + "\n", encoding="utf-8"
    )

    historical = load_verified_release_snapshot(tmp_path, key=_key())
    assert historical.generation_id == first.generation_id

    with pytest.raises(ReleaseRegistryError, match="trusted generation"):
        load_verified_release_snapshot(
            tmp_path,
            key=_key(),
            expected_generation_id=second.generation_id,
        )


def test_trusted_generation_guard_accepts_current_generation(tmp_path) -> None:
    committed = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )

    loaded = load_verified_release_snapshot(
        tmp_path,
        key=_key(),
        expected_generation_id=committed.generation_id.upper(),
    )
    assert loaded.generation_id == committed.generation_id


def test_trusted_generation_guard_rejects_malformed_trust_anchor(tmp_path) -> None:
    commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )

    with pytest.raises(ReleaseRegistryError, match="SHA-256"):
        load_verified_release_snapshot(
            tmp_path,
            key=_key(),
            expected_generation_id="not-a-generation",
        )


def test_activation_cas_rejects_stale_release_controller(tmp_path) -> None:
    first = commit_release_snapshot(
        _entries("film-v1"),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )
    second_entries = append_release(
        first.entries,
        release_id="film-v2",
        receipt_sha256=_sha("c"),
        native_model_manifest_sha256=_sha("d"),
    )
    second = commit_release_snapshot(
        second_entries,
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
        expected_active_generation_id=first.generation_id,
    )

    stale_entries = append_release(
        first.entries,
        release_id="film-v3-stale",
        receipt_sha256=_sha("e"),
        native_model_manifest_sha256=_sha("f"),
    )
    with pytest.raises(ReleaseRegistryError, match="expected activation generation"):
        commit_release_snapshot(
            stale_entries,
            tmp_path,
            key=_key(),
            key_id="kms-prod-v1",
            expected_active_generation_id=first.generation_id,
        )

    loaded = load_verified_release_snapshot(
        tmp_path,
        key=_key(),
        expected_generation_id=second.generation_id,
    )
    assert loaded.generation_id == second.generation_id
    assert not (tmp_path / ACTIVATION_LOCK_FILE).exists()


def test_activation_cas_rejects_missing_or_malformed_trust_anchor(tmp_path) -> None:
    with pytest.raises(ReleaseRegistryError, match="missing"):
        commit_release_snapshot(
            _entries(),
            tmp_path,
            key=_key(),
            key_id="kms-prod-v1",
            expected_active_generation_id=_sha("a"),
        )

    with pytest.raises(ReleaseRegistryError, match="SHA-256"):
        commit_release_snapshot(
            _entries(),
            tmp_path,
            key=_key(),
            key_id="kms-prod-v1",
            expected_active_generation_id="not-a-generation",
        )
    assert not (tmp_path / ACTIVATION_LOCK_FILE).exists()


def test_activation_lock_fails_closed_on_concurrent_writer(tmp_path) -> None:
    first = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )
    (tmp_path / ACTIVATION_LOCK_FILE).write_text("pid=999999\n", encoding="utf-8")

    with pytest.raises(ReleaseRegistryError, match="already locked"):
        commit_release_snapshot(
            first.entries,
            tmp_path,
            key=_key(),
            key_id="kms-prod-v1",
            expected_active_generation_id=first.generation_id,
        )

    assert (tmp_path / ACTIVE_SNAPSHOT_FILE).read_text(encoding="utf-8").strip() == (
        first.generation_id
    )
