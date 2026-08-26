import pytest

from cineos.native_video.release_chain import append_release
from cineos.native_video.release_recovery import (
    read_activation_lock,
    recover_activation_lock,
)
from cineos.native_video.release_registry import (
    ACTIVATION_LOCK_FILE,
    ReleaseRegistryError,
    commit_release_snapshot,
)


def _key(byte: int = 7) -> bytes:
    return bytes([byte]) * 32


def _sha(char: str) -> str:
    return char * 64


def _entries():
    return append_release(
        (),
        release_id="film-v1",
        receipt_sha256=_sha("a"),
        native_model_manifest_sha256=_sha("b"),
    )


def _leave_stale_lock(tmp_path) -> str:
    contents = "pid=999999\n"
    (tmp_path / ACTIVATION_LOCK_FILE).write_text(contents, encoding="utf-8")
    return contents


def test_recovery_authenticates_active_snapshot_before_removing_stale_lock(tmp_path) -> None:
    active = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )
    observed = _leave_stale_lock(tmp_path)

    recovered = recover_activation_lock(
        tmp_path,
        key=_key(),
        expected_key_id="kms-prod-v1",
        expected_generation_id=active.generation_id,
        expected_lock_contents=observed,
    )

    assert recovered.lock_contents == observed
    assert recovered.verified_snapshot.generation_id == active.generation_id
    assert not (tmp_path / ACTIVATION_LOCK_FILE).exists()


def test_recovery_rejects_lock_changed_since_operator_inspection(tmp_path) -> None:
    active = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )
    _leave_stale_lock(tmp_path)

    with pytest.raises(ReleaseRegistryError, match="changed since recovery inspection"):
        recover_activation_lock(
            tmp_path,
            key=_key(),
            expected_generation_id=active.generation_id,
            expected_lock_contents="pid=123456\n",
        )

    assert (tmp_path / ACTIVATION_LOCK_FILE).exists()


def test_recovery_rejects_untrusted_active_generation_and_preserves_lock(tmp_path) -> None:
    commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(),
        key_id="kms-prod-v1",
    )
    observed = _leave_stale_lock(tmp_path)

    with pytest.raises(ReleaseRegistryError, match="trusted generation"):
        recover_activation_lock(
            tmp_path,
            key=_key(),
            expected_generation_id=_sha("f"),
            expected_lock_contents=observed,
        )

    assert read_activation_lock(tmp_path) == observed


def test_recovery_rejects_wrong_authentication_key_and_preserves_lock(tmp_path) -> None:
    active = commit_release_snapshot(
        _entries(),
        tmp_path,
        key=_key(1),
        key_id="kms-prod-v1",
    )
    observed = _leave_stale_lock(tmp_path)

    with pytest.raises(ReleaseRegistryError, match="failed authentication"):
        recover_activation_lock(
            tmp_path,
            key=_key(2),
            expected_generation_id=active.generation_id,
            expected_lock_contents=observed,
        )

    assert read_activation_lock(tmp_path) == observed


def test_read_activation_lock_rejects_missing_and_empty_lock(tmp_path) -> None:
    with pytest.raises(ReleaseRegistryError, match="missing"):
        read_activation_lock(tmp_path)

    (tmp_path / ACTIVATION_LOCK_FILE).write_text("", encoding="utf-8")
    with pytest.raises(ReleaseRegistryError, match="empty"):
        read_activation_lock(tmp_path)
