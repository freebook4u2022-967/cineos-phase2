import json

import pytest

from cineos.native_video.release_seal import (
    ReleaseSealError,
    build_release_chain_seal,
    load_release_chain_seal,
    seal_release_chain_file,
    verify_release_chain_file,
    verify_release_chain_seal,
)


def _key(byte: int = 7) -> bytes:
    return bytes([byte]) * 32


def test_release_chain_seal_authenticates_exact_bytes() -> None:
    payload = b'{"schema":"cineos-release-chain/0.1","entries":[]}\n'
    seal = build_release_chain_seal(payload, key=_key(), key_id="kms-prod-v1")

    verify_release_chain_seal(
        payload,
        seal,
        key=_key(),
        expected_key_id="kms-prod-v1",
    )


def test_release_chain_seal_rejects_recomputed_unkeyed_tampering() -> None:
    original = b'{"entries":[{"release_id":"v1"}]}\n'
    tampered = b'{"entries":[{"release_id":"v2"}]}\n'
    seal = build_release_chain_seal(original, key=_key(), key_id="prod")

    with pytest.raises(ReleaseSealError, match="digest mismatch"):
        verify_release_chain_seal(tampered, seal, key=_key())


def test_release_chain_seal_rejects_wrong_secret() -> None:
    payload = b"release-chain-bytes"
    seal = build_release_chain_seal(payload, key=_key(1), key_id="prod")

    with pytest.raises(ReleaseSealError, match="authentication failed"):
        verify_release_chain_seal(payload, seal, key=_key(2))


def test_release_chain_seal_rejects_key_id_mismatch() -> None:
    payload = b"release-chain-bytes"
    seal = build_release_chain_seal(payload, key=_key(), key_id="prod-v2")

    with pytest.raises(ReleaseSealError, match="key_id mismatch"):
        verify_release_chain_seal(
            payload,
            seal,
            key=_key(),
            expected_key_id="prod-v1",
        )


def test_release_chain_file_round_trip_and_tamper_detection(tmp_path) -> None:
    chain_path = tmp_path / "release-chain.json"
    seal_path = tmp_path / "release-chain.seal.json"
    chain_path.write_text('{"entries":[]}\n', encoding="utf-8")

    seal_release_chain_file(
        chain_path,
        seal_path,
        key=_key(),
        key_id="vault-key-v1",
    )
    verified = verify_release_chain_file(
        chain_path,
        seal_path,
        key=_key(),
        expected_key_id="vault-key-v1",
    )
    assert verified == load_release_chain_seal(seal_path)

    chain_path.write_text('{"entries":["tampered"]}\n', encoding="utf-8")
    with pytest.raises(ReleaseSealError, match="digest mismatch"):
        verify_release_chain_file(chain_path, seal_path, key=_key())


def test_release_seal_file_contains_no_secret_key(tmp_path) -> None:
    chain_path = tmp_path / "release-chain.json"
    seal_path = tmp_path / "release-chain.seal.json"
    secret = b"super-secret-production-key-material"
    chain_path.write_text('{"entries":[]}\n', encoding="utf-8")

    seal_release_chain_file(chain_path, seal_path, key=secret, key_id="prod")
    persisted = seal_path.read_bytes()
    assert secret not in persisted
    document = json.loads(persisted)
    assert set(document) == {"schema", "key_id", "chain_sha256", "hmac_sha256"}


def test_release_chain_seal_requires_256_bit_key() -> None:
    with pytest.raises(ReleaseSealError, match="at least 32 bytes"):
        build_release_chain_seal(b"payload", key=b"too-short", key_id="prod")
