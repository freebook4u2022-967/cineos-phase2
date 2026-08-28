from __future__ import annotations

import hashlib
import os

import pytest

from cineos.native_image.artifact_verification import (
    ModelArtifactVerificationError,
    sha256_file,
)


def test_sha256_file_hashes_stable_regular_file(tmp_path) -> None:
    artifact = tmp_path / "decoder.safetensors"
    payload = b"cineos-native-model-bytes" * 4096
    artifact.write_bytes(payload)

    assert sha256_file(artifact, chunk_bytes=1024) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_rejects_symbolic_link(tmp_path) -> None:
    target = tmp_path / "weights.bin"
    target.write_bytes(b"released-weights")
    link = tmp_path / "active-weights.bin"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable in this environment: {exc}")

    with pytest.raises(ModelArtifactVerificationError, match="symbolic link"):
        sha256_file(link)


def test_sha256_file_rejects_non_regular_file(tmp_path) -> None:
    with pytest.raises(ModelArtifactVerificationError, match="not a regular file"):
        sha256_file(tmp_path)


def test_sha256_file_rejects_empty_artifact(tmp_path) -> None:
    artifact = tmp_path / "empty.bin"
    artifact.write_bytes(b"")

    with pytest.raises(ModelArtifactVerificationError, match="empty"):
        sha256_file(artifact)


def test_sha256_file_requests_close_on_exec_when_supported(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"weights")
    observed: list[int] = []
    original_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        observed.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)

    sha256_file(artifact)

    assert observed
    if hasattr(os, "O_CLOEXEC"):
        assert observed[0] & os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        assert observed[0] & os.O_NOFOLLOW
