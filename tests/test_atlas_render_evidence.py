import hashlib
import json
from pathlib import Path

import pytest

from cineos.atlas.render_evidence import (
    RenderEvidenceError,
    collect_render_evidence,
    sha256_file,
    write_render_evidence,
)


def _collect(path: Path):
    return collect_render_evidence(
        artifact_path=path,
        shot_id="shot-007",
        scene_id="scene-003",
        frame_count=81,
        seed=404,
        request_hash="request-sha256",
        foundation_model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        foundation_revision="abc123",
        foundation_license_id="Apache-2.0",
        device="cuda:1",
        dtype="bfloat16",
        memory_strategy="model_cpu_offload",
    )


def test_collect_render_evidence_binds_artifact_and_runtime(tmp_path):
    artifact = tmp_path / "scene-003-shot-007.mp4"
    payload = b"real-render-bytes-for-integrity-test"
    artifact.write_bytes(payload)

    evidence = _collect(artifact)

    assert evidence.schema == "cineos-render-evidence/0.1"
    assert evidence.artifact_bytes == len(payload)
    assert evidence.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert evidence.request_hash == "request-sha256"
    assert evidence.foundation_model_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert evidence.foundation_license_id == "Apache-2.0"
    assert evidence.device == "cuda:1"
    assert evidence.dtype == "bfloat16"
    assert evidence.memory_strategy == "model_cpu_offload"


def test_write_render_evidence_creates_deterministic_sidecar(tmp_path):
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"video")
    evidence = _collect(artifact)

    sidecar = write_render_evidence(evidence)
    document = json.loads(sidecar.read_text(encoding="utf-8"))

    assert sidecar == tmp_path / "shot.render-evidence.json"
    assert document == evidence.to_dict()
    assert not list(tmp_path.glob("*.tmp"))


def test_collect_render_evidence_rejects_missing_artifact(tmp_path):
    with pytest.raises(RenderEvidenceError, match="does not exist"):
        _collect(tmp_path / "missing.mp4")


def test_collect_render_evidence_rejects_empty_artifact(tmp_path):
    artifact = tmp_path / "empty.mp4"
    artifact.touch()

    with pytest.raises(RenderEvidenceError, match="empty"):
        _collect(artifact)


def test_collect_render_evidence_rejects_nonpositive_frame_count(tmp_path):
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"video")

    with pytest.raises(RenderEvidenceError, match="frame_count"):
        collect_render_evidence(
            artifact_path=artifact,
            shot_id="shot",
            scene_id="scene",
            frame_count=0,
            seed=1,
            request_hash="hash",
            foundation_model_id="model",
            foundation_revision=None,
            foundation_license_id=None,
            device="cuda",
            dtype="float16",
            memory_strategy="resident",
        )


def test_sha256_file_streams_exact_content(tmp_path):
    artifact = tmp_path / "large.mp4"
    payload = b"0123456789" * 1024
    artifact.write_bytes(payload)

    assert sha256_file(artifact, chunk_size=17) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_rejects_invalid_chunk_size(tmp_path):
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"video")

    with pytest.raises(ValueError, match="positive"):
        sha256_file(artifact, chunk_size=0)
