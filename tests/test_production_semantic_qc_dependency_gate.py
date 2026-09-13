from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cineos.atlas.latentsync_syncnet_scorer import LATENTSYNC_PINNED_REVISION
from cineos.atlas.production_semantic_qc_dependency_gate import (
    PRODUCTION_SEMANTIC_QC_DEPENDENCY_SCHEMA,
    evaluate_semantic_qc_dependencies,
)
from cineos.atlas.qwen25vl_semantic_judge import QWEN25VL_MODEL_REVISION


def _qwen_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / QWEN25VL_MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"model")
    return snapshot


def test_semantic_qc_dependency_gate_accepts_exact_pinned_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _qwen_snapshot(tmp_path)
    repository = tmp_path / "LatentSync"
    repository.mkdir()
    checkpoint = tmp_path / "syncnet.model"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    monkeypatch.setattr(
        "cineos.atlas.production_semantic_qc_dependency_gate._git_head",
        lambda path: LATENTSYNC_PINNED_REVISION,
    )

    report = evaluate_semantic_qc_dependencies(
        qwen_snapshot=snapshot,
        latentsync_repository=repository,
        latentsync_checkpoint=checkpoint,
        latentsync_checkpoint_sha256=digest,
    )

    assert report.ready is True
    assert report.blockers == ()
    payload = report.to_dict()
    assert payload["schema"] == PRODUCTION_SEMANTIC_QC_DEPENDENCY_SCHEMA
    assert payload["external_components"]["qwen25vl_visual_judge"]["origin"] == (
        "external_pretrained"
    )
    assert payload["external_components"]["latentsync_syncnet"]["origin"] == (
        "external_pretrained"
    )


def test_semantic_qc_dependency_gate_fails_closed_on_revision_and_checkpoint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _qwen_snapshot(tmp_path)
    repository = tmp_path / "LatentSync"
    repository.mkdir()
    checkpoint = tmp_path / "syncnet.model"
    checkpoint.write_bytes(b"wrong-checkpoint")
    approved_digest = hashlib.sha256(b"approved-checkpoint").hexdigest()

    monkeypatch.setattr(
        "cineos.atlas.production_semantic_qc_dependency_gate._git_head",
        lambda path: "0" * 40,
    )

    report = evaluate_semantic_qc_dependencies(
        qwen_snapshot=snapshot,
        latentsync_repository=repository,
        latentsync_checkpoint=checkpoint,
        latentsync_checkpoint_sha256=approved_digest,
    )

    assert report.ready is False
    assert report.qwen_ready is True
    assert report.latentsync_repository_ready is False
    assert report.latentsync_checkpoint_ready is False
    assert len(report.blockers) == 2


def test_semantic_qc_dependency_gate_rejects_unpinned_qwen_snapshot_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "mutable-main"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"model")
    repository = tmp_path / "LatentSync"
    repository.mkdir()
    checkpoint = tmp_path / "syncnet.model"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    monkeypatch.setattr(
        "cineos.atlas.production_semantic_qc_dependency_gate._git_head",
        lambda path: LATENTSYNC_PINNED_REVISION,
    )

    report = evaluate_semantic_qc_dependencies(
        qwen_snapshot=snapshot,
        latentsync_repository=repository,
        latentsync_checkpoint=checkpoint,
        latentsync_checkpoint_sha256=digest,
    )

    assert report.ready is False
    assert report.qwen_ready is False
    assert any("Qwen2.5-VL" in blocker for blocker in report.blockers)


def test_semantic_qc_dependency_gate_rejects_invalid_expected_checkpoint_digest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        evaluate_semantic_qc_dependencies(
            qwen_snapshot=tmp_path / "qwen",
            latentsync_repository=tmp_path / "LatentSync",
            latentsync_checkpoint=tmp_path / "syncnet.model",
            latentsync_checkpoint_sha256="not-a-digest",
        )
