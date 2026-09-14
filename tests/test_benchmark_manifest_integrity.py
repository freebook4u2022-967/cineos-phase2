from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from cineos.film.benchmark_manifest_integrity import (
    validate_persisted_benchmark_manifest,
)
from cineos.film.exceptions import AssemblyError


def _receipt(tmp_path):
    manifest = tmp_path / "benchmark.json"
    snapshot = {
        "schema": "cineos-gpu-connected-benchmark/0.3",
        "benchmark_id": "competitive-5shot",
        "profile_id": "wan22-a14b",
        "origin": "external-open-pretrained:wan2.2",
        "shot_count": 5,
        "chain_sha256": "a" * 64,
        "total_output_bytes": 500,
        "elapsed_seconds": 42.5,
        "manifest_path": str(manifest),
        "quality_gate_applied": True,
        "quality_reports": [{"shot_id": f"shot-{index}"} for index in range(5)],
        "dialogue_scope_declared": True,
        "dialogue_shot_ids": ["shot-2"],
        "production_gpu_evidence": True,
        "production_quality_evidence": True,
        "evidence_tier": "production-gpu-quality-gated",
        "shots": [{"output_sha256": f"{index + 1:064x}"} for index in range(5)],
    }
    receipt = SimpleNamespace(
        manifest_path=str(manifest),
        to_dict=lambda: copy.deepcopy(snapshot),
    )
    payload = copy.deepcopy(snapshot)
    payload["foundation_profile"] = {
        "profile_id": "wan22-a14b",
        "origin": "external-open-pretrained:wan2.2",
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return receipt, manifest, payload


def test_manifest_integrity_accepts_exact_persisted_receipt(tmp_path) -> None:
    receipt, _manifest, payload = _receipt(tmp_path)

    assert validate_persisted_benchmark_manifest(receipt) == payload


def test_manifest_integrity_rejects_mutated_receipt_field(tmp_path) -> None:
    receipt, _manifest, _payload = _receipt(tmp_path)
    original = receipt.to_dict()
    original["chain_sha256"] = "b" * 64
    receipt.to_dict = lambda: copy.deepcopy(original)

    with pytest.raises(
        AssemblyError, match="does not match receipt field 'chain_sha256'"
    ):
        validate_persisted_benchmark_manifest(receipt)


def test_manifest_integrity_rejects_mutated_persisted_quality_evidence(
    tmp_path,
) -> None:
    receipt, manifest, payload = _receipt(tmp_path)
    payload["quality_reports"][0]["shot_id"] = "substituted-shot"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        AssemblyError, match="does not match receipt field 'quality_reports'"
    ):
        validate_persisted_benchmark_manifest(receipt)


def test_manifest_integrity_rejects_substituted_foundation_profile(tmp_path) -> None:
    receipt, manifest, payload = _receipt(tmp_path)
    payload["foundation_profile"]["profile_id"] = "wan22-5b"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssemblyError, match="foundation profile does not match"):
        validate_persisted_benchmark_manifest(receipt)


def test_manifest_integrity_rejects_unbound_top_level_evidence(tmp_path) -> None:
    receipt, manifest, payload = _receipt(tmp_path)
    payload["seedance_parity_claim"] = {
        "passed": True,
        "source": "unbound-external-claim",
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssemblyError, match="contains unbound fields"):
        validate_persisted_benchmark_manifest(receipt)


def test_manifest_integrity_rejects_unbound_quality_override(tmp_path) -> None:
    receipt, manifest, payload = _receipt(tmp_path)
    payload["quality_override"] = {"production_quality_evidence": True}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssemblyError, match="'quality_override'"):
        validate_persisted_benchmark_manifest(receipt)


def test_manifest_integrity_requires_real_manifest_file(tmp_path) -> None:
    receipt, manifest, _payload = _receipt(tmp_path)
    manifest.unlink()

    with pytest.raises(AssemblyError, match="persisted manifest does not exist"):
        validate_persisted_benchmark_manifest(receipt)


def test_manifest_integrity_rejects_unserializable_receipt(tmp_path) -> None:
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(
        json.dumps({"schema": "cineos-gpu-connected-benchmark/0.3"}),
        encoding="utf-8",
    )
    receipt = SimpleNamespace(manifest_path=str(manifest))

    with pytest.raises(AssemblyError, match="cannot be bound to persisted manifest"):
        validate_persisted_benchmark_manifest(receipt)
