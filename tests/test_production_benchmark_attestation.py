"""Regression coverage for quality-first production benchmark attestation."""

import json
from dataclasses import dataclass

import pytest

from cineos.atlas.production_benchmark_attestation import (
    DEFAULT_FILENAME,
    ProductionBenchmarkAttestationError,
    remove_stale_quality_first_attestation,
    verify_quality_first_production_attestation,
    write_quality_first_production_attestation,
)


PROFILE_ID = "wan2.2-i2v-a14b"
ORIGIN = "external_pretrained_foundation"
BENCHMARK_ID = "quality-first-production"


@dataclass
class _Receipt:
    profile_id: str = PROFILE_ID
    origin: str = ORIGIN
    benchmark_id: str = BENCHMARK_ID

    def to_dict(self):
        return {
            "schema": "cineos-gpu-connected-benchmark/0.3",
            "benchmark_id": self.benchmark_id,
            "profile_id": self.profile_id,
            "origin": self.origin,
            "shot_count": 5,
            "chain_sha256": "1" * 64,
            "production_gpu_evidence": True,
            "production_quality_evidence": True,
            "evidence_tier": "production-gpu-quality-gated",
            "shots": [
                {"shot_id": f"shot-{index}", "output_sha256": f"{index + 1:064x}"}
                for index in range(5)
            ],
        }


def _selection_payload():
    return {
        "profile_id": PROFILE_ID,
        "origin": ORIGIN,
        "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "revision": "a" * 40,
        "fallback_used": False,
        "rejected_profiles": [],
        "execution_plan": {
            "device": "cuda:0",
            "dtype": "bfloat16",
            "memory_strategy": "resident",
            "estimated_model_vram_gb": 80.0,
            "observed_total_vram_gb": 96.0,
            "observed_free_vram_gb": 90.0,
            "fit_margin_gb": 10.0,
        },
    }


def _write_selection(tmp_path):
    path = tmp_path / "foundation-selection.json"
    path.write_text(
        json.dumps(_selection_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_attestation_binds_selection_sidecar_to_connected_benchmark(tmp_path):
    selection_path = _write_selection(tmp_path)

    attestation_path = write_quality_first_production_attestation(
        tmp_path,
        selection_manifest=selection_path,
        receipt=_Receipt(),
        benchmark_id=BENCHMARK_ID,
    )

    evidence = verify_quality_first_production_attestation(attestation_path)
    assert evidence["benchmark_id"] == BENCHMARK_ID
    assert evidence["foundation_selection"]["profile_id"] == PROFILE_ID
    assert evidence["connected_benchmark"]["shot_count"] == 5
    assert len(evidence["selection_manifest_sha256"]) == 64
    assert len(evidence["connected_evidence_sha256"]) == 64
    assert len(evidence["attestation_sha256"]) == 64


def test_attestation_rejects_selection_sidecar_substitution(tmp_path):
    selection_path = _write_selection(tmp_path)
    attestation_path = write_quality_first_production_attestation(
        tmp_path,
        selection_manifest=selection_path,
        receipt=_Receipt(),
        benchmark_id=BENCHMARK_ID,
    )

    substituted = _selection_payload()
    substituted["profile_id"] = "wan2.2-ti2v-5b"
    selection_path.write_text(json.dumps(substituted), encoding="utf-8")

    with pytest.raises(
        ProductionBenchmarkAttestationError,
        match="selection sidecar digest",
    ):
        verify_quality_first_production_attestation(attestation_path)


def test_attestation_rejects_embedded_connected_evidence_tampering(tmp_path):
    selection_path = _write_selection(tmp_path)
    attestation_path = write_quality_first_production_attestation(
        tmp_path,
        selection_manifest=selection_path,
        receipt=_Receipt(),
        benchmark_id=BENCHMARK_ID,
    )

    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["connected_benchmark"]["chain_sha256"] = "f" * 64
    attestation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ProductionBenchmarkAttestationError,
        match="root digest",
    ):
        verify_quality_first_production_attestation(attestation_path)


def test_attestation_rejects_selection_and_receipt_profile_mismatch(tmp_path):
    selection_path = _write_selection(tmp_path)

    with pytest.raises(
        ProductionBenchmarkAttestationError,
        match="profile_id does not match",
    ):
        write_quality_first_production_attestation(
            tmp_path,
            selection_manifest=selection_path,
            receipt=_Receipt(profile_id="wan2.2-ti2v-5b"),
            benchmark_id=BENCHMARK_ID,
        )


def test_attestation_rejects_receipt_benchmark_id_replay(tmp_path):
    selection_path = _write_selection(tmp_path)

    with pytest.raises(
        ProductionBenchmarkAttestationError,
        match="benchmark_id does not match invocation",
    ):
        write_quality_first_production_attestation(
            tmp_path,
            selection_manifest=selection_path,
            receipt=_Receipt(benchmark_id="old-run"),
            benchmark_id=BENCHMARK_ID,
        )


def test_stale_success_attestation_is_removed_before_new_attempt(tmp_path):
    stale = tmp_path / DEFAULT_FILENAME
    stale.write_text("stale-success", encoding="utf-8")

    remove_stale_quality_first_attestation(tmp_path)

    assert not stale.exists()
