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
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
REVISION = "a" * 40


def _execution_plan(device="cuda:0"):
    return {
        "device": device,
        "dtype": "bfloat16",
        "memory_strategy": "resident",
        "enable_vae_tiling": False,
        "enable_vae_slicing": False,
        "enable_attention_slicing": False,
        "estimated_model_vram_gb": 80.0,
        "observed_total_vram_gb": 96.0,
        "observed_free_vram_gb": 90.0,
        "fit_margin_gb": 10.0,
    }


@dataclass
class _Receipt:
    profile_id: str = PROFILE_ID
    origin: str = ORIGIN
    benchmark_id: str = BENCHMARK_ID
    shot_model_id: str = MODEL_ID
    shot_revision: str = REVISION
    shot_device: str = "cuda:0"
    runtime_device: str = "cuda:0"

    def to_dict(self):
        shots = []
        for index in range(5):
            shots.append(
                {
                    "shot_id": f"shot-{index}",
                    "scene_id": "scene-1",
                    "request_hash": f"{index + 20:064x}",
                    "output_sha256": f"{index + 1:064x}",
                    "profile_id": self.profile_id,
                    "origin": self.origin,
                    "foundation": {
                        "model_id": self.shot_model_id,
                        "revision": self.shot_revision,
                        "license_id": "Apache-2.0",
                        "source_url": "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                        "foundation_name": "Wan2.2 I2V A14B",
                    },
                    "execution_plan": _execution_plan(self.shot_device),
                    "runtime_provenance": {
                        "schema": "cineos-gpu-runtime-provenance/0.1",
                        "runtime_mode": "default",
                        "production_default_runtime": True,
                        "cuda_device": self.runtime_device,
                        "dtype": "bfloat16",
                    },
                }
            )
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
            "shots": shots,
        }


def _selection_payload():
    return {
        "profile_id": PROFILE_ID,
        "origin": ORIGIN,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "fallback_used": False,
        "rejected_profiles": [],
        "execution_plan": _execution_plan(),
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
    assert evidence["schema"] == "cineos-quality-first-production-attestation/0.2"
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


def test_attestation_rejects_per_shot_foundation_revision_substitution(tmp_path):
    selection_path = _write_selection(tmp_path)

    with pytest.raises(
        ProductionBenchmarkAttestationError,
        match="revision does not match foundation selection",
    ):
        write_quality_first_production_attestation(
            tmp_path,
            selection_manifest=selection_path,
            receipt=_Receipt(shot_revision="b" * 40),
            benchmark_id=BENCHMARK_ID,
        )


def test_attestation_rejects_per_shot_execution_device_drift(tmp_path):
    selection_path = _write_selection(tmp_path)

    with pytest.raises(
        ProductionBenchmarkAttestationError,
        match="execution field 'device' does not match foundation selection",
    ):
        write_quality_first_production_attestation(
            tmp_path,
            selection_manifest=selection_path,
            receipt=_Receipt(shot_device="cuda:1", runtime_device="cuda:1"),
            benchmark_id=BENCHMARK_ID,
        )


def test_attestation_rejects_runtime_device_drift(tmp_path):
    selection_path = _write_selection(tmp_path)

    with pytest.raises(
        ProductionBenchmarkAttestationError,
        match="runtime CUDA device does not match foundation selection",
    ):
        write_quality_first_production_attestation(
            tmp_path,
            selection_manifest=selection_path,
            receipt=_Receipt(runtime_device="cuda:1"),
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
