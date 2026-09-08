"""Regression coverage for the quality-first production GPU entrypoint."""

from types import SimpleNamespace

import pytest

from cineos.atlas import quality_first_gpu_benchmark_cli as cli
from cineos.atlas.foundation_profiles import (
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
)
from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.gpu_preflight import GPUDeviceProfile
from cineos.atlas.native_request import NativeShotRequest


def _gpu(total: float, free: float | None = None) -> GPUDeviceProfile:
    return GPUDeviceProfile(
        index=0,
        name="test-gpu",
        compute_capability=(9, 0),
        total_vram_gb=total,
        free_vram_gb=total if free is None else free,
        supports_bfloat16=True,
    )


def _request(index: int, *, refs=("hero", "partner")) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="quality-first",
        camera={"movement": "whip_pan"},
        characters=[{"character_id": "hero"}, {"character_id": "partner"}],
        environment={"lighting": "day_to_night transition"},
        wardrobe=[],
        props=[{"prop_id": "case", "action": "throwing"}],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={
            "action": "walk while throwing case",
            "gesture_tracks": [{"character_id": "hero", "action": "gripping"}],
            "dialogue_timing": [
                {"speaker_id": "hero", "start_seconds": 0.1, "end_seconds": 0.8}
            ],
        },
        approved_reference_ids=list(refs),
        deterministic_seed=7000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={
            "competitive_challenges": [
                "identity_consistency",
                "multi_character_interaction",
                "hands_anatomy",
                "walking_running",
                "dialogue_lip_sync",
                "object_interaction",
                "fast_camera_movement",
                "lighting_changes",
                "physics",
            ]
        },
    )
    request.refresh_hash()
    return request


def _receipt(profile=WAN22_I2V_A14B_PROFILE):
    return SimpleNamespace(
        profile_id=profile.profile_id,
        origin=profile.origin,
        production_gpu_evidence=True,
        production_quality_evidence=True,
        evidence_tier="production-gpu-quality-gated",
    )


def test_quality_first_entrypoint_routes_80gb_runner_to_a14b(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: object())

    def fake_run(benchmark_id, requests, profile, **kwargs):
        captured["profile"] = profile
        return _receipt(profile)

    monkeypatch.setattr(
        cli, "run_production_quality_retry_connected_gpu_benchmark", fake_run
    )
    requests = [_request(index) for index in range(5)]

    cli.run_quality_first_production_benchmark(
        "quality-first",
        requests,
        output_dir=tmp_path,
        reference_manifest="references.json",
        devices=(_gpu(96.0, 90.0),),
    )

    assert captured["profile"] is WAN22_I2V_A14B_PROFILE
    assert "wan2.2-i2v-a14b" in (tmp_path / "foundation-selection.json").read_text()


def test_quality_first_entrypoint_preserves_5b_fallback_on_48gb_runner(
    monkeypatch, tmp_path
):
    captured = {}
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: object())

    def fake_run(benchmark_id, requests, profile, **kwargs):
        captured["profile"] = profile
        return _receipt(profile)

    monkeypatch.setattr(
        cli, "run_production_quality_retry_connected_gpu_benchmark", fake_run
    )
    requests = [_request(index) for index in range(5)]

    cli.run_quality_first_production_benchmark(
        "quality-first",
        requests,
        output_dir=tmp_path,
        reference_manifest="references.json",
        devices=(_gpu(48.0, 44.0),),
    )

    assert captured["profile"] is WAN22_TI2V_5B_PROFILE
    evidence = (tmp_path / "foundation-selection.json").read_text()
    assert '"fallback_used": true' in evidence


def test_quality_first_entrypoint_rejects_receipt_for_different_profile(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: object())

    def fake_run(benchmark_id, requests, profile, **kwargs):
        return _receipt(WAN22_TI2V_5B_PROFILE)

    monkeypatch.setattr(
        cli, "run_production_quality_retry_connected_gpu_benchmark", fake_run
    )
    requests = [_request(index) for index in range(5)]

    with pytest.raises(GPUProductionBenchmarkCLIError, match="receipt profile"):
        cli.run_quality_first_production_benchmark(
            "quality-first",
            requests,
            output_dir=tmp_path,
            reference_manifest="references.json",
            devices=(_gpu(96.0, 90.0),),
        )


def test_quality_first_entrypoint_rejects_receipt_for_different_origin(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: object())

    def fake_run(benchmark_id, requests, profile, **kwargs):
        receipt = _receipt(profile)
        receipt.origin = "cineos_native"
        return receipt

    monkeypatch.setattr(
        cli, "run_production_quality_retry_connected_gpu_benchmark", fake_run
    )
    requests = [_request(index) for index in range(5)]

    with pytest.raises(GPUProductionBenchmarkCLIError, match="receipt origin"):
        cli.run_quality_first_production_benchmark(
            "quality-first",
            requests,
            output_dir=tmp_path,
            reference_manifest="references.json",
            devices=(_gpu(96.0, 90.0),),
        )
