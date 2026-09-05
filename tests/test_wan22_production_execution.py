from __future__ import annotations

import pytest

from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.wan22_execution import Wan22ExecutionConfig, Wan22ExecutionError
from cineos.atlas import wan22_production_execution as production


def _receipt(*, device: str = "cuda") -> dict:
    return {
        "status": "rendered",
        "runtime": {"device": device},
        "foundation_profile": {
            "origin": WAN22_TI2V_5B_PROFILE.origin,
        },
        "artifact": {"sha256": "a" * 64},
    }


def test_production_gate_rejects_non_cuda_before_execution(monkeypatch, tmp_path):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return _receipt(device="cpu")

    monkeypatch.setattr(production, "run_wan22_gpu_validation", fake_run)

    with pytest.raises(Wan22ExecutionError, match="requires a CUDA device"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="cinematic tracking shot"),
            output_dir=tmp_path,
            device="cpu",
        )

    assert called is False


def test_production_gate_has_no_renderer_injection_surface():
    import inspect

    parameters = inspect.signature(production.run_wan22_production_validation).parameters
    assert "pipeline_factory" not in parameters
    assert "video_exporter" not in parameters


def test_production_receipt_explicitly_classifies_external_foundation(monkeypatch, tmp_path):
    def fake_run(config, **kwargs):
        assert kwargs["device"] == "cuda:0"
        assert "pipeline_factory" not in kwargs
        assert "video_exporter" not in kwargs
        return _receipt(device="cuda:0")

    monkeypatch.setattr(production, "run_wan22_gpu_validation", fake_run)

    receipt = production.run_wan22_production_validation(
        Wan22ExecutionConfig(prompt="two actors crossing a rainy street"),
        output_dir=tmp_path,
        device="cuda:0",
    )

    evidence = receipt["execution_evidence"]
    assert evidence["classification"] == "external_pretrained_foundation"
    assert evidence["foundation_profile_id"] == WAN22_TI2V_5B_PROFILE.profile_id
    assert evidence["injected_pipeline_factory"] is False
    assert evidence["injected_video_exporter"] is False
    assert evidence["cuda_required"] is True


def test_production_gate_rejects_foundation_origin_drift(monkeypatch, tmp_path):
    receipt = _receipt()
    receipt["foundation_profile"]["origin"] = "cineos_native"
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: receipt,
    )

    with pytest.raises(Wan22ExecutionError, match="foundation origin"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="cinematic close-up"),
            output_dir=tmp_path,
        )


def test_production_gate_rejects_missing_artifact_binding(monkeypatch, tmp_path):
    receipt = _receipt()
    receipt["artifact"]["sha256"] = "not-a-digest"
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: receipt,
    )

    with pytest.raises(Wan22ExecutionError, match="SHA-256"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="cinematic close-up"),
            output_dir=tmp_path,
        )
