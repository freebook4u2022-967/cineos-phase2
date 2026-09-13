from __future__ import annotations

import pytest

from cineos.atlas import wan22_production_execution as production
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.wan22_execution import Wan22ExecutionConfig, Wan22ExecutionError


def _receipt(*, device: str = "cuda") -> dict:
    return {
        "status": "rendered",
        "runtime": {"device": device},
        "foundation_profile": WAN22_TI2V_5B_PROFILE.snapshot(),
        "artifact": {"sha256": "a" * 64},
        "request_hash": "b" * 64,
        "output_path": "/tmp/fake-wan22-production.mp4",
    }


@pytest.fixture(autouse=True)
def _stub_artifact_media_probe(monkeypatch):
    monkeypatch.setattr(
        production,
        "_probe_video_artifact",
        lambda output_path: {
            "probe": "ffprobe-count-frames",
            "codec_name": "h264",
            "width": 1280,
            "height": 704,
            "avg_frame_rate": "24",
            "fps": 24.0,
            "decoded_frame_count": 121,
        },
    )


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

    parameters = inspect.signature(
        production.run_wan22_production_validation
    ).parameters
    assert "pipeline_factory" not in parameters
    assert "video_exporter" not in parameters


def test_production_receipt_explicitly_classifies_external_foundation(
    monkeypatch, tmp_path
):
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
    assert evidence["schema"] == "cineos-wan22-production-execution/1.2"
    assert evidence["classification"] == "external_pretrained_foundation"
    assert evidence["foundation_profile_id"] == WAN22_TI2V_5B_PROFILE.profile_id
    assert evidence["foundation_model_id"] == WAN22_TI2V_5B_PROFILE.provenance.model_id
    assert evidence["foundation_revision"] == WAN22_TI2V_5B_PROFILE.provenance.revision
    assert (
        evidence["foundation_license_id"] == WAN22_TI2V_5B_PROFILE.provenance.license_id
    )
    assert evidence["injected_pipeline_factory"] is False
    assert evidence["injected_video_exporter"] is False
    assert evidence["cuda_required"] is True
    assert evidence["artifact_media"]["decoded_frame_count"] == 121
    assert evidence["artifact_media"]["width"] == 1280
    assert evidence["artifact_media"]["height"] == 704
    assert evidence["artifact_media"]["fps"] == 24.0


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


def test_production_gate_rejects_foundation_profile_id_drift(monkeypatch, tmp_path):
    receipt = _receipt()
    receipt["foundation_profile"]["profile_id"] = "different-external-profile"
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: receipt,
    )

    with pytest.raises(Wan22ExecutionError, match="profile id"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="cinematic close-up"),
            output_dir=tmp_path,
        )


@pytest.mark.parametrize("field", ["model_id", "revision", "license_id", "source_url"])
def test_production_gate_rejects_pinned_foundation_provenance_drift(
    monkeypatch, tmp_path, field
):
    receipt = _receipt()
    receipt["foundation_profile"]["provenance"][field] = "drifted-value"
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: receipt,
    )

    with pytest.raises(Wan22ExecutionError, match="foundation provenance"):
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


def test_production_gate_rejects_non_hex_artifact_binding(monkeypatch, tmp_path):
    receipt = _receipt()
    receipt["artifact"]["sha256"] = "z" * 64
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


def test_production_gate_rejects_invalid_request_hash_binding(monkeypatch, tmp_path):
    receipt = _receipt()
    receipt["request_hash"] = "not-a-request-hash"
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: receipt,
    )

    with pytest.raises(Wan22ExecutionError, match="request hash"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="cinematic close-up"),
            output_dir=tmp_path,
        )


def test_production_gate_rejects_missing_rendered_output_path(monkeypatch, tmp_path):
    receipt = _receipt()
    receipt.pop("output_path")
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: receipt,
    )

    with pytest.raises(Wan22ExecutionError, match="rendered output path"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="cinematic close-up"),
            output_dir=tmp_path,
        )


def test_production_gate_rejects_encoded_geometry_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: _receipt(),
    )
    monkeypatch.setattr(
        production,
        "_probe_video_artifact",
        lambda output_path: {
            "probe": "ffprobe-count-frames",
            "codec_name": "h264",
            "width": 960,
            "height": 544,
            "avg_frame_rate": "24",
            "fps": 24.0,
            "decoded_frame_count": 121,
        },
    )

    with pytest.raises(Wan22ExecutionError, match="geometry"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="fast tracking shot"),
            output_dir=tmp_path,
        )


def test_production_gate_rejects_encoded_frame_rate_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: _receipt(),
    )
    monkeypatch.setattr(
        production,
        "_probe_video_artifact",
        lambda output_path: {
            "probe": "ffprobe-count-frames",
            "codec_name": "h264",
            "width": 1280,
            "height": 704,
            "avg_frame_rate": "25",
            "fps": 25.0,
            "decoded_frame_count": 121,
        },
    )

    with pytest.raises(Wan22ExecutionError, match="frame rate"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="fast tracking shot"),
            output_dir=tmp_path,
        )


def test_production_gate_rejects_decoded_frame_count_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(
        production,
        "run_wan22_gpu_validation",
        lambda *args, **kwargs: _receipt(),
    )
    monkeypatch.setattr(
        production,
        "_probe_video_artifact",
        lambda output_path: {
            "probe": "ffprobe-count-frames",
            "codec_name": "h264",
            "width": 1280,
            "height": 704,
            "avg_frame_rate": "24",
            "fps": 24.0,
            "decoded_frame_count": 120,
        },
    )

    with pytest.raises(Wan22ExecutionError, match="decoded frame count"):
        production.run_wan22_production_validation(
            Wan22ExecutionConfig(prompt="fast tracking shot"),
            output_dir=tmp_path,
        )
