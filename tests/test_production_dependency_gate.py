from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from cineos.atlas.foundation_profiles import (
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
)
from cineos.atlas.production_dependency_gate import (
    CUDADeviceCapacity,
    IdentityAssetRequirement,
    ProductionDependencyGateError,
    collect_cuda_capacity,
    evaluate_production_dependencies,
    parse_nvidia_smi_capacity,
)


def _asset(tmp_path: Path) -> IdentityAssetRequirement:
    path = tmp_path / "hero.png"
    payload = b"approved-production-character-reference"
    path.write_bytes(payload)
    return IdentityAssetRequirement(path, hashlib.sha256(payload).hexdigest())


def test_quality_profile_is_selected_only_with_80gb_class_free_capacity(
    tmp_path: Path,
) -> None:
    report = evaluate_production_dependencies(
        cuda_devices=(
            CUDADeviceCapacity(0, "H100 96GB", 96.0, 90.0),
            CUDADeviceCapacity(1, "L40S", 48.0, 44.0),
        ),
        identity_assets=(_asset(tmp_path),),
        ffmpeg_available=True,
        ffprobe_available=True,
    )

    assert report.ready is True
    assert report.selected_profile_id == WAN22_I2V_A14B_PROFILE.profile_id
    assert report.selected_model_id == WAN22_I2V_A14B_PROFILE.provenance.model_id
    assert report.selected_revision == WAN22_I2V_A14B_PROFILE.provenance.revision
    assert report.selected_device == "cuda:0"
    assert report.blocking_dependencies == ()


def test_quality_mode_fails_closed_instead_of_silently_downgrading_to_5b(
    tmp_path: Path,
) -> None:
    report = evaluate_production_dependencies(
        cuda_devices=(CUDADeviceCapacity(0, "L40S", 48.0, 44.0),),
        identity_assets=(_asset(tmp_path),),
        ffmpeg_available=True,
        ffprobe_available=True,
    )

    assert report.ready is False
    assert report.selected_profile_id is None
    assert report.selected_model_id is None
    assert report.selected_device is None
    assert report.blocking_dependencies == ("cuda_free_vram_below_quality_floor:80GB",)


def test_explicit_fallback_mode_selects_pinned_5b_profile(tmp_path: Path) -> None:
    report = evaluate_production_dependencies(
        cuda_devices=(CUDADeviceCapacity(0, "L40S", 48.0, 44.0),),
        identity_assets=(_asset(tmp_path),),
        ffmpeg_available=True,
        ffprobe_available=True,
        quality_profile_required=False,
    )

    assert report.ready is True
    assert report.selected_profile_id == WAN22_TI2V_5B_PROFILE.profile_id
    assert report.selected_model_id == WAN22_TI2V_5B_PROFILE.provenance.model_id
    assert report.selected_device == "cuda:0"


def test_identity_asset_substitution_is_a_blocking_dependency(tmp_path: Path) -> None:
    requirement = _asset(tmp_path)
    requirement.path.write_bytes(b"substituted-character-reference")

    report = evaluate_production_dependencies(
        cuda_devices=(CUDADeviceCapacity(0, "H100 96GB", 96.0, 90.0),),
        identity_assets=(requirement,),
        ffmpeg_available=True,
        ffprobe_available=True,
    )

    assert report.ready is False
    assert len(report.identity_assets) == 1
    assert report.identity_assets[0]["hash_matches"] is False
    assert report.blocking_dependencies == (
        f"identity_asset_hash_mismatch:{requirement.path.resolve()}",
    )


def test_missing_identity_assets_and_media_tools_are_reported_together() -> None:
    report = evaluate_production_dependencies(
        cuda_devices=(CUDADeviceCapacity(0, "H100 96GB", 96.0, 90.0),),
        identity_assets=(),
        ffmpeg_available=False,
        ffprobe_available=False,
    )

    assert report.ready is False
    assert report.blocking_dependencies == (
        "approved_identity_assets_missing",
        "ffmpeg_missing",
        "ffprobe_missing",
    )


def test_identity_digest_must_be_real_hex_sha256(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal characters"):
        IdentityAssetRequirement(tmp_path / "hero.png", "g" * 64)


def test_parse_nvidia_smi_capacity_preserves_free_memory_for_selection() -> None:
    devices = parse_nvidia_smi_capacity(
        "0, NVIDIA H100 96GB HBM3, 98304, 92160\n" "1, NVIDIA L40S, 49152, 45056\n"
    )

    assert devices == (
        CUDADeviceCapacity(0, "NVIDIA H100 96GB HBM3", 96.0, 90.0),
        CUDADeviceCapacity(1, "NVIDIA L40S", 48.0, 44.0),
    )


def test_parse_nvidia_smi_capacity_rejects_ambiguous_rows() -> None:
    with pytest.raises(ProductionDependencyGateError, match="expected 4 columns"):
        parse_nvidia_smi_capacity("0, GPU, 98304")


def test_collect_cuda_capacity_treats_unavailable_nvidia_smi_as_no_gpu() -> None:
    def missing(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi")

    assert collect_cuda_capacity(run=missing) == ()


def test_collect_cuda_capacity_treats_failed_probe_as_no_gpu() -> None:
    def failed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [], 1, stdout="", stderr="driver unavailable"
        )

    assert collect_cuda_capacity(run=failed) == ()
