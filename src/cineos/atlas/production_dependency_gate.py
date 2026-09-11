"""Fail-closed external dependency gate for real CINEOS production GPU runs.

This module does not claim renderer quality and does not emulate CUDA. It verifies the
external prerequisites that must exist before CINEOS can truthfully run the pinned
quality-first video foundations: usable NVIDIA memory, approved hash-pinned identity
assets, and the FFmpeg tools required by the measured production pipeline.

The selected Wan foundation remains an external pretrained foundation. The gate only
reports whether the local environment can execute the already-pinned CINEOS production
path without silently substituting a weaker model or unapproved identity material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .foundation_profiles import (
    EXTERNAL_PRETRAINED_FOUNDATION,
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
    FoundationExecutionProfile,
)

SCHEMA = "cineos-production-dependency-readiness/0.1"
EXIT_DEPENDENCY_BLOCKED = 2


class ProductionDependencyGateError(RuntimeError):
    """Raised when dependency evidence itself is malformed or cannot be collected."""


@dataclass(frozen=True, slots=True)
class CUDADeviceCapacity:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("CUDA device index cannot be negative")
        if not self.name.strip():
            raise ValueError("CUDA device name cannot be empty")
        if self.total_vram_gb <= 0 or self.free_vram_gb < 0:
            raise ValueError("CUDA VRAM values must be non-negative with positive total")
        if self.free_vram_gb > self.total_vram_gb + 0.25:
            raise ValueError("CUDA free VRAM cannot exceed total VRAM")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentityAssetRequirement:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        digest = self.sha256.strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("identity asset SHA-256 must be 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True)
class ProductionDependencyReadiness:
    ready: bool
    selected_profile_id: str | None
    selected_model_id: str | None
    selected_revision: str | None
    selected_device: str | None
    external_foundation_origin: str
    blocking_dependencies: tuple[str, ...]
    cuda_devices: tuple[dict[str, Any], ...]
    identity_assets: tuple[dict[str, Any], ...]
    ffmpeg_available: bool
    ffprobe_available: bool
    quality_profile_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "ready": self.ready,
            "selected_profile_id": self.selected_profile_id,
            "selected_model_id": self.selected_model_id,
            "selected_revision": self.selected_revision,
            "selected_device": self.selected_device,
            "external_foundation_origin": self.external_foundation_origin,
            "blocking_dependencies": list(self.blocking_dependencies),
            "cuda_devices": list(self.cuda_devices),
            "identity_assets": list(self.identity_assets),
            "ffmpeg_available": self.ffmpeg_available,
            "ffprobe_available": self.ffprobe_available,
            "quality_profile_required": self.quality_profile_required,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionDependencyGateError(f"cannot hash identity asset: {path}") from exc
    return digest.hexdigest()


def parse_nvidia_smi_capacity(output: str) -> tuple[CUDADeviceCapacity, ...]:
    """Parse deterministic CSV emitted by the production NVIDIA capacity probe."""

    devices: list[CUDADeviceCapacity] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ProductionDependencyGateError(
                f"malformed nvidia-smi capacity row {line_number}: expected 4 columns"
            )
        try:
            index = int(parts[0])
            total_mib = float(parts[2])
            free_mib = float(parts[3])
        except ValueError as exc:
            raise ProductionDependencyGateError(
                f"malformed nvidia-smi numeric value on row {line_number}"
            ) from exc
        try:
            devices.append(
                CUDADeviceCapacity(
                    index=index,
                    name=parts[1],
                    total_vram_gb=total_mib / 1024.0,
                    free_vram_gb=free_mib / 1024.0,
                )
            )
        except ValueError as exc:
            raise ProductionDependencyGateError(
                f"invalid CUDA capacity on row {line_number}: {exc}"
            ) from exc
    return tuple(devices)


def collect_cuda_capacity(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[CUDADeviceCapacity, ...]:
    """Collect live CUDA capacity without importing a heavyweight model runtime."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = run(command, check=False, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    return parse_nvidia_smi_capacity(completed.stdout)


def _profile_for_capacity(
    devices: Sequence[CUDADeviceCapacity], *, quality_profile_required: bool
) -> tuple[FoundationExecutionProfile | None, CUDADeviceCapacity | None]:
    ordered = sorted(devices, key=lambda item: item.free_vram_gb, reverse=True)
    candidates = (WAN22_I2V_A14B_PROFILE,)
    if not quality_profile_required:
        candidates += (WAN22_TI2V_5B_PROFILE,)
    for profile in candidates:
        for device in ordered:
            if device.free_vram_gb >= profile.minimum_gpu_vram_gb:
                return profile, device
    return None, None


def evaluate_production_dependencies(
    *,
    cuda_devices: Sequence[CUDADeviceCapacity],
    identity_assets: Sequence[IdentityAssetRequirement],
    ffmpeg_available: bool,
    ffprobe_available: bool,
    quality_profile_required: bool = True,
) -> ProductionDependencyReadiness:
    """Return auditable readiness while preserving the quality-first milestone policy."""

    blockers: list[str] = []
    profile, device = _profile_for_capacity(
        cuda_devices, quality_profile_required=quality_profile_required
    )
    if profile is None:
        minimum = WAN22_I2V_A14B_PROFILE.minimum_gpu_vram_gb
        if quality_profile_required:
            blockers.append(f"cuda_free_vram_below_quality_floor:{minimum:g}GB")
        else:
            fallback = WAN22_TI2V_5B_PROFILE.minimum_gpu_vram_gb
            blockers.append(f"cuda_free_vram_below_fallback_floor:{fallback:g}GB")

    asset_evidence: list[dict[str, Any]] = []
    if not identity_assets:
        blockers.append("approved_identity_assets_missing")
    for requirement in identity_assets:
        path = requirement.path.expanduser().resolve(strict=False)
        exists = path.is_file()
        actual_sha256 = _sha256_file(path) if exists else None
        digest_matches = actual_sha256 == requirement.sha256
        if not exists:
            blockers.append(f"identity_asset_missing:{path}")
        elif not digest_matches:
            blockers.append(f"identity_asset_hash_mismatch:{path}")
        asset_evidence.append(
            {
                "path": str(path),
                "expected_sha256": requirement.sha256,
                "actual_sha256": actual_sha256,
                "exists": exists,
                "hash_matches": digest_matches,
            }
        )

    if not ffmpeg_available:
        blockers.append("ffmpeg_missing")
    if not ffprobe_available:
        blockers.append("ffprobe_missing")

    provenance = profile.provenance if profile is not None else None
    return ProductionDependencyReadiness(
        ready=not blockers,
        selected_profile_id=profile.profile_id if profile is not None else None,
        selected_model_id=provenance.model_id if provenance is not None else None,
        selected_revision=provenance.revision if provenance is not None else None,
        selected_device=f"cuda:{device.index}" if device is not None else None,
        external_foundation_origin=EXTERNAL_PRETRAINED_FOUNDATION,
        blocking_dependencies=tuple(blockers),
        cuda_devices=tuple(item.to_dict() for item in cuda_devices),
        identity_assets=tuple(asset_evidence),
        ffmpeg_available=bool(ffmpeg_available),
        ffprobe_available=bool(ffprobe_available),
        quality_profile_required=quality_profile_required,
    )


def _parse_identity_requirement(value: str) -> IdentityAssetRequirement:
    path_text, separator, digest = value.rpartition("=")
    if not separator or not path_text.strip():
        raise argparse.ArgumentTypeError("identity must use PATH=SHA256")
    try:
        return IdentityAssetRequirement(Path(path_text), digest)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed readiness gate for real CINEOS production GPU execution."
    )
    parser.add_argument(
        "--identity",
        action="append",
        default=[],
        type=_parse_identity_requirement,
        metavar="PATH=SHA256",
        help="Approved production identity/reference asset and its pinned SHA-256.",
    )
    parser.add_argument(
        "--allow-5b-fallback",
        action="store_true",
        help="Allow the pinned 5B foundation when the 80GB-class quality profile cannot run.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args(argv)

    report = evaluate_production_dependencies(
        cuda_devices=collect_cuda_capacity(),
        identity_assets=args.identity,
        ffmpeg_available=shutil.which("ffmpeg") is not None,
        ffprobe_available=shutil.which("ffprobe") is not None,
        quality_profile_required=not args.allow_5b_fallback,
    )
    payload = report.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.ready else EXIT_DEPENDENCY_BLOCKED


if __name__ == "__main__":  # pragma: no cover - exercised through main() tests.
    raise SystemExit(main())


__all__ = [
    "CUDADeviceCapacity",
    "EXIT_DEPENDENCY_BLOCKED",
    "IdentityAssetRequirement",
    "ProductionDependencyGateError",
    "ProductionDependencyReadiness",
    "SCHEMA",
    "collect_cuda_capacity",
    "evaluate_production_dependencies",
    "main",
    "parse_nvidia_smi_capacity",
]
