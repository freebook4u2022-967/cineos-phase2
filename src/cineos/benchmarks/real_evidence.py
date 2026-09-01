"""Fail-closed validation for production real-inference benchmark evidence.

This module does not execute a model. It validates the evidence emitted by the
production GPU path so a competitive benchmark cannot be declared passing from
synthetic metrics, missing media, malformed receipts, ambiguous foundation
provenance, or stale/swapped artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .case import BenchmarkCase
from .exceptions import BenchmarkError
from .metrics import MetricStatus
from .report import CaseResult

ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
ARTIFACT_MANIFEST_SCHEMA = "cineos-real-inference-artifacts/0.1"


def validate_real_inference_evidence(
    case: BenchmarkCase,
    result: CaseResult,
    output_dir: str | Path,
    *,
    foundation: Mapping[str, object],
    require_artifact_manifest: bool = False,
) -> None:
    """Validate one production GPU case before it can count toward release.

    Competitive cases require real inference, non-empty declared artifacts,
    structurally valid benchmark evidence, threshold metrics produced by measurement
    (not estimates/manual review), and explicit external-pretrained-foundation
    provenance. Production release callers can additionally require a content-
    addressed artifact manifest so stale, swapped, or modified evidence cannot be
    admitted after measurement. The function raises ``BenchmarkError`` on the first
    invalid condition and otherwise returns None.
    """

    if case.hardware_requirements.get("real_inference") is not True:
        raise BenchmarkError("case is not declared as a real-inference benchmark")
    if result.case_id != case.case_id:
        raise BenchmarkError("benchmark result case_id does not match case contract")
    if not result.passed:
        raise BenchmarkError("real-inference benchmark result did not pass")

    root = Path(output_dir).resolve()
    declared_outputs = set(result.outputs)
    for expected in case.expected_outputs:
        if expected not in declared_outputs:
            raise BenchmarkError(f"missing declared benchmark output: {expected}")
        artifact = _resolve_artifact(root, expected)
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise BenchmarkError(f"missing or empty benchmark artifact: {expected}")
        _validate_artifact_structure(expected, artifact)

    if require_artifact_manifest:
        _validate_artifact_manifest(case, root)

    metric_by_name = {metric.name: metric for metric in result.metrics}
    for name, threshold in case.validation_thresholds.items():
        metric = metric_by_name.get(name)
        if metric is None:
            raise BenchmarkError(f"missing required measured metric: {name}")
        if metric.status is not MetricStatus.MEASURED:
            raise BenchmarkError(f"required metric is not measured: {name}")
        if isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)):
            raise BenchmarkError(f"required metric is not numeric: {name}")
        if float(metric.value) < threshold:
            raise BenchmarkError(
                f"metric {name}={float(metric.value):.4f} is below threshold {threshold:.4f}"
            )

    origin = foundation.get("origin")
    model_id = foundation.get("model_id")
    if origin != "external_pretrained_foundation":
        raise BenchmarkError(
            "competitive real inference must declare external_pretrained_foundation origin"
        )
    if not isinstance(model_id, str) or not model_id.strip():
        raise BenchmarkError(
            "competitive real inference must declare foundation model_id"
        )


def write_real_inference_artifact_manifest(
    case: BenchmarkCase,
    output_dir: str | Path,
) -> Path:
    """Write a deterministic content manifest for finalized case artifacts.

    This helper is intended for the real GPU path *after* every expected output has
    been finalized. The manifest never hashes itself, avoiding circular evidence.
    Existing research callers need not opt into this production-release contract.
    """

    root = Path(output_dir).resolve()
    artifacts: dict[str, dict[str, object]] = {}
    for expected in case.expected_outputs:
        if expected == ARTIFACT_MANIFEST_NAME:
            raise BenchmarkError(
                "artifact manifest must not appear in case.expected_outputs"
            )
        artifact = _resolve_artifact(root, expected)
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise BenchmarkError(
                f"cannot manifest missing or empty benchmark artifact: {expected}"
            )
        artifacts[expected] = {
            "sha256": _sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }

    payload = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "case_id": case.case_id,
        "artifacts": artifacts,
    }
    manifest_path = _resolve_artifact(root, ARTIFACT_MANIFEST_NAME)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(manifest_path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise BenchmarkError("unable to write benchmark artifact manifest") from exc
    return manifest_path


def _validate_artifact_manifest(case: BenchmarkCase, root: Path) -> None:
    manifest_path = _resolve_artifact(root, ARTIFACT_MANIFEST_NAME)
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        raise BenchmarkError("missing production benchmark artifact manifest")
    payload = _read_json_object(manifest_path)
    if payload.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
        raise BenchmarkError("benchmark artifact manifest schema is invalid")
    if payload.get("case_id") != case.case_id:
        raise BenchmarkError("benchmark artifact manifest case_id does not match case")

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise BenchmarkError("benchmark artifact manifest artifacts must be an object")
    expected_names = set(case.expected_outputs)
    manifest_names = set(raw_artifacts)
    if manifest_names != expected_names:
        missing = sorted(expected_names - manifest_names)
        extra = sorted(manifest_names - expected_names)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise BenchmarkError(
            "benchmark artifact manifest does not exactly match case outputs"
            + (": " + "; ".join(details) if details else "")
        )

    for name in case.expected_outputs:
        entry = raw_artifacts.get(name)
        if not isinstance(entry, dict):
            raise BenchmarkError(f"artifact manifest entry is invalid: {name}")
        declared_hash = entry.get("sha256")
        declared_size = entry.get("size_bytes")
        if (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or any(char not in "0123456789abcdef" for char in declared_hash)
        ):
            raise BenchmarkError(f"artifact manifest sha256 is invalid: {name}")
        if isinstance(declared_size, bool) or not isinstance(declared_size, int):
            raise BenchmarkError(f"artifact manifest size_bytes is invalid: {name}")
        artifact = _resolve_artifact(root, name)
        if not artifact.is_file():
            raise BenchmarkError(f"manifested benchmark artifact is missing: {name}")
        actual_size = artifact.stat().st_size
        if actual_size != declared_size:
            raise BenchmarkError(
                f"benchmark artifact size does not match manifest: {name}"
            )
        if _sha256_file(artifact) != declared_hash:
            raise BenchmarkError(
                f"benchmark artifact sha256 does not match manifest: {name}"
            )


def _resolve_artifact(root: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute():
        raise BenchmarkError("benchmark output path must be relative")
    artifact = (root / candidate).resolve()
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise BenchmarkError(
            "benchmark output escapes the case output directory"
        ) from exc
    return artifact


def _sha256_file(artifact: Path) -> str:
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkError(
            f"unable to hash benchmark artifact: {artifact.name}"
        ) from exc
    return digest.hexdigest()


def _validate_artifact_structure(name: str, artifact: Path) -> None:
    """Reject placeholder files before they can satisfy the production gate.

    JSON evidence must parse to an object. MP4 evidence must expose an ISO-BMFF
    ``ftyp`` box near the start of the file. This is intentionally lightweight and
    dependency-free: decode-level validation remains the responsibility of the
    production media probe, while this guard prevents arbitrary non-empty bytes from
    being accepted as real benchmark evidence.
    """

    suffix = artifact.suffix.lower()
    if suffix == ".json":
        payload = _read_json_object(artifact)
        if name == "report.json" and not payload:
            raise BenchmarkError("benchmark report JSON object is empty")
        return
    if suffix == ".mp4":
        _validate_mp4_container(artifact)


def _read_json_object(artifact: Path) -> dict[str, Any]:
    try:
        with artifact.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"benchmark JSON artifact is malformed: {artifact.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(
            f"benchmark JSON artifact must contain an object: {artifact.name}"
        )
    return payload


def _validate_mp4_container(artifact: Path) -> None:
    try:
        with artifact.open("rb") as handle:
            header = handle.read(64)
    except OSError as exc:
        raise BenchmarkError(
            f"unable to read benchmark video artifact: {artifact.name}"
        ) from exc

    # ISO Base Media File Format starts with a small box whose type is commonly
    # ``ftyp`` at byte offset 4. Search the first 64 bytes to tolerate legal leading
    # boxes while still rejecting arbitrary placeholder text or raw frame dumps.
    if b"ftyp" not in header:
        raise BenchmarkError(
            f"benchmark video artifact is not an MP4/ISO-BMFF container: {artifact.name}"
        )


__all__ = [
    "ARTIFACT_MANIFEST_NAME",
    "ARTIFACT_MANIFEST_SCHEMA",
    "validate_real_inference_evidence",
    "write_real_inference_artifact_manifest",
]
