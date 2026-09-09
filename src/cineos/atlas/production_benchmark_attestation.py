"""Content-addressed attestation for quality-first production GPU benchmarks.

The quality-first selector writes its decision before rendering so operators can audit
why a particular external pretrained foundation was chosen. A successful connected
benchmark must also bind that exact decision to the exact accepted benchmark receipt;
otherwise the two JSON files can be substituted independently after a run.

This module provides that binding without pretending a content digest is a digital
signature. It is CINEOS evidence/orchestration metadata only. The underlying video
foundation remains identified by the selection payload as external pretrained weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

LEGACY_SCHEMA = "cineos-quality-first-production-attestation/0.1"
SCHEMA = "cineos-quality-first-production-attestation/0.2"
DEFAULT_FILENAME = "quality-first-production-attestation.json"


class ProductionBenchmarkAttestationError(RuntimeError):
    """Raised when quality-first benchmark evidence is incomplete or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionBenchmarkAttestationError(
            f"cannot hash evidence file: {path}"
        ) from exc
    return digest.hexdigest()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionBenchmarkAttestationError(
            f"cannot read {label} JSON: {path}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ProductionBenchmarkAttestationError(f"{label} must be a JSON object")
    return dict(raw)


def _jsonable(value: Any) -> Any:
    """Serialize exact evidence objects used by backward-compatible integration tests.

    Production receipts expose ``to_dict()`` and take that path. The recursive path is
    intentionally lossless for lightweight receipt objects used by legacy integration
    tests and custom callers: it captures their complete public attribute state rather
    than replacing missing evidence with invented values.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    serializer = getattr(value, "to_dict", None)
    if callable(serializer):
        return _jsonable(serializer())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    try:
        attributes = vars(value)
    except TypeError as exc:
        raise ProductionBenchmarkAttestationError(
            f"cannot serialize benchmark evidence object: {type(value).__name__}"
        ) from exc
    return {
        key: _jsonable(item)
        for key, item in attributes.items()
        if not key.startswith("_") and not callable(item)
    }


def _receipt_payload(receipt: Any, *, benchmark_id: str | None) -> dict[str, Any]:
    serializer = getattr(receipt, "to_dict", None)
    if callable(serializer):
        payload = serializer()
        if not isinstance(payload, Mapping):
            raise ProductionBenchmarkAttestationError(
                "connected benchmark receipt payload must be a mapping"
            )
        normalized = dict(payload)
        receipt_benchmark_id = normalized.get("benchmark_id")
        if benchmark_id is not None and receipt_benchmark_id != benchmark_id:
            raise ProductionBenchmarkAttestationError(
                "connected benchmark receipt benchmark_id does not match invocation"
            )
        return normalized

    payload = _jsonable(receipt)
    if not isinstance(payload, Mapping):
        raise ProductionBenchmarkAttestationError(
            "connected benchmark receipt payload must be a mapping"
        )
    normalized = dict(payload)
    resolved_benchmark_id = benchmark_id or getattr(receipt, "benchmark_id", None)
    if not isinstance(resolved_benchmark_id, str) or not resolved_benchmark_id.strip():
        resolved_benchmark_id = "legacy-injected-receipt"
    normalized["benchmark_id"] = resolved_benchmark_id
    return normalized


def _required_string(mapping: Mapping[str, Any], field: str, *, label: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProductionBenchmarkAttestationError(f"{label} is missing {field}")
    return value


def _shot_foundation(shot: Mapping[str, Any]) -> Mapping[str, Any] | None:
    foundation = shot.get("foundation")
    if isinstance(foundation, Mapping):
        return foundation
    result = shot.get("result")
    if isinstance(result, Mapping):
        nested = result.get("foundation")
        if isinstance(nested, Mapping):
            return nested
    return None


def _validate_detailed_shot_binding(
    selection: Mapping[str, Any], connected: Mapping[str, Any]
) -> None:
    """Bind each accepted shot to the exact selected foundation and execution policy.

    Selection happens from a live preflight before the renderer opens. The renderer
    intentionally performs a second live preflight at session open, so volatile
    observed-free-VRAM fields may legitimately differ. Stable policy fields must not.
    """

    selected_model = _required_string(
        selection, "model_id", label="foundation selection"
    )
    selected_revision = _required_string(
        selection, "revision", label="foundation selection"
    )
    if len(selected_revision) != 40 or any(
        character not in "0123456789abcdef" for character in selected_revision.lower()
    ):
        raise ProductionBenchmarkAttestationError(
            "foundation selection revision must be an immutable 40-character hash"
        )

    selected_plan = selection.get("execution_plan")
    if not isinstance(selected_plan, Mapping):
        raise ProductionBenchmarkAttestationError(
            "foundation selection is missing execution_plan"
        )
    required_plan_fields = (
        "device",
        "dtype",
        "memory_strategy",
        "enable_vae_tiling",
        "enable_vae_slicing",
        "enable_attention_slicing",
        "estimated_model_vram_gb",
    )
    for field in required_plan_fields:
        if field not in selected_plan:
            raise ProductionBenchmarkAttestationError(
                f"foundation selection execution_plan is missing {field}"
            )

    shots = connected.get("shots")
    if shots is None:
        shots = connected.get("shot_receipts")
    if not isinstance(shots, Sequence) or isinstance(shots, (str, bytes, bytearray)):
        raise ProductionBenchmarkAttestationError(
            "connected benchmark is missing per-shot evidence"
        )
    shot_count = connected.get("shot_count", len(shots))
    if (
        not isinstance(shot_count, int)
        or isinstance(shot_count, bool)
        or shot_count <= 0
    ):
        raise ProductionBenchmarkAttestationError(
            "connected benchmark has invalid shot_count"
        )
    if len(shots) != shot_count:
        raise ProductionBenchmarkAttestationError(
            "connected benchmark per-shot evidence count does not match shot_count"
        )

    selected_profile = selection["profile_id"]
    selected_origin = selection["origin"]
    for index, shot in enumerate(shots):
        if not isinstance(shot, Mapping):
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} evidence is malformed"
            )
        if shot.get("profile_id") != selected_profile:
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} profile_id does not match foundation selection"
            )
        if shot.get("origin") != selected_origin:
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} origin does not match foundation selection"
            )

        foundation = _shot_foundation(shot)
        if foundation is None:
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} is missing foundation provenance"
            )
        if foundation.get("model_id") != selected_model:
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} model_id does not match foundation selection"
            )
        if foundation.get("revision") != selected_revision:
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} revision does not match foundation selection"
            )

        execution_plan = shot.get("execution_plan")
        if not isinstance(execution_plan, Mapping):
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} is missing execution_plan"
            )
        for field in required_plan_fields:
            if execution_plan.get(field) != selected_plan.get(field):
                raise ProductionBenchmarkAttestationError(
                    f"connected benchmark shot {index} execution field {field!r} "
                    "does not match foundation selection"
                )

        runtime = shot.get("runtime_provenance")
        if not isinstance(runtime, Mapping):
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} is missing runtime provenance"
            )
        if runtime.get("cuda_device") != selected_plan.get("device"):
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} runtime CUDA device does not match foundation selection"
            )
        if runtime.get("dtype") != selected_plan.get("dtype"):
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark shot {index} runtime dtype does not match foundation selection"
            )


def _validate_cross_binding(
    selection: Mapping[str, Any],
    connected: Mapping[str, Any],
    *,
    require_detailed_shot_binding: bool,
) -> None:
    for field in ("profile_id", "origin"):
        selected = selection.get(field)
        rendered = connected.get(field)
        if not isinstance(selected, str) or not selected.strip():
            raise ProductionBenchmarkAttestationError(
                f"foundation selection is missing {field}"
            )
        if rendered != selected:
            raise ProductionBenchmarkAttestationError(
                f"connected benchmark {field} does not match foundation selection"
            )

    benchmark_id = connected.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ProductionBenchmarkAttestationError(
            "connected benchmark is missing benchmark_id"
        )
    if connected.get("production_gpu_evidence") is not True:
        raise ProductionBenchmarkAttestationError(
            "connected benchmark lacks production GPU evidence"
        )
    if connected.get("production_quality_evidence") is not True:
        raise ProductionBenchmarkAttestationError(
            "connected benchmark lacks production quality evidence"
        )
    if connected.get("evidence_tier") != "production-gpu-quality-gated":
        raise ProductionBenchmarkAttestationError(
            "connected benchmark is not production-gpu-quality-gated"
        )
    if require_detailed_shot_binding:
        _validate_detailed_shot_binding(selection, connected)


def remove_stale_quality_first_attestation(
    output_dir: str | Path, *, filename: str = DEFAULT_FILENAME
) -> None:
    """Remove prior success evidence before a new production attempt starts."""

    path = Path(output_dir) / filename
    if not path.exists():
        return
    if not path.is_file():
        raise ProductionBenchmarkAttestationError(
            f"quality-first attestation path is not a file: {path}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise ProductionBenchmarkAttestationError(
            f"cannot remove stale quality-first attestation: {path}"
        ) from exc


def write_quality_first_production_attestation(
    output_dir: str | Path,
    *,
    selection_manifest: str | Path,
    receipt: Any,
    benchmark_id: str | None = None,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Atomically bind the exact selector sidecar to exact accepted benchmark evidence."""

    if benchmark_id is not None and (
        not isinstance(benchmark_id, str) or not benchmark_id.strip()
    ):
        raise ProductionBenchmarkAttestationError("benchmark_id must not be empty")
    output_root = Path(output_dir)
    selection_path = Path(selection_manifest)
    if selection_path.parent.resolve(strict=False) != output_root.resolve(strict=False):
        raise ProductionBenchmarkAttestationError(
            "foundation selection manifest must live in the benchmark output directory"
        )
    if selection_path.name != "foundation-selection.json":
        raise ProductionBenchmarkAttestationError(
            "unexpected foundation selection manifest filename"
        )

    selection = _load_mapping(selection_path, label="foundation selection")
    connected = _receipt_payload(receipt, benchmark_id=benchmark_id)
    _validate_cross_binding(
        selection,
        connected,
        require_detailed_shot_binding=True,
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "benchmark_id": connected["benchmark_id"],
        "selection_manifest": selection_path.name,
        "selection_manifest_sha256": _sha256_file(selection_path),
        "selection_payload_sha256": _sha256_bytes(_canonical_bytes(selection)),
        "connected_evidence_sha256": _sha256_bytes(_canonical_bytes(connected)),
        "foundation_selection": selection,
        "connected_benchmark": connected,
    }
    attestation = dict(body)
    attestation["attestation_sha256"] = _sha256_bytes(_canonical_bytes(body))

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / filename
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=output_root
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(attestation, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except OSError as exc:
        try:
            if "temporary_name" in locals():
                Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise ProductionBenchmarkAttestationError(
            f"cannot persist quality-first production attestation: {destination}"
        ) from exc

    verify_quality_first_production_attestation(destination)
    return destination


def verify_quality_first_production_attestation(path: str | Path) -> dict[str, Any]:
    """Fail closed if either bound sidecar or embedded benchmark evidence changed."""

    attestation_path = Path(path)
    payload = _load_mapping(attestation_path, label="quality-first attestation")
    schema = payload.get("schema")
    if schema not in {LEGACY_SCHEMA, SCHEMA}:
        raise ProductionBenchmarkAttestationError(
            "unsupported quality-first production attestation schema"
        )

    supplied_root = payload.get("attestation_sha256")
    if not isinstance(supplied_root, str) or len(supplied_root) != 64:
        raise ProductionBenchmarkAttestationError(
            "quality-first attestation is missing its root digest"
        )
    body = dict(payload)
    del body["attestation_sha256"]
    expected_root = _sha256_bytes(_canonical_bytes(body))
    if supplied_root != expected_root:
        raise ProductionBenchmarkAttestationError(
            "quality-first attestation root digest does not match payload"
        )

    selection_name = payload.get("selection_manifest")
    if selection_name != "foundation-selection.json":
        raise ProductionBenchmarkAttestationError(
            "quality-first attestation has an unexpected selection manifest"
        )
    selection_path = attestation_path.parent / selection_name
    if _sha256_file(selection_path) != payload.get("selection_manifest_sha256"):
        raise ProductionBenchmarkAttestationError(
            "foundation selection sidecar digest does not match attestation"
        )

    selection = _load_mapping(selection_path, label="foundation selection")
    embedded_selection = payload.get("foundation_selection")
    if selection != embedded_selection:
        raise ProductionBenchmarkAttestationError(
            "foundation selection sidecar does not match embedded selection evidence"
        )
    if _sha256_bytes(_canonical_bytes(selection)) != payload.get(
        "selection_payload_sha256"
    ):
        raise ProductionBenchmarkAttestationError(
            "foundation selection payload digest does not match attestation"
        )

    connected = payload.get("connected_benchmark")
    if not isinstance(connected, Mapping):
        raise ProductionBenchmarkAttestationError(
            "quality-first attestation is missing connected benchmark evidence"
        )
    connected_dict = dict(connected)
    if _sha256_bytes(_canonical_bytes(connected_dict)) != payload.get(
        "connected_evidence_sha256"
    ):
        raise ProductionBenchmarkAttestationError(
            "connected benchmark digest does not match attestation"
        )
    _validate_cross_binding(
        selection,
        connected_dict,
        require_detailed_shot_binding=(schema == SCHEMA),
    )
    if connected_dict.get("benchmark_id") != payload.get("benchmark_id"):
        raise ProductionBenchmarkAttestationError(
            "attestation benchmark_id does not match connected benchmark"
        )
    return payload


__all__ = [
    "DEFAULT_FILENAME",
    "LEGACY_SCHEMA",
    "ProductionBenchmarkAttestationError",
    "SCHEMA",
    "remove_stale_quality_first_attestation",
    "verify_quality_first_production_attestation",
    "write_quality_first_production_attestation",
]
