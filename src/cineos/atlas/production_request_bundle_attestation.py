"""Bind the exact preflight-approved request bundle to production render evidence.

The connected benchmark fixture preflight proves that a committed 5-10 shot request
bundle covers the canonical competitive cases.  The quality-first production
attestation proves what foundation/runtime/QC produced the accepted shots.  This
module closes the remaining gap between those evidence chains: it revalidates the
request bundle immediately before inference and, after rendering, proves that every
accepted shot receipt refers to that exact ordered bundle.

This is CINEOS orchestration/evidence metadata.  It does not relabel external
pretrained foundation capability as native CINEOS capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gpu_benchmark_cli import GPUProductionBenchmarkCLIError, load_native_requests
from .production_benchmark_attestation import (
    ProductionBenchmarkAttestationError,
    verify_quality_first_production_attestation,
)

SCHEMA = "cineos-production-request-bundle-attestation/0.1"
PREFLIGHT_SCHEMA = "cineos-connected-benchmark-fixture-preflight/0.2"
PREFLIGHT_FILENAME = "connected-benchmark-fixture-preflight.json"
QUALITY_ATTESTATION_FILENAME = "quality-first-production-attestation.json"
DEFAULT_FILENAME = "production-request-bundle-attestation.json"


class ProductionRequestBundleAttestationError(RuntimeError):
    """Raised when production evidence is not bound to the approved request bundle."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionRequestBundleAttestationError(
            f"cannot hash production evidence file: {path}"
        ) from exc
    return digest.hexdigest()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionRequestBundleAttestationError(
            f"cannot read {label} JSON: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProductionRequestBundleAttestationError(f"{label} must be a JSON object")
    return dict(payload)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _bundle_binding(entries: Sequence[tuple[str, str]]) -> tuple[list[str], str]:
    normalized: list[dict[str, str]] = []
    ordered_shot_ids: list[str] = []
    for index, (shot_id, request_hash) in enumerate(entries):
        if not isinstance(shot_id, str) or not shot_id:
            raise ProductionRequestBundleAttestationError(
                f"request bundle entry {index} is missing shot_id"
            )
        if not _is_sha256(request_hash):
            raise ProductionRequestBundleAttestationError(
                f"request bundle entry {index} has invalid request hash"
            )
        ordered_shot_ids.append(shot_id)
        normalized.append({"shot_id": shot_id, "content_hash": request_hash})
    if len(set(ordered_shot_ids)) != len(ordered_shot_ids):
        raise ProductionRequestBundleAttestationError(
            "request bundle contains duplicate shot IDs"
        )
    return ordered_shot_ids, _sha256_bytes(_canonical_bytes(normalized))


def _validated_preflight(payload: Mapping[str, Any]) -> tuple[list[str], str]:
    if payload.get("schema") != PREFLIGHT_SCHEMA:
        raise ProductionRequestBundleAttestationError(
            "connected benchmark preflight schema is not request-bundle-bound"
        )
    if payload.get("validated") is not True:
        raise ProductionRequestBundleAttestationError(
            "connected benchmark preflight is not validated"
        )
    ordered = payload.get("ordered_shot_ids")
    if not isinstance(ordered, Sequence) or isinstance(ordered, (str, bytes)):
        raise ProductionRequestBundleAttestationError(
            "connected benchmark preflight is missing ordered shot IDs"
        )
    ordered_ids = list(ordered)
    if (
        any(not isinstance(item, str) or not item for item in ordered_ids)
        or len(set(ordered_ids)) != len(ordered_ids)
    ):
        raise ProductionRequestBundleAttestationError(
            "connected benchmark preflight has invalid ordered shot IDs"
        )
    shot_count = payload.get("shot_count")
    if shot_count != len(ordered_ids) or not 5 <= len(ordered_ids) <= 10:
        raise ProductionRequestBundleAttestationError(
            "connected benchmark preflight shot count does not match its ordered bundle"
        )
    bundle_hash = payload.get("normalized_request_bundle_sha256")
    if not _is_sha256(bundle_hash):
        raise ProductionRequestBundleAttestationError(
            "connected benchmark preflight is missing a canonical bundle digest"
        )
    return ordered_ids, bundle_hash


def validate_request_bundle_preflight(
    requests_path: str | Path,
    preflight_path: str | Path,
) -> dict[str, Any]:
    """Recompute the preflight bundle binding from exact requests before inference."""

    preflight = _load_mapping(Path(preflight_path), label="connected benchmark preflight")
    expected_ids, expected_hash = _validated_preflight(preflight)
    try:
        requests = load_native_requests(requests_path)
    except GPUProductionBenchmarkCLIError as exc:
        raise ProductionRequestBundleAttestationError(str(exc)) from exc
    actual_ids, actual_hash = _bundle_binding(
        [(request.shot_id, request.content_hash) for request in requests]
    )
    if actual_ids != expected_ids:
        raise ProductionRequestBundleAttestationError(
            "executable request order changed after connected benchmark preflight"
        )
    if actual_hash != expected_hash:
        raise ProductionRequestBundleAttestationError(
            "executable request content changed after connected benchmark preflight"
        )
    return {
        "schema": "cineos-production-request-bundle-gate/0.1",
        "shot_count": len(actual_ids),
        "ordered_shot_ids": actual_ids,
        "normalized_request_bundle_sha256": actual_hash,
        "validated": True,
    }


def _render_bundle_binding(quality_attestation: Mapping[str, Any]) -> tuple[list[str], str]:
    connected = quality_attestation.get("connected_benchmark")
    if not isinstance(connected, Mapping):
        raise ProductionRequestBundleAttestationError(
            "quality attestation is missing connected benchmark evidence"
        )
    shots = connected.get("shots")
    if shots is None:
        shots = connected.get("shot_receipts")
    if not isinstance(shots, Sequence) or isinstance(shots, (str, bytes)):
        raise ProductionRequestBundleAttestationError(
            "quality attestation is missing per-shot render evidence"
        )
    entries: list[tuple[str, str]] = []
    for index, shot in enumerate(shots):
        if not isinstance(shot, Mapping):
            raise ProductionRequestBundleAttestationError(
                f"render evidence shot {index} is malformed"
            )
        result = shot.get("result")
        if not isinstance(result, Mapping):
            raise ProductionRequestBundleAttestationError(
                f"render evidence shot {index} is missing result evidence"
            )
        entries.append((result.get("shot_id"), result.get("request_hash")))
    return _bundle_binding(entries)


def write_production_request_bundle_attestation(
    output_dir: str | Path,
    *,
    requests_path: str | Path,
    preflight_path: str | Path,
    quality_attestation_path: str | Path,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    """Create a root digest binding preflight inputs to accepted production renders."""

    output_root = Path(output_dir).resolve(strict=False)
    preflight_file = Path(preflight_path)
    quality_file = Path(quality_attestation_path)
    for path, expected_name, label in (
        (preflight_file, PREFLIGHT_FILENAME, "connected benchmark preflight"),
        (quality_file, QUALITY_ATTESTATION_FILENAME, "quality attestation"),
    ):
        if path.resolve(strict=False).parent != output_root:
            raise ProductionRequestBundleAttestationError(
                f"{label} must live in the benchmark output directory"
            )
        if path.name != expected_name:
            raise ProductionRequestBundleAttestationError(
                f"unexpected {label} filename"
            )

    gate = validate_request_bundle_preflight(requests_path, preflight_file)
    preflight = _load_mapping(preflight_file, label="connected benchmark preflight")
    expected_ids, expected_hash = _validated_preflight(preflight)
    try:
        quality = verify_quality_first_production_attestation(quality_file)
    except ProductionBenchmarkAttestationError as exc:
        raise ProductionRequestBundleAttestationError(str(exc)) from exc
    rendered_ids, rendered_hash = _render_bundle_binding(quality)
    if rendered_ids != expected_ids:
        raise ProductionRequestBundleAttestationError(
            "accepted render order does not match the preflight-approved request bundle"
        )
    if rendered_hash != expected_hash:
        raise ProductionRequestBundleAttestationError(
            "accepted render request hashes do not match the preflight-approved bundle"
        )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "shot_count": gate["shot_count"],
        "ordered_shot_ids": expected_ids,
        "normalized_request_bundle_sha256": expected_hash,
        "preflight_manifest": preflight_file.name,
        "preflight_manifest_sha256": _sha256_file(preflight_file),
        "quality_attestation": quality_file.name,
        "quality_attestation_sha256": _sha256_file(quality_file),
        "quality_attestation_root_sha256": quality.get("attestation_sha256"),
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
        raise ProductionRequestBundleAttestationError(
            f"cannot persist production request-bundle attestation: {destination}"
        ) from exc
    verify_production_request_bundle_attestation(destination)
    return destination


def verify_production_request_bundle_attestation(path: str | Path) -> dict[str, Any]:
    """Verify the root and both bound sidecars without needing the original requests."""

    attestation_path = Path(path)
    payload = _load_mapping(attestation_path, label="production request-bundle attestation")
    if payload.get("schema") != SCHEMA:
        raise ProductionRequestBundleAttestationError(
            "unsupported production request-bundle attestation schema"
        )
    supplied_root = payload.get("attestation_sha256")
    if not _is_sha256(supplied_root):
        raise ProductionRequestBundleAttestationError(
            "production request-bundle attestation is missing its root digest"
        )
    body = dict(payload)
    del body["attestation_sha256"]
    if _sha256_bytes(_canonical_bytes(body)) != supplied_root:
        raise ProductionRequestBundleAttestationError(
            "production request-bundle attestation root digest does not match payload"
        )

    preflight_name = payload.get("preflight_manifest")
    quality_name = payload.get("quality_attestation")
    if preflight_name != PREFLIGHT_FILENAME or quality_name != QUALITY_ATTESTATION_FILENAME:
        raise ProductionRequestBundleAttestationError(
            "production request-bundle attestation has unexpected sidecar names"
        )
    preflight_path = attestation_path.parent / preflight_name
    quality_path = attestation_path.parent / quality_name
    if _sha256_file(preflight_path) != payload.get("preflight_manifest_sha256"):
        raise ProductionRequestBundleAttestationError(
            "connected benchmark preflight digest does not match attestation"
        )
    if _sha256_file(quality_path) != payload.get("quality_attestation_sha256"):
        raise ProductionRequestBundleAttestationError(
            "quality attestation file digest does not match request-bundle attestation"
        )

    preflight = _load_mapping(preflight_path, label="connected benchmark preflight")
    expected_ids, expected_hash = _validated_preflight(preflight)
    try:
        quality = verify_quality_first_production_attestation(quality_path)
    except ProductionBenchmarkAttestationError as exc:
        raise ProductionRequestBundleAttestationError(str(exc)) from exc
    if quality.get("attestation_sha256") != payload.get("quality_attestation_root_sha256"):
        raise ProductionRequestBundleAttestationError(
            "quality attestation root does not match request-bundle attestation"
        )
    rendered_ids, rendered_hash = _render_bundle_binding(quality)
    if rendered_ids != expected_ids or rendered_hash != expected_hash:
        raise ProductionRequestBundleAttestationError(
            "accepted render evidence does not match the preflight-approved request bundle"
        )
    if payload.get("ordered_shot_ids") != expected_ids:
        raise ProductionRequestBundleAttestationError(
            "attested shot order does not match connected benchmark preflight"
        )
    if payload.get("normalized_request_bundle_sha256") != expected_hash:
        raise ProductionRequestBundleAttestationError(
            "attested request-bundle digest does not match connected benchmark preflight"
        )
    if payload.get("shot_count") != len(expected_ids):
        raise ProductionRequestBundleAttestationError(
            "attested shot count does not match connected benchmark preflight"
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate and attest production renders against the exact preflight request bundle."
    )
    parser.add_argument("--requests", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--quality-attestation")
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.quality_attestation:
            if not args.output_dir:
                raise ProductionRequestBundleAttestationError(
                    "--output-dir is required when writing the final bundle attestation"
                )
            path = write_production_request_bundle_attestation(
                args.output_dir,
                requests_path=args.requests,
                preflight_path=args.preflight,
                quality_attestation_path=args.quality_attestation,
            )
            payload = verify_production_request_bundle_attestation(path)
        else:
            payload = validate_request_bundle_preflight(args.requests, args.preflight)
    except ProductionRequestBundleAttestationError as exc:
        raise SystemExit(f"production request-bundle gate failed: {exc}") from exc
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FILENAME",
    "PREFLIGHT_FILENAME",
    "PREFLIGHT_SCHEMA",
    "ProductionRequestBundleAttestationError",
    "QUALITY_ATTESTATION_FILENAME",
    "SCHEMA",
    "main",
    "validate_request_bundle_preflight",
    "verify_production_request_bundle_attestation",
    "write_production_request_bundle_attestation",
]
