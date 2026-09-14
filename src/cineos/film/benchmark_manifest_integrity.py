"""Persisted-manifest integrity checks for production connected benchmarks.

The connected GPU benchmark writes a JSON manifest after all shots and quality reports
have completed.  This module independently binds a later in-memory receipt to that
persisted evidence before release tooling trusts it.  External foundation metadata is
preserved as provenance; validating it does not make the foundation CINEOS-native.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .exceptions import AssemblyError

_SUPPORTED_BENCHMARK_SCHEMA = "cineos-gpu-connected-benchmark/0.3"
_PERSISTED_RUNNER_FIELDS = frozenset({"foundation_profile"})


def _receipt_snapshot(benchmark: Any) -> Mapping[str, Any]:
    serializer = getattr(benchmark, "to_dict", None)
    if not callable(serializer):
        raise AssemblyError(
            "production benchmark receipt cannot be bound to persisted manifest evidence"
        )
    snapshot = serializer()
    if not isinstance(snapshot, Mapping):
        raise AssemblyError(
            "production benchmark receipt serializer returned invalid evidence"
        )
    return snapshot


def validate_persisted_benchmark_manifest(benchmark: Any) -> dict[str, Any]:
    """Return the persisted manifest after proving it matches ``benchmark`` exactly.

    ``run_connected_gpu_benchmark`` persists ``receipt.to_dict()`` plus the selected
    external foundation profile. Release-time validation compares every receipt-owned
    field to the persisted JSON rather than trusting a mutable in-memory object. The
    foundation-profile block is the only additional persisted field allowed because it
    is written by the benchmark runner, not by ``receipt.to_dict()``. Rejecting unknown
    top-level evidence prevents a later producer from smuggling unbound quality or
    provenance claims into a manifest that otherwise matches the trusted receipt.
    """

    manifest_path = getattr(benchmark, "manifest_path", None)
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise AssemblyError(
            "production benchmark receipt is missing persisted manifest path"
        )

    path = Path(manifest_path).expanduser()
    try:
        if not path.is_file():
            raise AssemblyError(
                "production benchmark persisted manifest does not exist"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except AssemblyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(
            "cannot read production benchmark persisted manifest"
        ) from exc

    if not isinstance(payload, dict):
        raise AssemblyError(
            "production benchmark persisted manifest must be a JSON object"
        )
    if payload.get("schema") != _SUPPORTED_BENCHMARK_SCHEMA:
        raise AssemblyError(
            "production benchmark persisted manifest has unsupported schema"
        )

    snapshot = dict(_receipt_snapshot(benchmark))
    if snapshot.get("schema") != _SUPPORTED_BENCHMARK_SCHEMA:
        raise AssemblyError("production benchmark receipt has unsupported schema")

    expected_fields = set(snapshot) | _PERSISTED_RUNNER_FIELDS
    unexpected_fields = sorted(set(payload) - expected_fields)
    if unexpected_fields:
        raise AssemblyError(
            "production benchmark persisted manifest contains unbound fields: "
            + ", ".join(repr(field) for field in unexpected_fields)
        )

    for key, expected in snapshot.items():
        if key not in payload:
            raise AssemblyError(
                f"production benchmark persisted manifest is missing receipt field {key!r}"
            )
        if payload[key] != expected:
            raise AssemblyError(
                f"production benchmark persisted manifest does not match receipt field {key!r}"
            )

    foundation_profile = payload.get("foundation_profile")
    if not isinstance(foundation_profile, Mapping):
        raise AssemblyError(
            "production benchmark persisted manifest is missing foundation profile provenance"
        )
    profile_id = str(snapshot.get("profile_id") or "").strip()
    origin = str(snapshot.get("origin") or "").strip()
    if foundation_profile.get("profile_id") != profile_id:
        raise AssemblyError(
            "production benchmark persisted foundation profile does not match receipt profile"
        )
    if foundation_profile.get("origin") != origin:
        raise AssemblyError(
            "production benchmark persisted foundation origin does not match receipt origin"
        )

    return payload


__all__ = ["validate_persisted_benchmark_manifest"]
