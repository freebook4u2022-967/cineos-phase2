"""Production runtime provenance for connected-shot identity-refresh strategies.

The external pretrained video foundation remains explicitly external. This module
only classifies the CINEOS-owned continuity conditioning strategy used by a
persistent GPU session so baseline and candidate A/B runs cannot be confused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .production_continuity_identity import (
    CONTINUITY_IDENTITY_ADAPTER_ID,
    CONTINUITY_IDENTITY_ADAPTER_VERSION,
    compose_continuity_identity_board,
)

CONTINUITY_IDENTITY_RUNTIME_SCHEMA = "cineos-continuity-identity-runtime/0.1"


class ContinuityIdentityRuntimeError(RuntimeError):
    """Raised when continuity strategy provenance cannot be represented safely."""


def bind_continuity_identity_runtime(
    runtime: Mapping[str, Any],
    continuity_identity_adapter: Any | None,
) -> dict[str, Any]:
    """Bind baseline/candidate strategy provenance to one GPU runtime receipt.

    ``None`` is the validated predecessor-terminal-frame baseline. The exact
    first-party deterministic compositor is the experimental CINEOS candidate and
    remains eligible for production GPU evidence because it is not a test injection
    or borrowed engine. Any other callable is explicitly classified as an injected
    boundary and therefore cannot masquerade as default production execution.
    """

    normalized = dict(runtime)
    boundaries = normalized.get("injected_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ContinuityIdentityRuntimeError(
            "GPU runtime provenance is missing injected-boundary evidence"
        )
    updated_boundaries = dict(boundaries)

    if continuity_identity_adapter is None:
        updated_boundaries["continuity_identity_adapter"] = False
        strategy = {
            "schema": CONTINUITY_IDENTITY_RUNTIME_SCHEMA,
            "mode": "predecessor_terminal_frame_baseline",
            "adapter_id": None,
            "adapter_version": None,
            "experimental": False,
        }
    elif continuity_identity_adapter is compose_continuity_identity_board:
        updated_boundaries["continuity_identity_adapter"] = False
        strategy = {
            "schema": CONTINUITY_IDENTITY_RUNTIME_SCHEMA,
            "mode": "predecessor_terminal_frame_plus_fresh_references",
            "adapter_id": CONTINUITY_IDENTITY_ADAPTER_ID,
            "adapter_version": CONTINUITY_IDENTITY_ADAPTER_VERSION,
            "experimental": True,
        }
    else:
        updated_boundaries["continuity_identity_adapter"] = True
        strategy = {
            "schema": CONTINUITY_IDENTITY_RUNTIME_SCHEMA,
            "mode": "injected_or_unrecognized",
            "adapter_id": None,
            "adapter_version": None,
            "experimental": True,
        }

    normalized["injected_boundaries"] = updated_boundaries
    runtime_mode = "injected" if any(updated_boundaries.values()) else "default"
    normalized["runtime_mode"] = runtime_mode
    normalized["production_default_runtime"] = runtime_mode == "default"
    normalized["continuity_identity_strategy"] = strategy
    return normalized


__all__ = [
    "CONTINUITY_IDENTITY_RUNTIME_SCHEMA",
    "ContinuityIdentityRuntimeError",
    "bind_continuity_identity_runtime",
]
