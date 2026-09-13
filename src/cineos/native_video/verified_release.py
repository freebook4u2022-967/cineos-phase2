"""High-assurance production composition for verified CINEOS model releases.

The ordinary released runtime binds durable film state to a versioned model manifest.
This stricter composition additionally verifies the actual component files on disk
against that manifest before the renderer is allowed into a production film job.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cineos.native_image.artifact_verification import (
    ModelArtifactAttestation,
    verify_component_artifacts,
)
from cineos.native_image.model_manifest import ModelManifestError, NativeModelRegistry

from .production_first_film import (
    ProductionFirstFilmRuntime,
    build_released_production_first_film_runtime,
)
from .renderer_binding import NativeTemporalShotRenderer


@dataclass(frozen=True, slots=True)
class VerifiedProductionFirstFilmRuntime:
    """Production runtime plus cryptographic evidence for learned component bytes."""

    runtime: ProductionFirstFilmRuntime
    model_artifacts: ModelArtifactAttestation


def build_verified_released_production_first_film_runtime(
    native_renderer: NativeTemporalShotRenderer,
    model_registry: NativeModelRegistry,
    component_artifacts: Mapping[str, str | Path],
    validator: Any | None = None,
    **kwargs: Any,
) -> VerifiedProductionFirstFilmRuntime:
    """Build a released runtime only after all active model files hash-verify.

    Verification is performed before runtime composition.  The active manifest digest
    is then passed explicitly to the existing released builder, which re-reads the
    registry and fails if activation changed between verification and composition.
    This closes the gap between "manifest is valid" and "the bytes about to be used
    are exactly the bytes declared by that manifest" without weakening the existing
    compatibility and durable-resume gates.
    """

    active = model_registry.active()
    if active is None:
        raise ModelManifestError(
            "verified production FIRST FILM requires an active native model release"
        )

    attestation = verify_component_artifacts(active, component_artifacts)
    requested_digest = kwargs.pop("native_model_manifest_sha256", None)
    if requested_digest is not None and requested_digest != active.manifest_sha256:
        raise ModelManifestError(
            "explicit native model manifest does not match verified active release"
        )

    runtime = build_released_production_first_film_runtime(
        native_renderer,
        model_registry,
        validator,
        native_model_manifest_sha256=active.manifest_sha256,
        **kwargs,
    )
    if runtime.manifest.native_model_manifest_sha256 != attestation.manifest_sha256:
        raise ModelManifestError(
            "verified model attestation does not match composed production runtime"
        )
    return VerifiedProductionFirstFilmRuntime(
        runtime=runtime,
        model_artifacts=attestation,
    )


__all__ = [
    "VerifiedProductionFirstFilmRuntime",
    "build_verified_released_production_first_film_runtime",
]
