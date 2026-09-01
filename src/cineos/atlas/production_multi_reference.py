"""CINEOS-native production adapter for multiple approved visual references.

The external foundation still owns image-to-video generation. This module only
constructs a deterministic conditioning board from already approved, hash-bound
reference images so a single-image foundation slot can receive every declared
identity reference without silently dropping characters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .native_request import NativeShotRequest
from .production_diffusers import MultiReferenceConditioningResult

MULTI_REFERENCE_RUNTIME_SCHEMA = "cineos-production-multi-reference-runtime/0.1"


class ProductionMultiReferenceError(RuntimeError):
    """Raised when production multi-reference conditioning cannot be audited."""


class ProductionReferenceBoardAdapter:
    """Compose 2-4 approved references into a deterministic shot-aspect board.

    No generated pixels, labels, face swaps, or external services are involved.
    Each source is contain-fitted into a stable cell without cropping so identity
    evidence is not silently discarded. The adapter is CINEOS-owned preprocessing,
    not a native capability claim about the external video foundation.
    """

    adapter_id = "cineos.production.reference_board"
    adapter_version = "0.1.0"
    maximum_references = 4

    def __call__(
        self, request: NativeShotRequest, references: Sequence[Any]
    ) -> MultiReferenceConditioningResult:
        expected_ids = tuple(request.approved_reference_ids)
        if len(references) != len(expected_ids):
            raise ProductionMultiReferenceError(
                "multi-reference board received a different number of images than "
                "approved reference ids"
            )
        if len(references) < 2:
            raise ProductionMultiReferenceError(
                "multi-reference board requires at least two approved references"
            )
        if len(references) > self.maximum_references:
            raise ProductionMultiReferenceError(
                "production reference board supports at most four identities per shot"
            )

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - exercised in video extra env
            raise ProductionMultiReferenceError(
                "production multi-reference conditioning requires Pillow"
            ) from exc

        raw_resolution = request.camera.get("resolution", (1280, 704))
        try:
            width, height = (int(raw_resolution[0]), int(raw_resolution[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ProductionMultiReferenceError(
                "shot camera resolution is invalid for reference-board composition"
            ) from exc
        if width <= 0 or height <= 0:
            raise ProductionMultiReferenceError(
                "shot camera resolution must be positive for reference-board composition"
            )

        columns, rows = (2, 1) if len(references) == 2 else (2, 2)
        board = Image.new("RGB", (width, height), (127, 127, 127))
        cell_width = width // columns
        cell_height = height // rows

        for index, source in enumerate(references):
            if not hasattr(source, "convert") or not hasattr(source, "resize"):
                raise ProductionMultiReferenceError(
                    f"approved reference {expected_ids[index]!r} is not image-like"
                )
            image = source.convert("RGB")
            source_width, source_height = image.size
            if source_width <= 0 or source_height <= 0:
                raise ProductionMultiReferenceError(
                    f"approved reference {expected_ids[index]!r} has invalid dimensions"
                )
            scale = min(cell_width / source_width, cell_height / source_height)
            target = (
                max(1, round(source_width * scale)),
                max(1, round(source_height * scale)),
            )
            resampling = getattr(Image, "Resampling", Image)
            fitted = image.resize(target, resampling.LANCZOS)
            column = index % columns
            row = index // columns
            left = column * cell_width + (cell_width - target[0]) // 2
            top = row * cell_height + (cell_height - target[1]) // 2
            board.paste(fitted, (left, top))

        return MultiReferenceConditioningResult(
            image=board,
            consumed_reference_ids=expected_ids,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
        )

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": MULTI_REFERENCE_RUNTIME_SCHEMA,
            "adapter": (
                "cineos.atlas.production_multi_reference."
                "ProductionReferenceBoardAdapter"
            ),
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "maximum_references": self.maximum_references,
            "composition": "deterministic_contain_fit_reference_board",
        }


def bind_production_multi_reference_runtime(
    runtime: Mapping[str, Any], adapter: Any | None
) -> dict[str, Any]:
    """Promote only the exact first-party adapter to production runtime evidence."""

    normalized = dict(runtime)
    boundaries = normalized.get("injected_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ProductionMultiReferenceError(
            "GPU runtime provenance is missing injected-boundary evidence"
        )
    updated = dict(boundaries)
    updated["multi_reference_adapter"] = adapter is not None
    if isinstance(adapter, ProductionReferenceBoardAdapter):
        updated["multi_reference_adapter"] = False
        normalized["multi_reference_conditioning"] = adapter.runtime_provenance()
    normalized["injected_boundaries"] = updated
    runtime_mode = "injected" if any(updated.values()) else "default"
    normalized["runtime_mode"] = runtime_mode
    normalized["production_default_runtime"] = runtime_mode == "default"
    return normalized


__all__ = [
    "MULTI_REFERENCE_RUNTIME_SCHEMA",
    "ProductionMultiReferenceError",
    "ProductionReferenceBoardAdapter",
    "bind_production_multi_reference_runtime",
]
