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
PRODUCTION_REFERENCE_BOARD_ADAPTER_ID = "cineos.production.reference_board"
PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION = "0.1.3"
PRODUCTION_REFERENCE_BOARD_MAXIMUM_REFERENCES = 4


class ProductionMultiReferenceError(RuntimeError):
    """Raised when production multi-reference conditioning cannot be audited."""


def _validated_character_reference_bindings(
    request: NativeShotRequest,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate and return the exact character-to-reference ownership contract.

    ProductionDiffusersVideoRenderer validates the same lineage before invoking the
    default adapter. Re-validating at this CINEOS-owned pixel-conditioning boundary
    is intentional defense in depth: the adapter is also a public callable and must
    never compose a board from malformed, escaped, unowned or ambiguously owned
    identity references when invoked independently.

    Legacy requests without character metadata remain supported and return no
    ownership bindings; in that case the board still consumes the globally approved
    references in request order exactly as before.
    """

    if not request.characters:
        return ()

    approved = set(request.approved_reference_ids)
    multi_character = len(request.characters) > 1
    reference_owner: dict[str, str] = {}
    character_ids: set[str] = set()
    bindings: list[tuple[str, tuple[str, ...]]] = []

    for index, character in enumerate(request.characters):
        if not isinstance(character, dict):
            raise ProductionMultiReferenceError(
                f"production character conditioning {index} must be an object"
            )

        character_id = character.get("character_uuid", f"index:{index}")
        if not isinstance(character_id, str) or not character_id.strip():
            character_id = f"index:{index}"
        else:
            character_id = character_id.strip()

        if multi_character:
            if character_id.startswith("index:"):
                raise ProductionMultiReferenceError(
                    "multi-character production conditioning requires a non-empty "
                    "character_uuid for every character"
                )
            if character_id in character_ids:
                raise ProductionMultiReferenceError(
                    "multi-character production conditioning requires unique "
                    f"character_uuid values: duplicate {character_id!r}"
                )
            character_ids.add(character_id)

        raw_ids = character.get("approved_reference_ids", [])
        if not isinstance(raw_ids, (list, tuple)) or any(
            not isinstance(reference_id, str) or not reference_id.strip()
            for reference_id in raw_ids
        ):
            raise ProductionMultiReferenceError(
                "character approved_reference_ids must be a sequence of non-empty "
                "strings"
            )
        if multi_character and not raw_ids:
            raise ProductionMultiReferenceError(
                "multi-character production conditioning requires at least one "
                f"approved reference per character: {character_id!r} has none"
            )

        escaped = [
            reference_id for reference_id in raw_ids if reference_id not in approved
        ]
        if escaped:
            raise ProductionMultiReferenceError(
                "character conditioning references are not approved by the shot: "
                f"{character_id!r} -> {', '.join(escaped)}"
            )

        for reference_id in raw_ids:
            previous_owner = reference_owner.get(reference_id)
            if previous_owner is not None and previous_owner != character_id:
                raise ProductionMultiReferenceError(
                    "character identity reference is ambiguously assigned to multiple "
                    "characters: "
                    f"{reference_id!r} -> {previous_owner!r}, {character_id!r}"
                )
            reference_owner[reference_id] = character_id

        if raw_ids:
            bindings.append((character_id, tuple(raw_ids)))

    unowned = [
        reference_id
        for reference_id in request.approved_reference_ids
        if reference_id not in reference_owner
    ]
    if unowned:
        scope = "multi-character" if multi_character else "single-character"
        raise ProductionMultiReferenceError(
            f"{scope} production conditioning requires every approved identity "
            "reference to have exactly one character owner; unowned: "
            + ", ".join(unowned)
        )

    return tuple(bindings)


def _character_reference_bindings(
    request: NativeShotRequest,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return validated character-to-reference ownership evidence."""

    return _validated_character_reference_bindings(request)


def _reference_board_cells(
    reference_count: int, width: int, height: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Return deterministic full-frame cells for two to four references.

    Three-reference conditioning deliberately uses a 2+1 layout: two equal cells on
    the top row and one full-width cell on the bottom row. The previous generic 2x2
    layout left one quadrant permanently gray, wasting 25% of the available
    conditioning canvas and shrinking the third identity to a quarter-frame cell.
    """

    half_width = width // 2
    half_height = height // 2
    if reference_count == 2:
        return (
            (0, 0, half_width, height),
            (half_width, 0, width - half_width, height),
        )
    if reference_count == 3:
        return (
            (0, 0, half_width, half_height),
            (half_width, 0, width - half_width, half_height),
            (0, half_height, width, height - half_height),
        )
    if reference_count == 4:
        return (
            (0, 0, half_width, half_height),
            (half_width, 0, width - half_width, half_height),
            (0, half_height, half_width, height - half_height),
            (half_width, half_height, width - half_width, height - half_height),
        )
    raise ProductionMultiReferenceError(
        "production reference board requires two to four approved references"
    )


class ProductionReferenceBoardAdapter:
    """Compose 2-4 approved references into a deterministic shot-aspect board.

    No generated pixels, labels, face swaps, or external services are involved.
    Each source is contain-fitted into a stable cell without cropping so identity
    evidence is not silently discarded. The adapter is CINEOS-owned preprocessing,
    not a native capability claim about the external video foundation.

    Every approved reference id must be unique. Repeating one identity under the
    same id could otherwise occupy multiple board cells and make a two-character
    request appear fully conditioned while one distinct character is absent.
    """

    adapter_id = PRODUCTION_REFERENCE_BOARD_ADAPTER_ID
    adapter_version = PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION
    maximum_references = PRODUCTION_REFERENCE_BOARD_MAXIMUM_REFERENCES

    def __call__(
        self, request: NativeShotRequest, references: Sequence[Any]
    ) -> MultiReferenceConditioningResult:
        expected_ids = tuple(request.approved_reference_ids)
        if len(set(expected_ids)) != len(expected_ids):
            duplicates = sorted(
                reference_id
                for reference_id in set(expected_ids)
                if expected_ids.count(reference_id) > 1
            )
            raise ProductionMultiReferenceError(
                "production multi-reference conditioning requires unique approved "
                "reference ids; duplicates: " + ", ".join(duplicates)
            )
        if len(references) != len(expected_ids):
            raise ProductionMultiReferenceError(
                "multi-reference board received a different number of images than "
                "approved reference ids"
            )
        if len(references) < 2:
            raise ProductionMultiReferenceError(
                "multi-reference board requires at least two approved references"
            )
        if len(references) > PRODUCTION_REFERENCE_BOARD_MAXIMUM_REFERENCES:
            raise ProductionMultiReferenceError(
                "production reference board supports at most four identities per shot"
            )

        # Fail before any image decoding/composition if identity ownership is not
        # auditable. This prevents direct adapter use from bypassing the production
        # renderer's lineage gate and producing a visually conditioned but
        # semantically misbound multi-character board.
        character_bindings = _validated_character_reference_bindings(request)

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

        cells = _reference_board_cells(len(references), width, height)
        board = Image.new("RGB", (width, height), (127, 127, 127))

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
            cell_left, cell_top, cell_width, cell_height = cells[index]
            if cell_width <= 0 or cell_height <= 0:
                raise ProductionMultiReferenceError(
                    "shot camera resolution is too small for multi-reference board layout"
                )
            scale = min(cell_width / source_width, cell_height / source_height)
            target = (
                max(1, round(source_width * scale)),
                max(1, round(source_height * scale)),
            )
            resampling = getattr(Image, "Resampling", Image)
            fitted = image.resize(target, resampling.LANCZOS)
            left = cell_left + (cell_width - target[0]) // 2
            top = cell_top + (cell_height - target[1]) // 2
            board.paste(fitted, (left, top))

        return MultiReferenceConditioningResult(
            image=board,
            consumed_reference_ids=expected_ids,
            adapter_id=PRODUCTION_REFERENCE_BOARD_ADAPTER_ID,
            adapter_version=PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION,
            consumed_character_reference_ids=character_bindings or None,
        )

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": MULTI_REFERENCE_RUNTIME_SCHEMA,
            "adapter": (
                "cineos.atlas.production_multi_reference."
                "ProductionReferenceBoardAdapter"
            ),
            "adapter_id": PRODUCTION_REFERENCE_BOARD_ADAPTER_ID,
            "adapter_version": PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION,
            "maximum_references": PRODUCTION_REFERENCE_BOARD_MAXIMUM_REFERENCES,
            "composition": "deterministic_contain_fit_reference_board",
            "requires_unique_reference_ids": True,
            "attests_character_reference_ownership": True,
        }


def bind_production_multi_reference_runtime(
    runtime: Mapping[str, Any], adapter: Any | None
) -> dict[str, Any]:
    """Promote only the exact first-party adapter to production runtime evidence.

    Exact type identity is intentional. An injected subclass can override execution
    while still satisfying ``isinstance`` and must therefore remain an injected
    boundary rather than being mislabeled as the CINEOS-owned default runtime.
    """

    normalized = dict(runtime)
    boundaries = normalized.get("injected_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ProductionMultiReferenceError(
            "GPU runtime provenance is missing injected-boundary evidence"
        )
    updated = dict(boundaries)
    updated["multi_reference_adapter"] = adapter is not None
    if type(adapter) is ProductionReferenceBoardAdapter:
        updated["multi_reference_adapter"] = False
        normalized["multi_reference_conditioning"] = adapter.runtime_provenance()
    normalized["injected_boundaries"] = updated
    runtime_mode = "injected" if any(updated.values()) else "default"
    normalized["runtime_mode"] = runtime_mode
    normalized["production_default_runtime"] = runtime_mode == "default"
    return normalized


__all__ = [
    "MULTI_REFERENCE_RUNTIME_SCHEMA",
    "PRODUCTION_REFERENCE_BOARD_ADAPTER_ID",
    "PRODUCTION_REFERENCE_BOARD_ADAPTER_VERSION",
    "PRODUCTION_REFERENCE_BOARD_MAXIMUM_REFERENCES",
    "ProductionMultiReferenceError",
    "ProductionReferenceBoardAdapter",
    "bind_production_multi_reference_runtime",
]
