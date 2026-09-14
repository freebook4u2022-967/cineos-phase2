"""Fresh identity refresh for connected production video conditioning.

The external Diffusers video foundation still owns generation. CINEOS owns this
adapter and orchestration only. Wan2.2 exposes one image-conditioning slot, so a
connected shot cannot independently pass both its predecessor terminal frame and
fresh approved identity images. This module composes both signals into one
explicit, auditable conditioning image instead of silently dropping either.

The compositor is deliberately deterministic and preserves the predecessor frame
as the dominant visual signal. Approved identity references occupy a smaller
refresh strip. GPU benchmark evidence must determine whether this strategy beats
predecessor-only inheritance for a given foundation; the provenance below keeps
that distinction measurable rather than calling the behavior native model
capability.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .diffusers_video import DiffusersVideoError
from .native_request import NativeShotRequest
from .production_continuity_diffusers import ProductionContinuityDiffusersVideoRenderer
from .production_diffusers import ProductionDiffusersVideoResult
from .reference_board import compose_reference_board

CONTINUITY_IDENTITY_ADAPTER_ID = "cineos.continuity-identity-board"
CONTINUITY_IDENTITY_ADAPTER_VERSION = "1.0"
CONTINUITY_IDENTITY_SCHEMA = "cineos-continuity-identity-conditioning/1.0"


@dataclass(frozen=True, slots=True)
class ContinuityIdentityConditioningResult:
    """Auditable single-slot composition of continuity and identity signals."""

    image: Any
    consumed_reference_ids: tuple[str, ...]
    adapter_id: str
    adapter_version: str
    predecessor_frame_consumed: bool


ContinuityIdentityAdapter = Callable[
    [NativeShotRequest, Any, Sequence[Any]], ContinuityIdentityConditioningResult
]


def compose_continuity_identity_board(
    request: NativeShotRequest,
    predecessor_frame: Any,
    references: Sequence[Any],
) -> ContinuityIdentityConditioningResult:
    """Compose predecessor visual state plus every approved identity reference.

    Landscape shots reserve the right quarter for identity refresh; portrait shots
    reserve the bottom quarter. The predecessor is contain-fitted into the remaining
    dominant region without cropping. Multi-character identity images first pass
    through the existing duplicate-safe reference-board adapter. A single-character
    shot uses its one approved image directly so connected single-character films do
    not regress merely because the multi-reference adapter requires two inputs.
    """

    expected = tuple(request.approved_reference_ids)
    if not expected:
        raise DiffusersVideoError(
            "continuity identity refresh requires approved identity references"
        )
    if len(references) != len(expected):
        raise DiffusersVideoError(
            "continuity identity refresh received a different number of images than "
            "approved reference ids"
        )
    if len(expected) != len(set(expected)):
        raise DiffusersVideoError(
            "continuity identity refresh requires unique approved reference ids"
        )

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - video extra dependency
        raise DiffusersVideoError(
            "continuity identity refresh requires Pillow; install cineos[video]"
        ) from exc

    if not isinstance(predecessor_frame, Image.Image):
        raise DiffusersVideoError(
            "continuity identity refresh requires a Pillow predecessor frame"
        )
    predecessor = predecessor_frame.convert("RGB")
    if predecessor.width <= 0 or predecessor.height <= 0:
        raise DiffusersVideoError("continuity predecessor frame has invalid dimensions")

    if len(references) == 1:
        reference = references[0]
        if not isinstance(reference, Image.Image):
            raise DiffusersVideoError(
                "continuity identity refresh requires Pillow identity references"
            )
        identity_board = reference.convert("RGB")
        if identity_board.width <= 0 or identity_board.height <= 0:
            raise DiffusersVideoError(
                "continuity identity reference has invalid dimensions"
            )
    else:
        identity_board = compose_reference_board(request, references).image
        if not isinstance(identity_board, Image.Image):
            raise DiffusersVideoError(
                "reference-board adapter returned a non-image result"
            )

    raw_resolution = request.camera.get("resolution", (0, 0))
    try:
        width, height = int(raw_resolution[0]), int(raw_resolution[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise DiffusersVideoError(
            "continuity identity refresh requires a valid camera resolution"
        ) from exc
    if width <= 3 or height <= 3:
        raise DiffusersVideoError(
            "continuity identity refresh requires a camera resolution above 3x3"
        )

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    if width >= height:
        predecessor_box = (max(1, width * 3 // 4), height)
        identity_box = (width - predecessor_box[0], height)
        identity_origin = (predecessor_box[0], 0)
    else:
        predecessor_box = (width, max(1, height * 3 // 4))
        identity_box = (width, height - predecessor_box[1])
        identity_origin = (0, predecessor_box[1])

    if min(*predecessor_box, *identity_box) <= 0:
        raise DiffusersVideoError(
            "continuity identity refresh produced an invalid conditioning layout"
        )

    predecessor_fit = ImageOps.contain(
        predecessor,
        predecessor_box,
        method=Image.Resampling.LANCZOS,
    )
    px = (predecessor_box[0] - predecessor_fit.width) // 2
    py = (predecessor_box[1] - predecessor_fit.height) // 2
    canvas.paste(predecessor_fit, (px, py))

    identity_fit = ImageOps.contain(
        identity_board.convert("RGB"),
        identity_box,
        method=Image.Resampling.LANCZOS,
    )
    ix = identity_origin[0] + (identity_box[0] - identity_fit.width) // 2
    iy = identity_origin[1] + (identity_box[1] - identity_fit.height) // 2
    canvas.paste(identity_fit, (ix, iy))

    return ContinuityIdentityConditioningResult(
        image=canvas,
        consumed_reference_ids=expected,
        adapter_id=CONTINUITY_IDENTITY_ADAPTER_ID,
        adapter_version=CONTINUITY_IDENTITY_ADAPTER_VERSION,
        predecessor_frame_consumed=True,
    )


class ProductionContinuityIdentityDiffusersVideoRenderer(
    ProductionContinuityDiffusersVideoRenderer
):
    """Connected renderer that can refresh identity pixels on every continuation.

    When ``continuity_identity_adapter`` is configured, continuation shots with
    approved references consume both the predecessor frame and freshly resolved
    reference pixels through a deterministic single-slot composition. Passing
    ``None`` preserves the predecessor-only behavior of the parent renderer for
    backwards compatibility and A/B benchmarking.
    """

    def __init__(
        self,
        *args: Any,
        continuity_identity_adapter: ContinuityIdentityAdapter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.continuity_identity_adapter = continuity_identity_adapter
        self._prepared_continuity_identity_image: Any | None = None

    def render(self, request: Any) -> ProductionDiffusersVideoResult:
        self._prepared_continuity_identity_image = None
        try:
            return super().render(request)
        finally:
            self._prepared_continuity_identity_image = None

    def _verify_reference_conditioning_path(self, request: NativeShotRequest) -> None:
        if (
            self._active_continuity_frame is None
            or self.continuity_identity_adapter is None
            or not request.approved_reference_ids
        ):
            return super()._verify_reference_conditioning_path(request)

        self._validate_character_reference_lineage(request)
        if self.reference_loader is None:
            raise DiffusersVideoError(
                "continuation identity refresh requires a production reference_loader"
            )
        if self._pipeline is None:
            raise DiffusersVideoError("renderer model is not loaded")

        parameters = inspect.signature(self._pipeline.__call__).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "image" not in parameters and not accepts_kwargs:
            raise DiffusersVideoError(
                "connected identity refresh requires image conditioning, but the "
                "loaded foundation pipeline does not expose it"
            )

        resolved: list[Any] = []
        for reference_id in request.approved_reference_ids:
            reference = self.reference_loader(reference_id)
            if reference is None:
                raise DiffusersVideoError(
                    "approved identity reference could not be resolved for connected "
                    f"shot {request.shot_id!r}: {reference_id!r}"
                )
            resolved.append(reference)

        result = self.continuity_identity_adapter(
            request,
            self._active_continuity_frame,
            tuple(resolved),
        )
        if not isinstance(result, ContinuityIdentityConditioningResult):
            raise DiffusersVideoError(
                "continuity_identity_adapter must return "
                "ContinuityIdentityConditioningResult"
            )
        expected = tuple(request.approved_reference_ids)
        if result.consumed_reference_ids != expected:
            raise DiffusersVideoError(
                "continuity_identity_adapter did not attest consumption of every "
                "approved reference in request order"
            )
        if result.image is None or not result.predecessor_frame_consumed:
            raise DiffusersVideoError(
                "continuity_identity_adapter must consume both predecessor and "
                "identity conditioning"
            )
        if not result.adapter_id.strip() or not result.adapter_version.strip():
            raise DiffusersVideoError(
                "continuity_identity_adapter must declare adapter provenance"
            )

        self._prepared_continuity_identity_image = result.image
        self._conditioning_provenance = {
            "schema": CONTINUITY_IDENTITY_SCHEMA,
            "mode": "predecessor_terminal_frame_plus_fresh_references",
            "consumed_reference_ids": list(result.consumed_reference_ids),
            "adapter_id": result.adapter_id.strip(),
            "adapter_version": result.adapter_version.strip(),
            "identity_signal_source": (
                "predecessor_terminal_frame_and_fresh_approved_references"
            ),
            "fresh_reference_pixels_consumed": True,
            "predecessor_frame_consumed": True,
        }

    def _load_primary_reference(self, request: NativeShotRequest) -> Any | None:
        if self._active_continuity_frame is not None:
            if self._prepared_continuity_identity_image is not None:
                return self._prepared_continuity_identity_image
            return self._active_continuity_frame
        return super()._load_primary_reference(request)


__all__ = [
    "CONTINUITY_IDENTITY_ADAPTER_ID",
    "CONTINUITY_IDENTITY_ADAPTER_VERSION",
    "CONTINUITY_IDENTITY_SCHEMA",
    "ContinuityIdentityAdapter",
    "ContinuityIdentityConditioningResult",
    "ProductionContinuityIdentityDiffusersVideoRenderer",
    "compose_continuity_identity_board",
]
