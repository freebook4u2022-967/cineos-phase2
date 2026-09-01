"""Production-safe Diffusers boundary for CINEOS reference-conditioned shots.

This module strengthens the execution contract around external pretrained video
foundations. It does not make the external checkpoint CINEOS-native. Instead,
it prevents a production shot that declares approved visual references from
silently degrading to text-only or partial-reference generation when those
references cannot actually reach the foundation pipeline.
"""

from __future__ import annotations

import inspect
from typing import Any

from .diffusers_video import DiffusersVideoError, DiffusersVideoRenderer
from .native_request import NativeShotRequest


class ProductionDiffusersVideoRenderer(DiffusersVideoRenderer):
    """Diffusers renderer with fail-closed approved-reference conditioning.

    Research callers may continue to use :class:`DiffusersVideoRenderer`, which
    preserves its historical permissive behavior. Production foundation profiles
    use this subclass so an approved identity reference is never merely recorded in
    a CINEOS request while being ignored by the external model.

    The generic Diffusers execution boundary currently has a single ``image``
    conditioning slot. Consequently, production requests that declare more than one
    approved reference fail closed rather than silently forwarding only the first
    reference. Multi-character production work must therefore use one deliberately
    composed/approved conditioning image until a foundation-specific adapter with an
    audited multi-reference contract is available.
    """

    def render(self, request: Any):
        if isinstance(request, NativeShotRequest) and request.approved_reference_ids:
            self._verify_reference_conditioning_path(request)
        return super().render(request)

    def _verify_reference_conditioning_path(self, request: NativeShotRequest) -> None:
        if self.reference_loader is None:
            raise DiffusersVideoError(
                "production shot declares approved_reference_ids but no "
                "reference_loader is configured"
            )
        if self._pipeline is None:
            raise DiffusersVideoError("renderer model is not loaded")

        if len(request.approved_reference_ids) > 1:
            raise DiffusersVideoError(
                "production Diffusers boundary cannot safely consume multiple "
                "approved_reference_ids through its single image-conditioning slot; "
                "provide one deliberately composed approved reference or use an "
                "audited foundation-specific multi-reference adapter"
            )

        parameters = inspect.signature(self._pipeline.__call__).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "image" not in parameters and not accepts_kwargs:
            raise DiffusersVideoError(
                "production shot declares approved_reference_ids but the loaded "
                "foundation pipeline does not expose image conditioning"
            )

    def _load_primary_reference(self, request: NativeShotRequest) -> Any | None:
        reference = super()._load_primary_reference(request)
        if request.approved_reference_ids and reference is None:
            raise DiffusersVideoError(
                "approved identity reference could not be resolved for production "
                f"shot {request.shot_id!r}"
            )
        return reference


__all__ = ["ProductionDiffusersVideoRenderer"]
