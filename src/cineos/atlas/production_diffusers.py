"""Production-safe Diffusers boundary for CINEOS reference-conditioned shots.

This module strengthens the execution contract around external pretrained video
foundations. It does not make the external checkpoint CINEOS-native. Instead,
it prevents a production shot that declares approved visual references from
silently degrading to text-only or partial-reference generation when those
references cannot actually reach the foundation pipeline.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .diffusers_video import DiffusersVideoError, DiffusersVideoRenderer
from .native_request import NativeShotRequest


@dataclass(frozen=True, slots=True)
class MultiReferenceConditioningResult:
    """Auditable output from a CINEOS-approved multi-reference adapter.

    ``consumed_reference_ids`` must exactly match the approved IDs on the shot.
    This prevents an adapter from silently dropping one character/reference while
    still allowing a foundation with one image-conditioning slot to consume a
    deliberately composed conditioning image.
    """

    image: Any
    consumed_reference_ids: tuple[str, ...]
    adapter_id: str
    adapter_version: str


MultiReferenceAdapter = Callable[
    [NativeShotRequest, Sequence[Any]], MultiReferenceConditioningResult
]


class ProductionDiffusersVideoRenderer(DiffusersVideoRenderer):
    """Diffusers renderer with fail-closed approved-reference conditioning.

    Research callers may continue to use :class:`DiffusersVideoRenderer`, which
    preserves its historical permissive behavior. Production foundation profiles
    use this subclass so an approved identity reference is never merely recorded in
    a CINEOS request while being ignored by the external model.

    The generic Diffusers execution boundary currently has a single ``image``
    conditioning slot. Multiple approved references therefore require an explicit
    audited ``multi_reference_adapter`` which must consume every approved reference
    and return one composed conditioning image. Without that adapter, production
    requests fail closed rather than silently forwarding only the first reference.
    """

    def __init__(
        self,
        *args: Any,
        multi_reference_adapter: MultiReferenceAdapter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.multi_reference_adapter = multi_reference_adapter
        self._prepared_multi_reference_image: Any | None = None

    def render(self, request: Any):
        self._prepared_multi_reference_image = None
        if isinstance(request, NativeShotRequest) and request.approved_reference_ids:
            self._verify_reference_conditioning_path(request)
        try:
            return super().render(request)
        finally:
            self._prepared_multi_reference_image = None

    def _verify_reference_conditioning_path(self, request: NativeShotRequest) -> None:
        if self.reference_loader is None:
            raise DiffusersVideoError(
                "production shot declares approved_reference_ids but no "
                "reference_loader is configured"
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
                "production shot declares approved_reference_ids but the loaded "
                "foundation pipeline does not expose image conditioning"
            )

        if len(request.approved_reference_ids) > 1:
            self._prepared_multi_reference_image = self._prepare_multi_reference_image(
                request
            )

    def _prepare_multi_reference_image(self, request: NativeShotRequest) -> Any:
        if self.multi_reference_adapter is None:
            raise DiffusersVideoError(
                "production Diffusers boundary cannot safely consume multiple "
                "approved_reference_ids through its single image-conditioning slot; "
                "configure an audited multi_reference_adapter"
            )
        assert self.reference_loader is not None

        resolved: list[Any] = []
        for reference_id in request.approved_reference_ids:
            reference = self.reference_loader(reference_id)
            if reference is None:
                raise DiffusersVideoError(
                    "approved identity reference could not be resolved for production "
                    f"shot {request.shot_id!r}: {reference_id!r}"
                )
            resolved.append(reference)

        result = self.multi_reference_adapter(request, tuple(resolved))
        if not isinstance(result, MultiReferenceConditioningResult):
            raise DiffusersVideoError(
                "multi_reference_adapter must return MultiReferenceConditioningResult"
            )
        expected = tuple(request.approved_reference_ids)
        if result.consumed_reference_ids != expected:
            raise DiffusersVideoError(
                "multi_reference_adapter did not attest consumption of every approved "
                "reference in request order"
            )
        if result.image is None:
            raise DiffusersVideoError(
                "multi_reference_adapter returned no conditioning image"
            )
        if not result.adapter_id.strip() or not result.adapter_version.strip():
            raise DiffusersVideoError(
                "multi_reference_adapter must declare non-empty adapter_id and "
                "adapter_version provenance"
            )
        return result.image

    def _load_primary_reference(self, request: NativeShotRequest) -> Any | None:
        if len(request.approved_reference_ids) > 1:
            if self._prepared_multi_reference_image is None:
                raise DiffusersVideoError(
                    "multi-reference conditioning was not prepared before inference"
                )
            return self._prepared_multi_reference_image

        reference = super()._load_primary_reference(request)
        if request.approved_reference_ids and reference is None:
            raise DiffusersVideoError(
                "approved identity reference could not be resolved for production "
                f"shot {request.shot_id!r}"
            )
        return reference


__all__ = [
    "MultiReferenceAdapter",
    "MultiReferenceConditioningResult",
    "ProductionDiffusersVideoRenderer",
]
