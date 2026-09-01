"""Production-safe Diffusers boundary for CINEOS reference-conditioned shots.

This module strengthens the execution contract around external pretrained video
foundations. It does not make the external checkpoint CINEOS-native. Instead,
it prevents a production shot that declares approved visual references from
silently degrading to text-only or partial-reference generation when those
references cannot actually reach the foundation pipeline.
"""

from __future__ import annotations

import inspect
import json
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

    Production prompt compilation also appends compact CINEOS-owned identity and
    continuity constraints to any director-authored prompt. This prevents an
    explicit prompt from accidentally suppressing structured CineDNA invariants or
    reference-to-character lineage before external-foundation inference.
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
        self._validate_character_reference_lineage(request)
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

    @staticmethod
    def _validate_character_reference_lineage(request: NativeShotRequest) -> None:
        """Reject character-level references that escape the approved shot lineage."""

        approved = set(request.approved_reference_ids)
        for index, character in enumerate(request.characters):
            if not isinstance(character, dict):
                raise DiffusersVideoError(
                    f"production character conditioning {index} must be an object"
                )
            raw_ids = character.get("approved_reference_ids", [])
            if not isinstance(raw_ids, (list, tuple)) or any(
                not isinstance(reference_id, str) or not reference_id.strip()
                for reference_id in raw_ids
            ):
                raise DiffusersVideoError(
                    "character approved_reference_ids must be a sequence of non-empty "
                    "strings"
                )
            escaped = [
                reference_id for reference_id in raw_ids if reference_id not in approved
            ]
            if escaped:
                character_id = character.get("character_uuid", f"index:{index}")
                raise DiffusersVideoError(
                    "character conditioning references are not approved by the shot: "
                    f"{character_id!r} -> {', '.join(escaped)}"
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

    @staticmethod
    def _compile_prompt(request: NativeShotRequest) -> str:
        """Preserve director prompt while injecting structured production constraints.

        The base renderer historically returned ``metadata['prompt']`` verbatim when
        present. For production that discards structured character identity and
        continuity information exactly when a high-quality hand-authored prompt is
        supplied. We retain that prompt, then append deterministic compact JSON with
        only identity/continuity fields that materially affect connected-shot quality.
        """

        base_prompt = DiffusersVideoRenderer._compile_prompt(request)
        character_constraints: list[dict[str, Any]] = []
        for character in request.characters:
            if not isinstance(character, dict):
                continue
            constraint: dict[str, Any] = {}
            character_id = character.get("character_uuid")
            if isinstance(character_id, str) and character_id.strip():
                constraint["character_uuid"] = character_id.strip()
            reference_ids = character.get("approved_reference_ids")
            if isinstance(reference_ids, (list, tuple)) and reference_ids:
                constraint["approved_reference_ids"] = list(reference_ids)
            invariants = character.get("identity_invariants")
            if isinstance(invariants, list) and invariants:
                constraint["identity_invariants"] = list(invariants)
            face_constraints = character.get("face_constraints")
            if isinstance(face_constraints, dict) and face_constraints:
                constraint["face_constraints"] = dict(face_constraints)
            body_constraints = character.get("body_constraints")
            if isinstance(body_constraints, dict) and body_constraints:
                constraint["body_constraints"] = dict(body_constraints)
            if constraint:
                character_constraints.append(constraint)

        structured: dict[str, Any] = {}
        if request.approved_reference_ids:
            structured["reference_board_order"] = list(request.approved_reference_ids)
        if character_constraints:
            structured["characters"] = character_constraints
        if request.continuity:
            structured["continuity"] = request.continuity

        if not structured:
            return base_prompt
        suffix = json.dumps(
            structured,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return f"{base_prompt}\nCINEOS production constraints (must preserve): {suffix}"


__all__ = [
    "MultiReferenceAdapter",
    "MultiReferenceConditioningResult",
    "ProductionDiffusersVideoRenderer",
]
