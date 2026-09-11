"""Production-safe Diffusers boundary for CINEOS reference-conditioned shots.

This module strengthens the execution contract around external pretrained video
foundations. It does not make the external checkpoint CINEOS-native. Instead,
it prevents a production shot that declares approved visual references from
silently degrading to text-only or partial-reference generation when those
references cannot actually reach the foundation pipeline.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diffusers_video import (
    DiffusersVideoError,
    DiffusersVideoRenderer,
    DiffusersVideoResult,
)
from .native_request import NativeShotRequest


@dataclass(frozen=True, slots=True)
class MultiReferenceConditioningResult:
    """Auditable output from a CINEOS-approved multi-reference adapter.

    ``consumed_reference_ids`` must exactly match the approved IDs on the shot.
    For multi-character shots, ``consumed_character_reference_ids`` additionally
    binds each identity slot to its exact approved references so an adapter cannot
    silently swap character identities while still consuming the correct global set.
    """

    image: Any
    consumed_reference_ids: tuple[str, ...]
    adapter_id: str
    adapter_version: str
    consumed_character_reference_ids: tuple[tuple[str, tuple[str, ...]], ...] | None = (
        None
    )


@dataclass(frozen=True, slots=True)
class ProductionDiffusersVideoResult(DiffusersVideoResult):
    """Production render plus the exact CINEOS conditioning path used.

    ``conditioning_provenance`` is intentionally separate from foundation
    provenance. It records CINEOS-owned orchestration/conditioning without
    implying that the external pretrained foundation itself is native CINEOS.
    """

    conditioning_provenance: dict[str, Any] | None = None


MultiReferenceAdapter = Callable[
    [NativeShotRequest, Sequence[Any]], MultiReferenceConditioningResult
]


def _conditioning_content_sha256(value: Any) -> str:
    """Fingerprint the exact decoded conditioning content sent toward inference.

    The fingerprint is intentionally content-oriented rather than tied to an asset
    filename. Production loaders commonly return PIL images, numpy arrays or torch
    tensors after decoding; hashing that consumed representation catches an asset
    replacement even when its logical reference ID and source path stay unchanged.
    Simple bytes/paths are supported for lower-level loaders and strings remain
    supported for lightweight test/dry-run boundaries.
    """

    header: dict[str, Any]
    payload: bytes

    if isinstance(value, bytes):
        header = {"kind": "bytes"}
        payload = value
    elif isinstance(value, bytearray):
        header = {"kind": "bytes"}
        payload = bytes(value)
    elif isinstance(value, memoryview):
        header = {"kind": "bytes"}
        payload = value.tobytes()
    elif isinstance(value, Path):
        if not value.is_file():
            raise DiffusersVideoError(
                f"conditioning reference path does not exist: {value}"
            )
        header = {"kind": "file-bytes"}
        payload = value.read_bytes()
    elif isinstance(value, str):
        path = Path(value)
        if path.is_file():
            header = {"kind": "file-bytes"}
            payload = path.read_bytes()
        else:
            header = {"kind": "string"}
            payload = value.encode("utf-8")
    else:
        candidate = value
        if all(hasattr(candidate, attr) for attr in ("detach", "cpu")):
            try:
                candidate = candidate.detach().cpu()
                if hasattr(candidate, "contiguous"):
                    candidate = candidate.contiguous()
                if hasattr(candidate, "numpy"):
                    candidate = candidate.numpy()
            except Exception as exc:  # pragma: no cover - backend-specific guard
                raise DiffusersVideoError(
                    "conditioning tensor could not be normalized for fingerprinting"
                ) from exc

        tobytes = getattr(candidate, "tobytes", None)
        if not callable(tobytes):
            raise DiffusersVideoError(
                "production conditioning content cannot be deterministically "
                f"fingerprinted: {type(value).__name__}"
            )
        try:
            payload = tobytes()
        except Exception as exc:  # pragma: no cover - backend-specific guard
            raise DiffusersVideoError(
                "production conditioning content could not be serialized for "
                "fingerprinting"
            ) from exc
        if not isinstance(payload, bytes):
            payload = bytes(payload)

        shape = getattr(candidate, "shape", None)
        size = getattr(candidate, "size", None)
        if callable(size):
            size = None
        header = {
            "kind": "decoded-array",
            "type": type(candidate).__name__,
            "dtype": str(getattr(candidate, "dtype", "")),
            "mode": str(getattr(candidate, "mode", "")),
            "shape": list(shape) if shape is not None else None,
            "size": list(size) if isinstance(size, tuple) else size,
        }

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    digest.update(b"\x00")
    digest.update(payload)
    return digest.hexdigest()


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

    Production prompt compilation also appends compact CINEOS-owned identity,
    continuity, performance, interaction, environment and camera constraints to
    any director-authored prompt. This prevents an explicit prompt from accidentally
    suppressing structured CineDNA or choreography before external-foundation
    inference.
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
        self._conditioning_provenance: dict[str, Any] | None = None

    def render(self, request: Any) -> ProductionDiffusersVideoResult:
        self._prepared_multi_reference_image = None
        self._conditioning_provenance = None
        if isinstance(request, NativeShotRequest):
            if self._foundation_requires_image_conditioning():
                self._verify_mandatory_image_conditioning_path()
            if request.approved_reference_ids:
                self._verify_reference_conditioning_path(request)
        try:
            result = super().render(request)
            return ProductionDiffusersVideoResult(
                shot_id=result.shot_id,
                scene_id=result.scene_id,
                output_path=result.output_path,
                frame_count=result.frame_count,
                seed=result.seed,
                foundation=result.foundation,
                request_hash=result.request_hash,
                artifact_sha256=result.artifact_sha256,
                artifact_size_bytes=result.artifact_size_bytes,
                conditioning_provenance=(
                    dict(self._conditioning_provenance)
                    if self._conditioning_provenance is not None
                    else None
                ),
            )
        finally:
            self._prepared_multi_reference_image = None
            self._conditioning_provenance = None

    def _foundation_requires_image_conditioning(self) -> bool:
        """Return whether the declared production capability contract is I2V-only."""

        features = frozenset(self.capabilities.supported_features)
        return "image_to_video" in features and "text_to_video" not in features

    def _verify_mandatory_image_conditioning_path(self) -> None:
        """Reject an I2V-only foundation that cannot receive an image at inference."""

        if self._pipeline is None:
            raise DiffusersVideoError("renderer model is not loaded")
        parameters = inspect.signature(self._pipeline.__call__).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "image" not in parameters and not accepts_kwargs:
            raise DiffusersVideoError(
                "production foundation is image-to-video only, but the loaded "
                "pipeline does not expose image conditioning"
            )

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
        else:
            self._conditioning_provenance = {
                "mode": "single_reference",
                "consumed_reference_ids": list(request.approved_reference_ids),
            }

    @staticmethod
    def _validate_character_reference_lineage(request: NativeShotRequest) -> None:
        """Reject ambiguous or escaped character reference ownership in production."""

        approved = set(request.approved_reference_ids)
        multi_character = len(request.characters) > 1
        reference_owner: dict[str, str] = {}
        character_ids: set[str] = set()
        for index, character in enumerate(request.characters):
            if not isinstance(character, dict):
                raise DiffusersVideoError(
                    f"production character conditioning {index} must be an object"
                )
            character_id = character.get("character_uuid", f"index:{index}")
            if not isinstance(character_id, str) or not character_id.strip():
                character_id = f"index:{index}"
            else:
                character_id = character_id.strip()
            if multi_character:
                if character_id.startswith("index:"):
                    raise DiffusersVideoError(
                        "multi-character production conditioning requires a non-empty "
                        "character_uuid for every character"
                    )
                if character_id in character_ids:
                    raise DiffusersVideoError(
                        "multi-character production conditioning requires unique "
                        f"character_uuid values: duplicate {character_id!r}"
                    )
                character_ids.add(character_id)
            raw_ids = character.get("approved_reference_ids", [])
            if not isinstance(raw_ids, (list, tuple)) or any(
                not isinstance(reference_id, str) or not reference_id.strip()
                for reference_id in raw_ids
            ):
                raise DiffusersVideoError(
                    "character approved_reference_ids must be a sequence of non-empty "
                    "strings"
                )
            if multi_character and not raw_ids:
                raise DiffusersVideoError(
                    "multi-character production conditioning requires at least one "
                    f"approved reference per character: {character_id!r} has none"
                )
            escaped = [
                reference_id for reference_id in raw_ids if reference_id not in approved
            ]
            if escaped:
                raise DiffusersVideoError(
                    "character conditioning references are not approved by the shot: "
                    f"{character_id!r} -> {', '.join(escaped)}"
                )
            for reference_id in raw_ids:
                previous_owner = reference_owner.get(reference_id)
                if previous_owner is not None and previous_owner != character_id:
                    raise DiffusersVideoError(
                        "character identity reference is ambiguously assigned to "
                        "multiple characters: "
                        f"{reference_id!r} -> {previous_owner!r}, {character_id!r}"
                    )
                reference_owner[reference_id] = character_id

    @staticmethod
    def _expected_character_reference_bindings(
        request: NativeShotRequest,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the canonical character-to-reference ownership contract."""

        bindings: list[tuple[str, tuple[str, ...]]] = []
        for index, character in enumerate(request.characters):
            if not isinstance(character, dict):
                continue
            character_id = character.get("character_uuid", f"index:{index}")
            if not isinstance(character_id, str) or not character_id.strip():
                character_id = f"index:{index}"
            else:
                character_id = character_id.strip()
            raw_ids = character.get("approved_reference_ids", [])
            if isinstance(raw_ids, (list, tuple)) and raw_ids:
                bindings.append((character_id, tuple(raw_ids)))
        return tuple(bindings)

    def _prepare_multi_reference_image(self, request: NativeShotRequest) -> Any:
        if self.multi_reference_adapter is None:
            raise DiffusersVideoError(
                "production Diffusers boundary cannot safely consume multiple "
                "approved_reference_ids through its single image-conditioning slot; "
                "configure an audited multi_reference_adapter"
            )
        assert self.reference_loader is not None

        resolved: list[Any] = []
        resolved_sha256: list[str] = []
        for reference_id in request.approved_reference_ids:
            reference = self.reference_loader(reference_id)
            if reference is None:
                raise DiffusersVideoError(
                    "approved identity reference could not be resolved for production "
                    f"shot {request.shot_id!r}: {reference_id!r}"
                )
            resolved.append(reference)
            resolved_sha256.append(_conditioning_content_sha256(reference))

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
        expected_bindings = self._expected_character_reference_bindings(request)
        reported_bindings = result.consumed_character_reference_ids
        if (
            len(request.characters) > 1
            and expected_bindings
            and reported_bindings is None
        ):
            raise DiffusersVideoError(
                "multi-character multi_reference_adapter must attest exact "
                "character_uuid-to-reference ownership"
            )
        if reported_bindings is not None and reported_bindings != expected_bindings:
            raise DiffusersVideoError(
                "multi_reference_adapter character identity binding does not match "
                "the approved CINEOS character-to-reference ownership contract"
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
        self._conditioning_provenance = {
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": list(result.consumed_reference_ids),
            "consumed_reference_sha256": resolved_sha256,
            "conditioning_image_sha256": _conditioning_content_sha256(result.image),
            "adapter_id": result.adapter_id.strip(),
            "adapter_version": result.adapter_version.strip(),
        }
        if reported_bindings is not None:
            self._conditioning_provenance["consumed_character_reference_ids"] = [
                {
                    "character_uuid": character_id,
                    "reference_ids": list(reference_ids),
                }
                for character_id, reference_ids in reported_bindings
            ]
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
        if reference is None and self._foundation_requires_image_conditioning():
            raise DiffusersVideoError(
                "production foundation is image-to-video only, but this shot has no "
                "resolved image conditioning source"
            )
        if reference is not None and self._conditioning_provenance is not None:
            reference_sha256 = _conditioning_content_sha256(reference)
            self._conditioning_provenance["consumed_reference_sha256"] = [
                reference_sha256
            ]
            self._conditioning_provenance["conditioning_image_sha256"] = (
                reference_sha256
            )
        return reference

    @staticmethod
    def _compile_prompt(request: NativeShotRequest) -> str:
        """Preserve director prompt while injecting structured production constraints.

        The base renderer historically returned ``metadata['prompt']`` verbatim when
        present. For production that can discard structured character identity,
        continuity, performance and object-interaction information exactly when a
        high-quality hand-authored prompt is supplied. We retain that prompt, then
        append deterministic compact JSON containing the CINEOS-native constraints
        that materially affect connected-shot quality.
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
        if request.environment:
            structured["environment"] = request.environment
        if request.wardrobe:
            structured["wardrobe"] = request.wardrobe
        if request.props:
            structured["props"] = request.props
        if request.continuity:
            structured["continuity"] = request.continuity
        if request.performance:
            structured["performance"] = request.performance

        camera_constraints = {
            key: request.camera[key]
            for key in ("shot_size", "movement", "lens")
            if key in request.camera and request.camera[key] not in (None, "", [], {})
        }
        if camera_constraints:
            structured["camera"] = camera_constraints

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
    "ProductionDiffusersVideoResult",
]
