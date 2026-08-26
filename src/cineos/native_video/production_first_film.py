"""Hardened native FIRST FILM composition for production CINEOS runs.

The lower-level film runner intentionally keeps quality gates optional for backwards
compatibility with legacy and lightweight test integrations. Production native video
must not rely on callers remembering each safety-critical option independently.
This module provides one explicit composition root that binds:

* the native temporal renderer to transactional continuity state,
* continuity checkpoint/retry hooks into ``FilmOrchestrator``,
* measured final-film picture/duration/edit QC,
* measured encoded-audio QC as a required acceptance criterion, and
* a versioned runtime manifest that is persisted with continuity state and checked
  before any long-running production job is resumed.

No external video generator is introduced here. FFmpeg/ffprobe remain inspectors and
container tools only through the existing gates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from cineos.film.first_film import FirstFilmRunner
from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelManifest,
    NativeModelRegistry,
    check_runtime_compatibility,
)

from .film_bridge import NativeFilmContinuityBridge, temporal_model_fingerprint
from .final_gate import MeasuredFinalFilmGate
from .renderer_binding import NativeFilmRendererBinding, NativeTemporalShotRenderer
from .runtime_manifest import (
    LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST,
    ProductionRuntimeManifest,
)

PRODUCTION_FIRST_FILM_RUNTIME_KIND = "cineos-production-first-film-runtime/0.1"


@dataclass(frozen=True, slots=True)
class ProductionFirstFilmRuntime:
    """Owned production composition with inspectable native dependencies.

    Keeping the continuity bridge and renderer binding alongside the public runner
    prevents their lifecycle from becoming implicit. Long-running callers may use
    ``continuity`` for diagnostics/checkpoint inspection while invoking ``runner``
    as the provider-neutral complete-film API. ``manifest`` records the production
    invariants required to decide whether a persisted job can safely resume after a
    software or model upgrade.
    """

    runner: FirstFilmRunner
    continuity: NativeFilmContinuityBridge
    renderer_binding: NativeFilmRendererBinding
    final_gate: MeasuredFinalFilmGate
    manifest: ProductionRuntimeManifest


def _stable_policy_value(value: Any) -> Any:
    """Return deterministic JSON-safe configuration for an acceptance component.

    Production resume compatibility must include more than the boolean fact that a
    quality gate exists. Threshold changes alter acceptance semantics and therefore
    must invalidate a checkpoint created under the old policy. This serializer is
    deliberately configuration-oriented: dataclass fields and ordinary public
    instance attributes are captured together with the fully-qualified component
    type, while callables and private caches are excluded.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _stable_policy_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_policy_value(item) for item in value]

    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type_name,
            "fields": {
                field.name: _stable_policy_value(getattr(value, field.name))
                for field in fields(value)
            },
        }

    try:
        attributes = vars(value)
    except TypeError:
        attributes = {}
    public = {
        name: _stable_policy_value(item)
        for name, item in sorted(attributes.items())
        if not name.startswith("_") and not callable(item)
    }
    return {"type": type_name, "attributes": public}


def final_gate_policy_fingerprint(gate: MeasuredFinalFilmGate) -> str:
    """Hash the complete configured final-film acceptance policy.

    The fingerprint is semantic configuration evidence, not a source-code hash. It
    catches threshold/evaluator/binary-policy changes across resume while remaining
    stable across process restarts and object identities.
    """
    payload = _stable_policy_value(gate)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _production_checkpoint_hooks(
    continuity: NativeFilmContinuityBridge,
    manifest: ProductionRuntimeManifest,
) -> dict[str, Any]:
    """Bind durable continuity state to the production runtime identity.

    ``FilmOrchestrator`` already integrity-hashes the complete runtime-state object.
    This envelope adds semantic compatibility: even an intact checkpoint is unsafe
    to resume when it was created by a different renderer/model or acceptance
    policy. Operational changes such as CPU/GPU placement and retry budget remain
    compatible according to :meth:`ProductionRuntimeManifest.assert_resume_compatible`.
    """

    def snapshot() -> dict[str, object]:
        return {
            "kind": PRODUCTION_FIRST_FILM_RUNTIME_KIND,
            "runtime_manifest": manifest.snapshot(),
            "continuity": continuity.snapshot(),
        }

    def restore(payload: dict[str, Any]) -> None:
        if str(payload.get("kind", "")) != PRODUCTION_FIRST_FILM_RUNTIME_KIND:
            raise ValueError("unsupported production FIRST FILM runtime checkpoint")
        raw_manifest = payload.get("runtime_manifest")
        if not isinstance(raw_manifest, dict):
            raise ValueError(
                "production runtime checkpoint is missing runtime_manifest"
            )
        raw_continuity = payload.get("continuity")
        if not isinstance(raw_continuity, dict):
            raise ValueError("production runtime checkpoint is missing continuity")

        saved_manifest = ProductionRuntimeManifest.restore(raw_manifest)
        manifest.assert_resume_compatible(saved_manifest)
        # Restore model state only after all production-level invariants pass. This
        # ordering guarantees an incompatible resume cannot mutate live continuity.
        continuity.restore(raw_continuity)

    return {
        "checkpoint_state_provider": snapshot,
        "checkpoint_state_restorer": restore,
        "checkpoint_state_resetter": continuity.reset,
        "shot_attempt_start": continuity.start_attempt,
        "shot_attempt_accepted": continuity.accept_attempt,
        "shot_attempt_rejected": continuity.reject_attempt,
    }


def _compatible_active_manifest(model_registry: NativeModelRegistry) -> NativeModelManifest:
    """Return the active release only when it is compatible with *this* runtime.

    Registry activation is compatibility-gated, but a persisted registry can outlive
    the runtime that activated it. After a software upgrade the runtime contract or
    supported component contracts may be narrower/different. Re-checking at the
    production composition root prevents a stale previously-approved model from
    silently entering a new film job.
    """
    active = model_registry.active()
    if active is None:
        raise ModelManifestError(
            "production FIRST FILM requires an active native model release"
        )
    compatibility = check_runtime_compatibility(
        active,
        runtime_contract_version=model_registry.runtime_contract_version,
        supported_component_contracts=model_registry.supported_component_contracts,
    )
    if not compatibility.compatible:
        raise ModelManifestError(
            "refusing incompatible active native model release: "
            + compatibility.reason
        )
    return active


def build_production_first_film_runtime(
    native_renderer: NativeTemporalShotRenderer,
    validator: Any | None = None,
    *,
    continuity: NativeFilmContinuityBridge | None = None,
    final_gate: MeasuredFinalFilmGate | None = None,
    renderer_id: str = "cineos-native",
    max_recovery_attempts: int = 2,
    device: str = "cpu",
    native_model_manifest_sha256: str = LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST,
) -> ProductionFirstFilmRuntime:
    """Build the fail-closed native CINEOS FIRST FILM production path.

    Production acceptance always requires post-assembly measured QC. When no gate
    is injected, the default gate also requires a real encoded audio stream. A
    custom gate is permitted for policy evolution/testing, but production callers
    cannot accidentally disable *final-film* evaluation itself because the runner
    is always constructed with ``require_final_film_evaluation=True``.

    Durable production checkpoints contain both continuity memory and this runtime's
    versioned manifest. Resume therefore fails before recurrent state is restored if
    renderer identity, temporal weights, released native-model manifest, final-gate
    policy, or final acceptance requirements changed.

    ``native_model_manifest_sha256`` defaults to an explicit legacy-unbound marker
    for backwards compatibility. New production deployments should use
    :func:`build_released_production_first_film_runtime`, which requires a compatible
    active entry in the native model registry and binds its digest automatically.
    """

    if max_recovery_attempts < 0:
        raise ValueError("max_recovery_attempts must be non-negative")
    if not renderer_id.strip():
        raise ValueError("renderer_id must not be empty")
    if not device.strip():
        raise ValueError("device must not be empty")
    if not native_model_manifest_sha256.strip():
        raise ValueError("native_model_manifest_sha256 must not be empty")

    active_continuity = continuity or NativeFilmContinuityBridge.default(device=device)
    if continuity is not None and continuity.device != device:
        raise ValueError(
            "explicit continuity device does not match requested production device"
        )

    active_gate = final_gate or MeasuredFinalFilmGate(require_audio=True)
    binding = NativeFilmRendererBinding(native_renderer, active_continuity)
    manifest = ProductionRuntimeManifest(
        renderer_id=renderer_id,
        temporal_model_fingerprint=temporal_model_fingerprint(active_continuity.model),
        device=device,
        max_recovery_attempts=max_recovery_attempts,
        require_final_film_evaluation=True,
        require_audio=active_gate.require_audio,
        final_gate_policy_fingerprint=final_gate_policy_fingerprint(active_gate),
        native_model_manifest_sha256=native_model_manifest_sha256,
    )
    runner = FirstFilmRunner(
        binding,
        validator,
        renderer_id=renderer_id,
        max_recovery_attempts=max_recovery_attempts,
        orchestrator_kwargs=_production_checkpoint_hooks(active_continuity, manifest),
        final_film_evaluator=active_gate,
        require_final_film_evaluation=True,
    )
    return ProductionFirstFilmRuntime(
        runner=runner,
        continuity=active_continuity,
        renderer_binding=binding,
        final_gate=active_gate,
        manifest=manifest,
    )


def build_released_production_first_film_runtime(
    native_renderer: NativeTemporalShotRenderer,
    model_registry: NativeModelRegistry,
    validator: Any | None = None,
    **kwargs: Any,
) -> ProductionFirstFilmRuntime:
    """Build production FIRST FILM bound to the registry's active model release.

    The active manifest is hash-verified by the registry and compatibility-checked
    again against the *current* runtime contract before composition. This matters
    after upgrades: a model that was valid when activated must not be assumed valid
    forever merely because its persisted registry entry remains active.

    Requiring that manifest here makes model release identity part of durable film
    state, so a checkpoint cannot silently resume after any released component has
    changed even when the temporal sub-model itself happens to be identical.
    """
    active = _compatible_active_manifest(model_registry)
    requested_digest = kwargs.pop("native_model_manifest_sha256", None)
    if requested_digest is not None and requested_digest != active.manifest_sha256:
        raise ModelManifestError(
            "explicit native model manifest does not match active registry release"
        )
    return build_production_first_film_runtime(
        native_renderer,
        validator,
        native_model_manifest_sha256=active.manifest_sha256,
        **kwargs,
    )


__all__ = [
    "PRODUCTION_FIRST_FILM_RUNTIME_KIND",
    "ProductionFirstFilmRuntime",
    "build_production_first_film_runtime",
    "build_released_production_first_film_runtime",
    "final_gate_policy_fingerprint",
]
