"""Compose independent production semantic scorers without relabelling ownership.

Identity/motion and difficult-case measurements intentionally come from separate
model families. This bridge lets the artifact observer consume one scorer while
preserving each component's provenance and rejecting metric collisions. External
pretrained components remain external; composition is CINEOS-owned orchestration,
not a claim that their weights are CINEOS-native.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifact_video_observer import RGBVideoSample

COMPOSITE_SEMANTIC_SCORER_SCHEMA = "cineos-composite-semantic-scorer/0.2"
_PRIMARY_METRICS = frozenset({"identity_similarity", "motion_quality"})
_OBSERVER_METRICS = frozenset({"artifact_integrity", "temporal_consistency"})


class CompositeSemanticScorerError(RuntimeError):
    """Raised when composed semantic evidence is incomplete or ambiguous."""


def _component_provenance(component: Any, *, role: str) -> dict[str, Any]:
    provider = getattr(component, "runtime_provenance", None)
    if provider is None or not callable(provider):
        raise CompositeSemanticScorerError(
            f"{role} semantic scorer must expose callable runtime_provenance()"
        )
    raw = provider()
    if not isinstance(raw, Mapping):
        raise CompositeSemanticScorerError(
            f"{role} semantic scorer provenance must be a mapping"
        )
    provenance = dict(raw)
    schema = provenance.get("schema")
    origin = provenance.get("origin")
    if not isinstance(schema, str) or not schema.strip():
        raise CompositeSemanticScorerError(
            f"{role} semantic scorer provenance requires a non-empty schema"
        )
    if not isinstance(origin, str) or not origin.strip():
        raise CompositeSemanticScorerError(
            f"{role} semantic scorer provenance requires a non-empty origin"
        )
    if provenance.get("production_measurement_evidence") is not True:
        raise CompositeSemanticScorerError(
            f"{role} semantic scorer provenance must attest production measurement evidence"
        )
    return provenance


def _declared_specialist_metrics(
    provenance: Mapping[str, Any], *, role: str
) -> frozenset[str]:
    """Return the exact semantic metric ownership attested by a specialist.

    Production specialists are not allowed to rely on orchestration inference for
    ownership. Their provenance must explicitly name every metric they can emit so
    later score evidence cannot be relabelled independently of the model/runtime
    attestation that produced it.
    """

    raw = provenance.get("measured_metrics")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise CompositeSemanticScorerError(
            f"{role} provenance requires a non-empty measured_metrics sequence"
        )
    metrics: list[str] = []
    for name in raw:
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise CompositeSemanticScorerError(
                f"{role} provenance contains an invalid measured metric name"
            )
        metrics.append(name)
    if len(set(metrics)) != len(metrics):
        raise CompositeSemanticScorerError(
            f"{role} provenance contains duplicate measured metric ownership"
        )
    forbidden = sorted((_PRIMARY_METRICS | _OBSERVER_METRICS).intersection(metrics))
    if forbidden:
        raise CompositeSemanticScorerError(
            f"{role} provenance cannot claim core/observer metric(s): "
            + ", ".join(forbidden)
        )
    return frozenset(metrics)


def _metric_mapping(raw: Any, *, role: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise CompositeSemanticScorerError(
            f"{role} semantic scorer must return a metric mapping"
        )
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise CompositeSemanticScorerError(
                f"{role} semantic scorer returned an invalid metric name"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompositeSemanticScorerError(
                f"{role} semantic metric {name!r} must be numeric"
            )
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise CompositeSemanticScorerError(
                f"{role} semantic metric {name!r} must be between 0 and 1"
            )
        metrics[name] = numeric
    return metrics


class CompositeSemanticVideoScorer:
    """Merge one primary identity/motion scorer with specialist semantic scorers.

    Every component must explicitly attest real production measurement evidence and
    publish runtime provenance. Specialist scorers must additionally declare exact
    metric ownership in ``measured_metrics``. They may add difficult-case metrics,
    but cannot replace identity/motion or observer-owned transport metrics. Duplicate
    ownership fails closed so benchmark evidence always has one unambiguous source.
    """

    semantic_measurement_evidence = True

    def __init__(self, primary: Any, specialists: Sequence[Any]) -> None:
        if not callable(primary):
            raise TypeError("primary semantic scorer must be callable")
        if getattr(primary, "semantic_measurement_evidence", False) is not True:
            raise TypeError("primary semantic scorer must attest measured evidence")
        if isinstance(specialists, (str, bytes)):
            raise TypeError("specialists must be a sequence of semantic scorers")
        self.primary = primary
        self.specialists = tuple(specialists)
        for index, specialist in enumerate(self.specialists):
            if not callable(specialist):
                raise TypeError(f"specialist semantic scorer {index} must be callable")
            if getattr(specialist, "semantic_measurement_evidence", False) is not True:
                raise TypeError(
                    f"specialist semantic scorer {index} must attest measured evidence"
                )

        # Validate provenance at construction so a production observer cannot be
        # created around an anonymous/mocked component and only fail after rendering.
        self._primary_provenance = _component_provenance(primary, role="primary")
        self._specialist_provenance = tuple(
            _component_provenance(component, role=f"specialist[{index}]")
            for index, component in enumerate(self.specialists)
        )
        self._specialist_metric_ownership = tuple(
            _declared_specialist_metrics(provenance, role=f"specialist[{index}]")
            for index, provenance in enumerate(self._specialist_provenance)
        )
        claimed: set[str] = set()
        for index, owned in enumerate(self._specialist_metric_ownership):
            duplicates = sorted(claimed.intersection(owned))
            if duplicates:
                raise CompositeSemanticScorerError(
                    f"specialist[{index}] provenance duplicates metric ownership: "
                    + ", ".join(duplicates)
                )
            claimed.update(owned)

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": COMPOSITE_SEMANTIC_SCORER_SCHEMA,
            "origin": "cineos_composed_measurement_pipeline",
            "production_measurement_evidence": True,
            "ownership": {
                "primary": sorted(_PRIMARY_METRICS),
                "specialists": [
                    {
                        "role": f"specialist[{index}]",
                        "measured_metrics": sorted(owned),
                    }
                    for index, owned in enumerate(self._specialist_metric_ownership)
                ],
            },
            "components": [
                {"role": "primary", "provenance": dict(self._primary_provenance)},
                *[
                    {
                        "role": f"specialist[{index}]",
                        "provenance": dict(provenance),
                    }
                    for index, provenance in enumerate(self._specialist_provenance)
                ],
            ],
        }

    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        primary = _metric_mapping(
            self.primary(
                sample,
                artifact=artifact,
                shot=shot,
                attempt_index=attempt_index,
            ),
            role="primary",
        )
        missing = sorted(_PRIMARY_METRICS - set(primary))
        if missing:
            raise CompositeSemanticScorerError(
                "primary semantic scorer missing required metric(s): "
                + ", ".join(missing)
            )
        illegal_primary = sorted(_OBSERVER_METRICS.intersection(primary))
        if illegal_primary:
            raise CompositeSemanticScorerError(
                "semantic scorer cannot claim observer-owned metric(s): "
                + ", ".join(illegal_primary)
            )

        merged = dict(primary)
        for index, specialist in enumerate(self.specialists):
            metrics = _metric_mapping(
                specialist(
                    sample,
                    artifact=artifact,
                    shot=shot,
                    attempt_index=attempt_index,
                ),
                role=f"specialist[{index}]",
            )
            forbidden = sorted(
                (_PRIMARY_METRICS | _OBSERVER_METRICS).intersection(metrics)
            )
            if forbidden:
                raise CompositeSemanticScorerError(
                    f"specialist[{index}] cannot replace core/observer metric(s): "
                    + ", ".join(forbidden)
                )
            undeclared = sorted(
                set(metrics) - self._specialist_metric_ownership[index]
            )
            if undeclared:
                raise CompositeSemanticScorerError(
                    f"specialist[{index}] emitted metric(s) not attested by provenance: "
                    + ", ".join(undeclared)
                )
            duplicates = sorted(set(merged).intersection(metrics))
            if duplicates:
                raise CompositeSemanticScorerError(
                    f"specialist[{index}] duplicates metric ownership: "
                    + ", ".join(duplicates)
                )
            merged.update(metrics)
        return merged


__all__ = [
    "COMPOSITE_SEMANTIC_SCORER_SCHEMA",
    "CompositeSemanticScorerError",
    "CompositeSemanticVideoScorer",
]
