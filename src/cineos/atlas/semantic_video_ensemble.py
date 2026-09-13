"""Auditable composition of production semantic video QC scorers.

CINEOS needs specialist evidence for difficult cases such as anatomy, object
interaction, locomotion, physics, and dialogue lip-sync.  A single generic visual
encoder must not be relabelled as proof of those capabilities.  This module therefore
composes independently measured semantic scorers while preserving explicit metric
ownership and runtime provenance for every component.

The ensemble is a CINEOS-owned orchestration boundary, not a CINEOS-native model.
External pretrained scorers remain external in the emitted provenance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_video_observer import RGBVideoSample

SEMANTIC_SCORER_ENSEMBLE_SCHEMA = "cineos-semantic-video-scorer-ensemble/0.1"


class SemanticScorerEnsembleError(RuntimeError):
    """Raised when specialist semantic evidence is ambiguous or unauditable."""


@dataclass(frozen=True, slots=True)
class SemanticScorerComponent:
    """Bind one scorer to the exact metrics it is permitted to produce."""

    name: str
    scorer: Any
    measured_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("semantic scorer component name must be non-empty")
        if not callable(self.scorer):
            raise TypeError("semantic scorer component scorer must be callable")
        if not isinstance(self.measured_metrics, tuple) or not self.measured_metrics:
            raise ValueError(
                "semantic scorer component measured_metrics must be non-empty"
            )
        normalized: list[str] = []
        for metric in self.measured_metrics:
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError(
                    "semantic scorer metric names must be non-empty strings"
                )
            normalized.append(metric.strip())
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "semantic scorer component cannot declare duplicate metrics"
            )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "measured_metrics", tuple(normalized))


class ProductionSemanticScorerEnsemble:
    """Compose disjoint specialist measurements without erasing provenance.

    Every component owns an explicit metric set.  At inference time it must return
    exactly that set, preventing an identity model from opportunistically emitting an
    anatomy or lip-sync score that was never declared.  Production attestation is true
    only when every child scorer explicitly attests measured semantic evidence.
    """

    def __init__(self, components: Sequence[SemanticScorerComponent]) -> None:
        if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
            raise TypeError("components must be a sequence of SemanticScorerComponent")
        materialized = tuple(components)
        if not materialized:
            raise ValueError("semantic scorer ensemble requires at least one component")
        if any(not isinstance(item, SemanticScorerComponent) for item in materialized):
            raise TypeError(
                "components must contain only SemanticScorerComponent values"
            )

        names = [item.name for item in materialized]
        if len(set(names)) != len(names):
            raise ValueError("semantic scorer ensemble component names must be unique")

        owners: dict[str, str] = {}
        for component in materialized:
            for metric in component.measured_metrics:
                previous = owners.get(metric)
                if previous is not None:
                    raise ValueError(
                        f"semantic metric {metric!r} is owned by both {previous!r} "
                        f"and {component.name!r}"
                    )
                owners[metric] = component.name

        self.components = materialized
        self.metric_owners = dict(owners)
        self.semantic_measurement_evidence = all(
            getattr(item.scorer, "semantic_measurement_evidence", False) is True
            for item in materialized
        )

    @staticmethod
    def _component_provenance(component: SemanticScorerComponent) -> dict[str, Any]:
        provider = getattr(component.scorer, "runtime_provenance", None)
        if not callable(provider):
            raise SemanticScorerEnsembleError(
                f"semantic scorer component {component.name!r} requires callable "
                "runtime_provenance"
            )
        raw = provider()
        if not isinstance(raw, Mapping):
            raise SemanticScorerEnsembleError(
                f"semantic scorer component {component.name!r} provenance must be a mapping"
            )
        provenance = dict(raw)
        schema = provenance.get("schema")
        origin = provenance.get("origin")
        if not isinstance(schema, str) or not schema.strip():
            raise SemanticScorerEnsembleError(
                f"semantic scorer component {component.name!r} provenance requires schema"
            )
        if not isinstance(origin, str) or not origin.strip():
            raise SemanticScorerEnsembleError(
                f"semantic scorer component {component.name!r} provenance requires origin"
            )
        declared_attestation = provenance.get("production_measurement_evidence")
        scorer_attestation = (
            getattr(component.scorer, "semantic_measurement_evidence", False) is True
        )
        if declared_attestation is not scorer_attestation:
            raise SemanticScorerEnsembleError(
                f"semantic scorer component {component.name!r} provenance attestation "
                "does not match runtime scorer attestation"
            )
        return provenance

    def runtime_provenance(self) -> dict[str, Any]:
        component_evidence = []
        for component in self.components:
            component_evidence.append(
                {
                    "name": component.name,
                    "measured_metrics": list(component.measured_metrics),
                    "scorer": self._component_provenance(component),
                }
            )
        return {
            "schema": SEMANTIC_SCORER_ENSEMBLE_SCHEMA,
            "origin": "cineos-composition-of-declared-semantic-scorers",
            "production_measurement_evidence": self.semantic_measurement_evidence,
            "metric_owners": dict(self.metric_owners),
            "components": component_evidence,
        }

    @staticmethod
    def _validated_component_metrics(
        component: SemanticScorerComponent,
        raw: Any,
    ) -> dict[str, float]:
        if not isinstance(raw, Mapping):
            raise SemanticScorerEnsembleError(
                f"semantic scorer component {component.name!r} must return a mapping"
            )
        actual = set(raw)
        expected = set(component.measured_metrics)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("undeclared=" + ",".join(extra))
            raise SemanticScorerEnsembleError(
                f"semantic scorer component {component.name!r} violated metric ownership: "
                + "; ".join(details)
            )

        normalized: dict[str, float] = {}
        for metric in component.measured_metrics:
            value = raw[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SemanticScorerEnsembleError(
                    f"semantic metric {metric!r} from {component.name!r} must be numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise SemanticScorerEnsembleError(
                    f"semantic metric {metric!r} from {component.name!r} must be finite "
                    "and between 0 and 1"
                )
            normalized[metric] = numeric
        return normalized

    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for component in self.components:
            raw = component.scorer(
                sample,
                artifact=artifact,
                shot=shot,
                attempt_index=attempt_index,
            )
            measured = self._validated_component_metrics(component, raw)
            # Constructor validation guarantees disjoint ownership; keep the runtime
            # assertion as a corruption guard if state is ever deserialized/mutated.
            collisions = set(metrics).intersection(measured)
            if collisions:
                raise SemanticScorerEnsembleError(
                    "semantic scorer ensemble metric collision: "
                    + ", ".join(sorted(collisions))
                )
            metrics.update(measured)
        return metrics


__all__ = [
    "ProductionSemanticScorerEnsemble",
    "SEMANTIC_SCORER_ENSEMBLE_SCHEMA",
    "SemanticScorerComponent",
    "SemanticScorerEnsembleError",
]
