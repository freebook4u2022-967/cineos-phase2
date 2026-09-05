"""Pinned external SigLIP2 feature scorer for production video QC.

This is a CINEOS-owned measurement adapter around an explicitly external,
Apache-2.0 pretrained Google SigLIP2 vision encoder. It is not a CINEOS-native
model and does not claim to measure anatomy, physics, or action correctness.
It provides learned visual identity similarity and a conservative feature-space
temporal-coherence proxy for the core ``motion_quality`` gate. Stronger specialist
QC can replace this backend without changing the artifact-bound evidence contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from .artifact_video_observer import RGBVideoSample
from .production_references import ProductionReferenceLoader

SIGLIP2_QC_MODEL_ID = "google/siglip2-base-patch16-256"
SIGLIP2_QC_REVISION = "ce3bda6b1094ecd25dabd523e58ddab69b83baf2"
SIGLIP2_QC_LICENSE = "Apache-2.0"
SIGLIP2_QC_SCHEMA = "cineos-external-siglip2-video-qc/0.7"
DEFAULT_MOTION_ACTIVITY_FLOOR = 1e-4
DEFAULT_MOTION_SUPPORT_FRACTION = 0.25
DEFAULT_MOTION_STEP_CEILING = 0.5


class SigLIP2VideoScorerError(RuntimeError):
    """Raised when pinned learned QC cannot produce trustworthy measurements."""


def _normalize(vector: Sequence[float], *, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise SigLIP2VideoScorerError(f"{label} is not a numeric embedding") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise SigLIP2VideoScorerError(f"{label} is empty or non-finite")
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude <= 0:
        raise SigLIP2VideoScorerError(f"{label} has zero magnitude")
    return tuple(value / magnitude for value in values)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise SigLIP2VideoScorerError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(1.0, (_cosine(left, right) + 1.0) / 2.0))


def _top_fraction_mean(scores: Sequence[float], *, fraction: float) -> float:
    """Return mean support over the strongest required fraction of observations."""
    if not scores:
        raise SigLIP2VideoScorerError("identity support requires frame scores")
    if not 0.0 < fraction <= 1.0:
        raise SigLIP2VideoScorerError(
            "multi-identity support fraction must be greater than 0 and at most 1"
        )
    support_count = max(1, math.ceil(len(scores) * fraction))
    strongest = sorted((float(score) for score in scores), reverse=True)[:support_count]
    if any(not math.isfinite(score) for score in strongest):
        raise SigLIP2VideoScorerError("identity support scores must be finite")
    return sum(strongest) / len(strongest)


def _identity_score(
    frame_features: Sequence[Sequence[float]],
    references: Sequence[Sequence[float]],
    *,
    mean_weight: float,
    multi_identity_support_fraction: float = 0.25,
) -> float:
    """Aggregate identity similarity without hiding sparse approved identities."""
    if not frame_features:
        raise SigLIP2VideoScorerError("identity QC requires decoded frame features")
    if not references:
        raise SigLIP2VideoScorerError("identity QC requires approved references")
    frame_scores = [
        max(_similarity(frame, reference) for reference in references)
        for frame in frame_features
    ]
    mean_score = sum(frame_scores) / len(frame_scores)
    worst_score = min(frame_scores)
    aggregate = mean_weight * mean_score + (1.0 - mean_weight) * worst_score
    if len(references) > 1:
        weakest_reference_support = min(
            _top_fraction_mean(
                [_similarity(frame, reference) for frame in frame_features],
                fraction=multi_identity_support_fraction,
            )
            for reference in references
        )
        aggregate = min(aggregate, weakest_reference_support)
    return max(0.0, min(1.0, aggregate))


def _rows(value: Any) -> list[list[float]]:
    """Convert a torch-like 2D feature tensor to ordinary numeric rows."""
    candidate = value
    for method in ("detach", "float", "cpu"):
        operation = getattr(candidate, method, None)
        if callable(operation):
            candidate = operation()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        raise SigLIP2VideoScorerError("vision encoder returned non-sequence features")
    rows: list[list[float]] = []
    for row in candidate:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise SigLIP2VideoScorerError(
                "vision encoder features must be two-dimensional"
            )
        rows.append([float(item) for item in row])
    if not rows:
        raise SigLIP2VideoScorerError("vision encoder returned no features")
    return rows


class SigLIP2FeatureVideoScorer:
    """Measure identity and feature-space motion coherence on decoded video frames."""

    def __init__(
        self,
        reference_loader: ProductionReferenceLoader,
        *,
        device: str = "cuda",
        model: Any | None = None,
        processor: Any | None = None,
        torch_module: Any | None = None,
        identity_mean_weight: float = 0.7,
        multi_identity_support_fraction: float = 0.25,
        motion_activity_floor: float = DEFAULT_MOTION_ACTIVITY_FLOOR,
        motion_support_fraction: float = DEFAULT_MOTION_SUPPORT_FRACTION,
        motion_step_ceiling: float = DEFAULT_MOTION_STEP_CEILING,
    ) -> None:
        if not isinstance(reference_loader, ProductionReferenceLoader):
            raise TypeError("reference_loader must be ProductionReferenceLoader")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be non-empty")
        if not 0.0 <= identity_mean_weight <= 1.0:
            raise ValueError("identity_mean_weight must be between 0 and 1")
        if not 0.0 < multi_identity_support_fraction <= 1.0:
            raise ValueError(
                "multi_identity_support_fraction must be greater than 0 and at most 1"
            )
        self._validate_motion_policy(
            activity_floor=motion_activity_floor,
            support_fraction=motion_support_fraction,
            step_ceiling=motion_step_ceiling,
        )
        injected = any(value is not None for value in (model, processor, torch_module))
        try:
            torch = torch_module or import_module("torch")
            transformers = (
                None
                if model is not None and processor is not None
                else import_module("transformers")
            )
        except ImportError as exc:
            raise SigLIP2VideoScorerError(
                "SigLIP2 production QC requires torch and transformers from the video extra"
            ) from exc
        if model is None:
            auto_model = getattr(transformers, "AutoModel", None)
            if auto_model is None:
                raise SigLIP2VideoScorerError("transformers.AutoModel is unavailable")
            model = auto_model.from_pretrained(
                SIGLIP2_QC_MODEL_ID,
                revision=SIGLIP2_QC_REVISION,
                local_files_only=True,
            )
        if processor is None:
            auto_processor = getattr(transformers, "AutoProcessor", None)
            if auto_processor is None:
                raise SigLIP2VideoScorerError(
                    "transformers.AutoProcessor is unavailable"
                )
            processor = auto_processor.from_pretrained(
                SIGLIP2_QC_MODEL_ID,
                revision=SIGLIP2_QC_REVISION,
                local_files_only=True,
            )
        to_device = getattr(model, "to", None)
        eval_mode = getattr(model, "eval", None)
        if not callable(to_device) or not callable(eval_mode):
            raise SigLIP2VideoScorerError(
                "SigLIP2 model lacks required inference methods"
            )
        self.model = to_device(device)
        self.model.eval()
        self.processor = processor
        self.torch = torch
        self.device = device.strip()
        self.reference_loader = reference_loader
        self.identity_mean_weight = float(identity_mean_weight)
        self.multi_identity_support_fraction = float(multi_identity_support_fraction)
        self.motion_activity_floor = float(motion_activity_floor)
        self.motion_support_fraction = float(motion_support_fraction)
        self.motion_step_ceiling = float(motion_step_ceiling)
        self.semantic_measurement_evidence = not injected
        self._reference_cache: dict[str, tuple[float, ...]] = {}

    @staticmethod
    def _validate_motion_policy(
        *, activity_floor: float, support_fraction: float, step_ceiling: float
    ) -> None:
        if not math.isfinite(activity_floor) or not 0.0 < activity_floor <= 1.0:
            raise ValueError(
                "activity_floor must be finite, greater than 0, and at most 1"
            )
        if not math.isfinite(support_fraction) or not 0.0 < support_fraction <= 1.0:
            raise ValueError(
                "support_fraction must be finite, greater than 0, and at most 1"
            )
        if not math.isfinite(step_ceiling) or not 0.0 < step_ceiling <= 1.0:
            raise ValueError(
                "step_ceiling must be finite, greater than 0, and at most 1"
            )
        if step_ceiling <= activity_floor:
            raise ValueError("step_ceiling must be greater than activity_floor")

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": SIGLIP2_QC_SCHEMA,
            "origin": "external-pretrained-foundation",
            "model_id": SIGLIP2_QC_MODEL_ID,
            "revision": SIGLIP2_QC_REVISION,
            "license": SIGLIP2_QC_LICENSE,
            "identity_metric": "approved-reference-temporal-support-cosine",
            "multi_identity_support_fraction": self.multi_identity_support_fraction,
            "motion_metric": "siglip2-feature-step-coherence-with-cut-rejection",
            "motion_activity_floor": self.motion_activity_floor,
            "motion_support_fraction": self.motion_support_fraction,
            "motion_step_ceiling": self.motion_step_ceiling,
            "production_measurement_evidence": self.semantic_measurement_evidence,
            "reference_manifest_sha256": self.reference_loader.manifest_sha256,
        }

    def _pil_frames(self, sample: RGBVideoSample) -> list[Any]:
        try:
            image_module = import_module("PIL.Image")
        except ImportError as exc:
            raise SigLIP2VideoScorerError("SigLIP2 QC requires Pillow") from exc
        return [
            image_module.frombytes("RGB", (sample.width, sample.height), frame)
            for frame in sample.frames
        ]

    def _encode_images(self, images: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
        if not images:
            raise SigLIP2VideoScorerError("cannot encode an empty image batch")
        processed = self.processor(images=list(images), return_tensors="pt")
        if not isinstance(processed, Mapping):
            raise SigLIP2VideoScorerError("SigLIP2 processor returned invalid inputs")
        inputs: dict[str, Any] = {}
        for name, value in processed.items():
            move = getattr(value, "to", None)
            inputs[name] = move(self.device) if callable(move) else value
        no_grad = getattr(self.torch, "inference_mode", None)
        if not callable(no_grad):
            no_grad = getattr(self.torch, "no_grad", None)
        if not callable(no_grad):
            raise SigLIP2VideoScorerError("torch inference context is unavailable")
        with no_grad():
            features = self.model.get_image_features(**inputs)
        return tuple(
            _normalize(row, label=f"SigLIP2 image feature {index}")
            for index, row in enumerate(_rows(features))
        )

    def _reference_embedding(self, reference_id: str) -> tuple[float, ...]:
        cached = self._reference_cache.get(reference_id)
        if cached is not None:
            return cached
        image = self.reference_loader(reference_id)
        embedding = self._encode_images([image])[0]
        self._reference_cache[reference_id] = embedding
        return embedding

    @staticmethod
    def _motion_coherence(
        features: Sequence[Sequence[float]],
        *,
        activity_floor: float = DEFAULT_MOTION_ACTIVITY_FLOOR,
        support_fraction: float = DEFAULT_MOTION_SUPPORT_FRACTION,
        step_ceiling: float = DEFAULT_MOTION_STEP_CEILING,
    ) -> float:
        """Score sustained motion while rejecting freeze, jitter, and cut-like jumps."""
        if len(features) < 2:
            return 0.0
        SigLIP2FeatureVideoScorer._validate_motion_policy(
            activity_floor=activity_floor,
            support_fraction=support_fraction,
            step_ceiling=step_ceiling,
        )
        steps = [
            max(0.0, min(1.0, 1.0 - _cosine(previous, current)))
            for previous, current in zip(features, features[1:])
        ]
        if any(step > step_ceiling for step in steps):
            return 0.0
        active_count = sum(step >= activity_floor for step in steps)
        required_count = max(1, math.ceil(len(steps) * support_fraction))
        if active_count < required_count:
            return 0.0
        active_fraction = active_count / len(steps)
        if len(steps) == 1:
            return active_fraction
        accelerations = [abs(left - right) for left, right in zip(steps, steps[1:])]
        coherence = max(
            0.0,
            min(1.0, 1.0 - sum(accelerations) / len(accelerations)),
        )
        return min(coherence, active_fraction)

    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        del artifact, attempt_index
        reference_ids = getattr(shot, "approved_reference_ids", None)
        if not isinstance(reference_ids, list) or not reference_ids:
            raise SigLIP2VideoScorerError(
                "SigLIP2 identity QC requires approved_reference_ids"
            )
        self.reference_loader.validate_reference_ids(reference_ids)
        references = [self._reference_embedding(item) for item in reference_ids]
        frame_features = self._encode_images(self._pil_frames(sample))
        identity = _identity_score(
            frame_features,
            references,
            mean_weight=self.identity_mean_weight,
            multi_identity_support_fraction=self.multi_identity_support_fraction,
        )
        return {
            "identity_similarity": identity,
            "motion_quality": self._motion_coherence(
                frame_features,
                activity_floor=self.motion_activity_floor,
                support_fraction=self.motion_support_fraction,
                step_ceiling=self.motion_step_ceiling,
            ),
        }


__all__ = [
    "DEFAULT_MOTION_ACTIVITY_FLOOR",
    "DEFAULT_MOTION_STEP_CEILING",
    "DEFAULT_MOTION_SUPPORT_FRACTION",
    "SIGLIP2_QC_LICENSE",
    "SIGLIP2_QC_MODEL_ID",
    "SIGLIP2_QC_REVISION",
    "SIGLIP2_QC_SCHEMA",
    "SigLIP2FeatureVideoScorer",
    "SigLIP2VideoScorerError",
]
