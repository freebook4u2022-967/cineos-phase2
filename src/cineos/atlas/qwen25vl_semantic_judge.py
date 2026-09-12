"""Pinned external-pretrained multimodal judge for difficult-case video QC.

This module provides a real model-execution path for visual semantic measurements
that cannot be derived honestly from the core SigLIP2 identity/motion scorer.  It
uses the Apache-2.0 Qwen2.5-VL-7B-Instruct foundation at an immutable revision and
judges a bounded sequence of frames decoded from the rendered artifact.

The scorer is deliberately labelled as external pretrained capability.  CINEOS owns
the sampling, rubric, schema validation, provenance and reject/rerender integration;
it does not claim Qwen weights as a CINEOS-native model.  Dialogue lip-sync is
excluded because sampled visual frames alone cannot prove audio/visual synchrony.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifact_video_observer import RGBVideoSample

QWEN25VL_SEMANTIC_JUDGE_SCHEMA = "cineos-qwen25vl-semantic-judge/0.1"
QWEN25VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN25VL_MODEL_REVISION = "b901af65fa3b2801b73d1c5b1ff59b89d81a708f"
QWEN25VL_MODEL_LICENSE = "Apache-2.0"
QWEN25VL_METRICS = (
    "multi_character_interaction_quality",
    "anatomy_quality",
    "locomotion_quality",
    "object_interaction_quality",
    "camera_motion_quality",
    "lighting_transition_quality",
    "physics_plausibility",
)
_PROMPT_SCHEMA_VERSION = "cineos-qwen25vl-difficult-case-rubric/0.1"
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Qwen25VLSemanticJudgeError(RuntimeError):
    """Raised when the external semantic judge cannot produce auditable evidence."""


def _shot_context(shot: Any) -> str:
    parts: list[str] = []
    for attr in ("prompt", "description", "shot_prompt"):
        value = getattr(shot, attr, None)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            break
    metadata = getattr(shot, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("action", "camera", "lighting", "physics", "props", "dialogue"):
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                parts.append(f"{key}: {value}")
    return "\n".join(parts) or "No additional shot description supplied."


def _rubric_prompt(shot: Any, frame_count: int) -> str:
    metric_lines = "\n".join(f'- "{name}"' for name in QWEN25VL_METRICS)
    return f"""You are a strict film-generation visual QC judge. You are given {frame_count}
ordered frames sampled from one generated video shot. Judge only evidence visible in
these frames. Scores are continuous from 0.0 (clearly broken) to 1.0 (production
quality). Penalize uncertainty; do not invent unseen evidence.

Evaluate:
- multi_character_interaction_quality: spatially coherent interaction, contact, gaze,
  occlusion and identity separation when multiple people are visible.
- anatomy_quality: hands, fingers, limbs, joints, face/body geometry and deformation.
- locomotion_quality: plausible walking/running pose progression, foot placement and
  body mechanics across ordered frames.
- object_interaction_quality: stable object geometry, grasp/contact, hand-object
  relationship and continuity.
- camera_motion_quality: visual coherence during apparent rapid camera motion; penalize
  tearing, duplicated subjects, warped geometry and incoherent motion progression.
- lighting_transition_quality: physically and temporally coherent illumination and
  shadows across changing light.
- physics_plausibility: gravity, inertia, collision/contact, cloth/hair/object behavior
  and scene causality visible across frames.

Shot intent/context:
{_shot_context(shot)}

Return exactly one JSON object with exactly these keys and numeric values only:
{metric_lines}
Do not return prose, markdown, explanations, confidence fields, or extra keys.
"""


def _parse_metrics(raw: str) -> dict[str, float]:
    if not isinstance(raw, str) or not raw.strip():
        raise Qwen25VLSemanticJudgeError("Qwen2.5-VL returned empty semantic evidence")
    match = _JSON_OBJECT.search(raw.strip())
    if match is None:
        raise Qwen25VLSemanticJudgeError("Qwen2.5-VL response did not contain a JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise Qwen25VLSemanticJudgeError("Qwen2.5-VL semantic JSON was malformed") from exc
    if not isinstance(payload, dict):
        raise Qwen25VLSemanticJudgeError("Qwen2.5-VL semantic result must be an object")
    actual = set(payload)
    expected = set(QWEN25VL_METRICS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise Qwen25VLSemanticJudgeError(
            "Qwen2.5-VL semantic result violated metric schema: " + "; ".join(details)
        )
    result: dict[str, float] = {}
    for name in QWEN25VL_METRICS:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Qwen25VLSemanticJudgeError(f"semantic metric {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise Qwen25VLSemanticJudgeError(
                f"semantic metric {name!r} must be finite and between 0 and 1"
            )
        result[name] = numeric
    return result


class Qwen25VLSemanticJudge:
    """Run pinned Qwen2.5-VL over decoded video evidence with strict provenance."""

    semantic_measurement_evidence = True

    def __init__(
        self,
        *,
        model_id: str = QWEN25VL_MODEL_ID,
        revision: str = QWEN25VL_MODEL_REVISION,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 320,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable 40-character git commit SHA")
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if max_new_tokens < 64:
            raise ValueError("max_new_tokens must be at least 64")
        if (model is None) is not (processor is None):
            raise ValueError("model and processor must either both be supplied or both omitted")
        self.model_id = model_id.strip()
        self.revision = revision
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self._model = model
        self._processor = processor

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise Qwen25VLSemanticJudgeError(
                "Qwen2.5-VL production QC requires the CINEOS video dependencies"
            ) from exc
        torch_dtype = getattr(torch, self.dtype)
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=False,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                trust_remote_code=False,
            )
        except Exception as exc:  # dependency/model-runtime boundary
            raise Qwen25VLSemanticJudgeError(
                "failed to load pinned Qwen2.5-VL semantic QC foundation"
            ) from exc
        self._processor = processor
        self._model = model
        return model, processor

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": QWEN25VL_SEMANTIC_JUDGE_SCHEMA,
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
            "model_id": self.model_id,
            "model_revision": self.revision,
            "model_license": QWEN25VL_MODEL_LICENSE,
            "measurement_method": "multimodal_model_judgment_of_ordered_decoded_frames",
            "prompt_schema": _PROMPT_SCHEMA_VERSION,
            "measured_metrics": list(QWEN25VL_METRICS),
            "device_map": self.device_map,
            "dtype": self.dtype,
            "limitations": [
                "not a CINEOS-native model",
                "not biomechanical or physical-simulation ground truth",
                "does not measure audio-visual dialogue lip-sync",
            ],
        }

    @staticmethod
    def _pil_frames(sample: RGBVideoSample) -> list[Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise Qwen25VLSemanticJudgeError(
                "Qwen2.5-VL semantic QC requires Pillow"
            ) from exc
        return [
            Image.frombytes("RGB", (sample.width, sample.height), frame)
            for frame in sample.frames
        ]

    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        del artifact, attempt_index
        model, processor = self._load()
        images = self._pil_frames(sample)
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": _rubric_prompt(shot, len(images))})
        messages = [{"role": "user", "content": content}]
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
            model_device = getattr(model, "device", None)
            if model_device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            input_ids = inputs["input_ids"] if isinstance(inputs, Mapping) else inputs.input_ids
            trimmed = [out[len(inp) :] for inp, out in zip(input_ids, generated, strict=True)]
            decoded = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise Qwen25VLSemanticJudgeError(
                "Qwen2.5-VL semantic inference failed"
            ) from exc
        if len(decoded) != 1:
            raise Qwen25VLSemanticJudgeError(
                "Qwen2.5-VL semantic inference must return exactly one response"
            )
        return _parse_metrics(decoded[0])


__all__ = [
    "QWEN25VL_METRICS",
    "QWEN25VL_MODEL_ID",
    "QWEN25VL_MODEL_LICENSE",
    "QWEN25VL_MODEL_REVISION",
    "QWEN25VL_SEMANTIC_JUDGE_SCHEMA",
    "Qwen25VLSemanticJudge",
    "Qwen25VLSemanticJudgeError",
]
