from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.atlas.artifact_video_observer import RGBVideoSample
from cineos.atlas.qwen25vl_semantic_judge import (
    QWEN25VL_METRICS,
    QWEN25VL_MODEL_ID,
    QWEN25VL_MODEL_REVISION,
    Qwen25VLSemanticJudge,
    Qwen25VLSemanticJudgeError,
    _parse_metrics,
)


class _Inputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class _Processor:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "rendered-chat-template"

    def __call__(self, *, text, images, padding, return_tensors):
        assert text == ["rendered-chat-template"]
        assert padding is True
        assert return_tensors == "pt"
        self.images = images
        return _Inputs(input_ids=[[10, 11]])

    def batch_decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert token_ids == [[12, 13]]
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return [self.response]


class _Model:
    device = None

    def generate(self, **kwargs):
        assert kwargs["input_ids"] == [[10, 11]]
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 320
        return [[10, 11, 12, 13]]


def _valid_payload(value: float = 0.81) -> dict[str, float]:
    return {name: value for name in QWEN25VL_METRICS}


def test_qwen25vl_judge_executes_injected_real_model_boundary() -> None:
    processor = _Processor(json.dumps(_valid_payload()))
    judge = Qwen25VLSemanticJudge(model=_Model(), processor=processor)
    sample = RGBVideoSample(
        width=2,
        height=1,
        frames=(bytes([0, 0, 0, 255, 255, 255]), bytes([8, 9, 10, 20, 21, 22])),
    )
    shot = SimpleNamespace(
        prompt="Two actors exchange a parcel while walking through changing light.",
        metadata={"props": ["parcel"], "camera": "fast tracking"},
    )

    result = judge(
        sample,
        artifact=Path("shot.mp4"),
        shot=shot,
        attempt_index=0,
    )

    assert result == _valid_payload()
    assert processor.images is not None
    assert len(processor.images) == 2
    assert processor.messages is not None
    prompt = processor.messages[0]["content"][-1]["text"]
    assert "Two actors exchange a parcel" in prompt
    assert "fast tracking" in prompt
    assert "dialogue_lip_sync" not in result


def test_qwen25vl_provenance_is_external_pinned_and_explicit_about_limitations() -> None:
    judge = Qwen25VLSemanticJudge(model=_Model(), processor=_Processor("{}"))

    provenance = judge.runtime_provenance()

    assert provenance["origin"] == "external_pretrained"
    assert provenance["production_measurement_evidence"] is True
    assert provenance["model_id"] == QWEN25VL_MODEL_ID
    assert provenance["model_revision"] == QWEN25VL_MODEL_REVISION
    assert len(provenance["model_revision"]) == 40
    assert provenance["model_license"] == "Apache-2.0"
    assert "does not measure audio-visual dialogue lip-sync" in provenance["limitations"]
    assert set(provenance["measured_metrics"]) == set(QWEN25VL_METRICS)


def test_qwen25vl_rejects_mutable_model_revision() -> None:
    with pytest.raises(ValueError, match="immutable 40-character git commit SHA"):
        Qwen25VLSemanticJudge(
            revision="main",
            model=_Model(),
            processor=_Processor("{}"),
        )


def test_qwen25vl_rejects_missing_or_extra_semantic_metrics() -> None:
    payload = _valid_payload()
    payload.pop("physics_plausibility")
    payload["identity_similarity"] = 0.99

    with pytest.raises(Qwen25VLSemanticJudgeError, match="metric schema"):
        _parse_metrics(json.dumps(payload))


def test_qwen25vl_rejects_nonfinite_or_out_of_range_scores() -> None:
    payload = _valid_payload()
    payload["anatomy_quality"] = 1.1

    with pytest.raises(Qwen25VLSemanticJudgeError, match="between 0 and 1"):
        _parse_metrics(json.dumps(payload))
