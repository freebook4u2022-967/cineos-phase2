import json

import pytest

from cineos.atlas.qwen25vl_semantic_judge import (
    QWEN25VL_METRICS,
    Qwen25VLSemanticJudgeError,
    _parse_metrics,
)


def _valid_payload() -> dict[str, float]:
    return {metric: 0.8 for metric in QWEN25VL_METRICS}


def test_parse_metrics_accepts_exact_json_with_surrounding_whitespace() -> None:
    payload = _valid_payload()

    assert _parse_metrics(f"\n  {json.dumps(payload)}  \t") == payload


@pytest.mark.parametrize(
    "raw",
    [
        lambda body: f"Here is the result: {body}",
        lambda body: f"{body}\nAll metrics passed.",
        lambda body: f"```json\n{body}\n```",
        lambda body: f"{body}{body}",
    ],
)
def test_parse_metrics_rejects_non_json_envelopes(raw) -> None:
    body = json.dumps(_valid_payload())

    with pytest.raises(
        Qwen25VLSemanticJudgeError,
        match="exactly one JSON object",
    ):
        _parse_metrics(raw(body))
