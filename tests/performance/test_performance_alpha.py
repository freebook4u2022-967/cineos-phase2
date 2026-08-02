from cineos.performance import (
    PerformancePlan,
    PerformanceValidator,
    phonemes_to_visemes,
)
from cineos.performance.serializer import calculate_content_hash, deserialize, serialize


def test_phonemes_and_deterministic_serialization():
    assert phonemes_to_visemes([{"phoneme": "M", "time": 0}])[0]["viseme"] == "closed"
    plan = PerformancePlan("shot", "scene", performance_id="stable")
    plan.content_hash = calculate_content_hash(plan)
    restored = deserialize(serialize(plan))
    assert calculate_content_hash(restored) == plan.content_hash


def test_capability_failure_is_explicit():
    plan = PerformancePlan("shot", "scene")
    plan.renderer_capability_requirements.facial_control = True
    report = PerformanceValidator().validate(plan, renderer_features=set())
    assert "unsupported renderer capability: facial-control" in report.errors
