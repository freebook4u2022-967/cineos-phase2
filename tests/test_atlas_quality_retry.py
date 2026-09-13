import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.quality_retry import (
    QualityRetryError,
    QualityRetryPolicy,
    build_quality_retry_request,
)


def _request():
    request = NativeShotRequest(
        shot_id="shot-02",
        scene_id="scene-01",
        camera={"resolution": [1280, 704], "fps": 24.0, "duration": 5.0},
        characters=[{"identity_invariants": ["same approved face"]}],
        environment={"description": "rainy harbor"},
        wardrobe=[{"character_id": "hero", "description": "black coat"}],
        props=[],
        continuity={"previous_shot": "shot-01"},
        performance={"facial_targets": ["focused"]},
        approved_reference_ids=["hero-approved-01"],
        deterministic_seed=1234,
        renderer_requirements={"features": ["image_to_video"]},
        metadata={"prompt": "hero walks toward the ship"},
    )
    request.refresh_hash()
    return request


def _rejection():
    return {
        "accepted": False,
        "failed_metrics": ["identity_similarity", "temporal_consistency"],
        "directives": [
            "preserve approved character identity and facial structure",
            "reduce cross-frame and cross-shot temporal drift",
        ],
    }


def test_builds_fresh_hash_bound_retry_without_mutating_original():
    request = _request()
    original_hash = request.content_hash

    retry = build_quality_retry_request(request, _rejection(), attempt_index=1)

    assert request.content_hash == original_hash
    assert request.deterministic_seed == 1234
    assert "quality_retry" not in request.metadata
    assert retry.scene_id == request.scene_id
    assert retry.shot_id == request.shot_id
    assert retry.approved_reference_ids == request.approved_reference_ids
    assert retry.continuity == request.continuity
    assert retry.deterministic_seed == 1234 + 104_729
    assert retry.content_hash != original_hash
    assert retry.metadata["quality_retry"]["parent_request_hash"] == original_hash
    assert retry.metadata["quality_retry"]["root_request_hash"] == original_hash
    assert retry.metadata["quality_retry"]["original_seed"] == 1234
    assert retry.metadata["quality_retry"]["attempt_index"] == 1
    assert retry.metadata["quality_directives"] == _rejection()["directives"]


def test_retry_hash_is_deterministic_for_same_report_and_attempt():
    request = _request()

    first = build_quality_retry_request(request, _rejection(), attempt_index=1)
    second = build_quality_retry_request(request, _rejection(), attempt_index=1)

    assert first.content_hash == second.content_hash
    assert first.deterministic_seed == second.deterministic_seed


def test_second_retry_uses_root_seed_and_preserves_parent_lineage():
    request = _request()
    root_hash = request.content_hash
    first = build_quality_retry_request(request, _rejection(), attempt_index=1)
    report = {
        "accepted": False,
        "failed_metrics": ["motion_quality"],
        "directives": ["stabilize physically plausible subject and camera motion"],
    }

    second = build_quality_retry_request(first, report, attempt_index=2)

    assert first.deterministic_seed == 1234 + 104_729
    assert second.deterministic_seed == 1234 + (104_729 * 2)
    assert second.metadata["quality_retry"]["original_seed"] == 1234
    assert second.metadata["quality_retry"]["root_request_hash"] == root_hash
    assert second.metadata["quality_retry"]["parent_request_hash"] == first.content_hash
    assert second.metadata["quality_directives"] == [
        "preserve approved character identity and facial structure",
        "reduce cross-frame and cross-shot temporal drift",
        "stabilize physically plausible subject and camera motion",
    ]


def test_chained_retries_follow_linear_seed_schedule():
    request = _request()
    policy = QualityRetryPolicy(max_attempts=4, seed_stride=17)
    seeds = [request.deterministic_seed]
    effective = request

    for attempt_index in range(1, policy.max_attempts):
        effective = build_quality_retry_request(
            effective,
            _rejection(),
            attempt_index=attempt_index,
            policy=policy,
        )
        seeds.append(effective.deterministic_seed)

    assert seeds == [1234, 1251, 1268, 1285]
    assert effective.metadata["quality_retry"]["original_seed"] == 1234
    assert (
        effective.metadata["quality_retry"]["root_request_hash"] == request.content_hash
    )


def test_rejects_corrupt_retry_original_seed_lineage():
    request = _request()
    first = build_quality_retry_request(request, _rejection(), attempt_index=1)
    first.metadata["quality_retry"]["original_seed"] = "1234"
    first.refresh_hash()

    with pytest.raises(QualityRetryError, match="original_seed must be an integer"):
        build_quality_retry_request(first, _rejection(), attempt_index=2)


def test_rejects_accepted_report_and_missing_directives():
    request = _request()

    with pytest.raises(QualityRetryError, match="accepted quality reports"):
        build_quality_retry_request(
            request,
            {"accepted": True, "directives": ["unused"]},
            attempt_index=1,
        )

    with pytest.raises(QualityRetryError, match="must provide correction directives"):
        build_quality_retry_request(
            request,
            {"accepted": False, "failed_metrics": ["motion_quality"]},
            attempt_index=1,
        )


def test_policy_bounds_retry_count():
    request = _request()
    policy = QualityRetryPolicy(max_attempts=2, seed_stride=7)

    retry = build_quality_retry_request(
        request, _rejection(), attempt_index=1, policy=policy
    )
    assert retry.deterministic_seed == 1241

    with pytest.raises(QualityRetryError, match="exceeds max_attempts"):
        build_quality_retry_request(
            request, _rejection(), attempt_index=2, policy=policy
        )
