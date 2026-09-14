import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cineos.atlas.connected_benchmark_fixture_preflight as fixture_preflight
from cineos.atlas.connected_benchmark_fixture_preflight import (
    ConnectedBenchmarkFixturePreflightError,
    preflight_connected_benchmark_fixture,
    validate_connected_benchmark_fixture,
)
from cineos.atlas.gpu_benchmark_cli import REQUIRED_COMPETITIVE_CHALLENGES

_FIXTURE_PATH = Path("benchmarks/projects/competitive-connected-film.json")


def _payload() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _write_fixture(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "connected-film.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _benchmark_metadata() -> dict[str, object]:
    return {"competitive_challenges": sorted(REQUIRED_COMPETITIVE_CHALLENGES)}


def _fake_requests(order: tuple[int, ...] = (0, 1, 2, 3, 4)):
    return tuple(
        SimpleNamespace(
            shot_id=f"shot-{index}",
            content_hash=hashlib.sha256(f"payload-{index}".encode()).hexdigest(),
            metadata=_benchmark_metadata(),
            characters=[
                {
                    "character_uuid": "character-a",
                    "approved_reference_ids": ["identity-a"],
                }
            ],
            approved_reference_ids=["identity-a"],
        )
        for index in order
    )


def _multi_character_requests(*, bind_second_character: bool = True):
    requests = list(_fake_requests())
    second_refs = ["identity-b"] if bind_second_character else []
    requests[1] = SimpleNamespace(
        shot_id="shot-1",
        content_hash=hashlib.sha256(b"multi-character-payload").hexdigest(),
        metadata=_benchmark_metadata(),
        characters=[
            {
                "character_uuid": "character-a",
                "approved_reference_ids": ["identity-a"],
            },
            {
                "character_uuid": "character-b",
                "approved_reference_ids": second_refs,
            },
        ],
        approved_reference_ids=["identity-a", "identity-b"],
    )
    return tuple(requests)


def test_canonical_connected_fixture_binds_executable_gpu_challenges():
    result = validate_connected_benchmark_fixture(_FIXTURE_PATH, shot_count=5)

    assert result["validated"] is True
    assert result["case_id"] == "competitive-connected-film"
    assert result["shot_count"] == 5
    assert result["required_competitive_challenges"] == sorted(
        REQUIRED_COMPETITIVE_CHALLENGES
    )
    assert len(result["fixture_sha256"]) == 64


def test_preflight_hash_binds_exact_ordered_normalized_request_bundle(monkeypatch):
    requests = _fake_requests()
    monkeypatch.setattr(fixture_preflight, "load_native_requests", lambda _: requests)

    result = preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)

    expected_payload = [
        {"shot_id": request.shot_id, "content_hash": request.content_hash}
        for request in requests
    ]
    expected_hash = hashlib.sha256(
        json.dumps(
            expected_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert result["schema"] == "cineos-connected-benchmark-fixture-preflight/0.4"
    assert result["ordered_shot_ids"] == [request.shot_id for request in requests]
    assert result["normalized_request_bundle_sha256"] == expected_hash
    assert result["identity_assignment_shot_ids"] == []
    assert result["identity_consistency_bindings"]["shot-0"] == [
        {"character_id": "character-a", "approved_reference_id": "identity-a"}
    ]


def test_preflight_binds_each_competitive_challenge_to_exact_shot_ids(monkeypatch):
    requests = _fake_requests()
    monkeypatch.setattr(fixture_preflight, "load_native_requests", lambda _: requests)

    result = preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)

    expected_shot_ids = [request.shot_id for request in requests]
    assert result["competitive_challenge_shot_ids"] == {
        challenge: expected_shot_ids
        for challenge in sorted(REQUIRED_COMPETITIVE_CHALLENGES)
    }


def test_preflight_rejects_validated_bundle_missing_challenge_binding(monkeypatch):
    requests = list(_fake_requests())
    reduced = sorted(REQUIRED_COMPETITIVE_CHALLENGES - {"physics"})
    for request in requests:
        request.metadata["competitive_challenges"] = reduced
    monkeypatch.setattr(
        fixture_preflight, "load_native_requests", lambda _: tuple(requests)
    )

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="lost mandatory competitive challenge coverage: physics",
    ):
        preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)


def test_preflight_requires_explicit_identity_assignment_for_each_multi_character_cast_member(
    monkeypatch,
):
    requests = _multi_character_requests(bind_second_character=False)
    monkeypatch.setattr(fixture_preflight, "load_native_requests", lambda _: requests)

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="character-local approved_reference_ids for every cast member",
    ):
        preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)


def test_preflight_records_explicit_multi_character_identity_assignment(monkeypatch):
    requests = _multi_character_requests()
    monkeypatch.setattr(fixture_preflight, "load_native_requests", lambda _: requests)

    result = preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)

    assert result["identity_assignment_shot_ids"] == ["shot-1"]


def test_preflight_bundle_hash_changes_when_same_requests_are_reordered(monkeypatch):
    requests = _fake_requests()
    reversed_requests = _fake_requests((4, 3, 2, 1, 0))
    monkeypatch.setattr(fixture_preflight, "load_native_requests", lambda _: requests)
    original = preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)
    monkeypatch.setattr(
        fixture_preflight, "load_native_requests", lambda _: reversed_requests
    )
    reordered = preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)

    assert original["shot_count"] == reordered["shot_count"] == 5
    assert (
        original["normalized_request_bundle_sha256"]
        != reordered["normalized_request_bundle_sha256"]
    )


def test_preflight_rejects_noncanonical_validated_request_hash(monkeypatch):
    requests = list(_fake_requests())
    requests[2] = SimpleNamespace(
        shot_id="shot-2",
        content_hash="g" * 64,
        metadata=_benchmark_metadata(),
        characters=[
            {
                "character_uuid": "character-a",
                "approved_reference_ids": ["identity-a"],
            }
        ],
        approved_reference_ids=["identity-a"],
    )
    monkeypatch.setattr(
        fixture_preflight, "load_native_requests", lambda _: tuple(requests)
    )

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="canonical SHA-256 content_hash",
    ):
        preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)


def test_connected_fixture_rejects_missing_executable_challenge(tmp_path):
    payload = _payload()
    challenges = list(payload["required_competitive_challenges"])
    challenges.remove("hands_anatomy")
    payload["required_competitive_challenges"] = challenges

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="competitive challenges do not match",
    ):
        validate_connected_benchmark_fixture(
            _write_fixture(tmp_path, payload), shot_count=5
        )


def test_connected_fixture_rejects_relaxed_real_inference_contract(tmp_path):
    payload = _payload()
    payload["real_inference_required"] = False

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="must require real inference",
    ):
        validate_connected_benchmark_fixture(
            _write_fixture(tmp_path, payload), shot_count=5
        )


def test_connected_fixture_rejects_relaxed_shot_envelope(tmp_path):
    payload = _payload()
    payload["minimum_connected_shots"] = 1

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="minimum_connected_shots must be 5",
    ):
        validate_connected_benchmark_fixture(
            _write_fixture(tmp_path, payload), shot_count=5
        )


def test_connected_fixture_rejects_request_bundle_outside_five_to_ten_shots():
    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="requires 5-10 shots",
    ):
        validate_connected_benchmark_fixture(_FIXTURE_PATH, shot_count=4)


def test_connected_fixture_rejects_wrong_case_identity(tmp_path):
    payload = _payload()
    payload["case_id"] = "competitive-identity-closeup"

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError, match="case_id must be"
    ):
        validate_connected_benchmark_fixture(
            _write_fixture(tmp_path, payload), shot_count=5
        )
