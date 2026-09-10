import json
from pathlib import Path

import pytest

from cineos.atlas.connected_benchmark_fixture_preflight import (
    ConnectedBenchmarkFixturePreflightError,
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


def test_canonical_connected_fixture_binds_executable_gpu_challenges():
    result = validate_connected_benchmark_fixture(_FIXTURE_PATH, shot_count=5)

    assert result["validated"] is True
    assert result["case_id"] == "competitive-connected-film"
    assert result["shot_count"] == 5
    assert result["required_competitive_challenges"] == sorted(
        REQUIRED_COMPETITIVE_CHALLENGES
    )
    assert len(result["fixture_sha256"]) == 64


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
