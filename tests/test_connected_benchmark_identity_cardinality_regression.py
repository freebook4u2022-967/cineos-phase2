import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import cineos.atlas.connected_benchmark_fixture_preflight as fixture_preflight
from cineos.atlas.connected_benchmark_fixture_preflight import (
    ConnectedBenchmarkFixturePreflightError,
    preflight_connected_benchmark_fixture,
)

_FIXTURE_PATH = Path("benchmarks/projects/competitive-connected-film.json")


def test_two_approved_refs_cannot_replace_per_character_identity_ownership(monkeypatch):
    requests = []
    for index in range(5):
        characters = []
        approved_reference_ids = []
        if index == 2:
            approved_reference_ids = ["identity-a", "identity-b"]
            characters = [
                {
                    "character_uuid": "character-a",
                    "approved_reference_ids": ["identity-a"],
                },
                {"character_uuid": "character-b"},
            ]
        requests.append(
            SimpleNamespace(
                shot_id=f"shot-{index}",
                content_hash=hashlib.sha256(f"shot-{index}".encode()).hexdigest(),
                characters=characters,
                approved_reference_ids=approved_reference_ids,
            )
        )

    monkeypatch.setattr(
        fixture_preflight, "load_native_requests", lambda _: tuple(requests)
    )

    with pytest.raises(
        ConnectedBenchmarkFixturePreflightError,
        match="character-local approved_reference_ids for every cast member",
    ):
        preflight_connected_benchmark_fixture("requests.json", _FIXTURE_PATH)
