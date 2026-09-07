from __future__ import annotations

import json
from pathlib import Path

import pytest

from cineos.atlas.production_connected_evidence import (
    ProductionConnectedEvidenceError,
    validate_production_connected_evidence,
)
from test_atlas_production_connected_evidence import (
    _benchmark,
    _challenge,
    _resign_challenge_contract,
)


def test_rejects_resigned_challenge_contract_repeating_same_shot(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    manifest = Path(benchmark.manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    contract = _challenge(payload)
    mapping = contract["challenge_to_shots"]
    assert isinstance(mapping, dict)
    mapping["dialogue"] = ["scene-1/shot-1", "scene-1/shot-1"]
    _resign_challenge_contract(contract)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="repeats the same shot",
    ):
        validate_production_connected_evidence(benchmark)
