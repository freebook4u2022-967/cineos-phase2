from __future__ import annotations

import hashlib
import json

from cineos.atlas.seedance_style_challenge import REQUIRED_CHALLENGES, ChallengeCoverage


def test_competitive_challenge_contract_uses_production_supported_schema() -> None:
    coverage = ChallengeCoverage(
        challenge_to_shots={
            challenge: ("scene-01/shot-01",) for challenge in REQUIRED_CHALLENGES
        }
    )

    payload = coverage.to_dict()

    assert payload["schema"] == "cineos-seedance-style-challenge-coverage/0.1"
    recorded = payload.pop("contract_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert recorded == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
