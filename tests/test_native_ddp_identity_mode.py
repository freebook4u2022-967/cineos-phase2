import json
from types import SimpleNamespace

import pytest

from cineos.native_image.ddp_entrypoint import (
    _anchors_for_batch,
    load_identity_anchors,
    validate_identity_coverage,
)
from cineos.native_image.neural_backend import _load_torch, torch_available


def test_load_identity_anchors_normalizes_vectors(tmp_path):
    path = tmp_path / "identity-bank.json"
    path.write_text(
        json.dumps(
            {"characters": {"arif": [3.0, 4.0], "hana": {"vector": [0.0, 2.0]}}}
        ),
        encoding="utf-8",
    )
    anchors = load_identity_anchors(path)
    assert anchors["arif"] == pytest.approx((0.6, 0.8))
    assert anchors["hana"] == pytest.approx((0.0, 1.0))


def test_load_identity_anchors_rejects_dimension_mismatch(tmp_path):
    path = tmp_path / "identity-bank.json"
    path.write_text(
        json.dumps({"characters": {"arif": [1.0, 0.0], "hana": [1.0, 0.0, 0.0]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="same dimension"):
        load_identity_anchors(path)


def test_validate_identity_coverage_counts_real_training_records():
    dataset = SimpleNamespace(
        records=(
            SimpleNamespace(character_id="arif"),
            SimpleNamespace(character_id="hana"),
            SimpleNamespace(character_id="arif"),
        )
    )
    counts = validate_identity_coverage(
        dataset,
        {"arif": (1.0, 0.0), "hana": (0.0, 1.0), "unused": (0.5, 0.5)},
    )
    assert counts == {"arif": 2, "hana": 1}


def test_validate_identity_coverage_fails_before_training_for_missing_anchor():
    dataset = SimpleNamespace(
        records=(
            SimpleNamespace(character_id="arif"),
            SimpleNamespace(character_id="hana"),
        )
    )
    with pytest.raises(ValueError, match="hana"):
        validate_identity_coverage(dataset, {"arif": (1.0, 0.0)})


def test_anchors_for_batch_preserves_character_order():
    if not torch_available():
        pytest.skip("PyTorch optional neural extra not installed")
    torch = _load_torch()
    anchors = {"arif": (1.0, 0.0), "hana": (0.0, 1.0)}
    batch = _anchors_for_batch(torch, ("hana", "arif"), anchors, torch.device("cpu"))
    assert batch.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_anchors_for_batch_rejects_missing_character():
    if not torch_available():
        pytest.skip("PyTorch optional neural extra not installed")
    torch = _load_torch()
    with pytest.raises(ValueError, match="hana"):
        _anchors_for_batch(torch, ("hana",), {"arif": (1.0, 0.0)}, torch.device("cpu"))
