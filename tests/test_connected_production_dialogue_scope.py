from types import SimpleNamespace

import pytest

from cineos.film.connected_production_evidence import (
    ConnectedProductionFilmEvidenceError,
    _resolve_dialogue_shot_ids,
)


def _benchmark(*, declared=True, dialogue_shot_ids=None, shot_ids=("shot-1", "shot-2")):
    return SimpleNamespace(
        dialogue_scope_declared=declared,
        dialogue_shot_ids=dialogue_shot_ids,
        shot_receipts=[
            SimpleNamespace(result=SimpleNamespace(shot_id=shot_id))
            for shot_id in shot_ids
        ],
    )


def test_gpu_declared_dialogue_scope_is_authoritative_when_caller_omits_scope():
    benchmark = _benchmark(dialogue_shot_ids=["shot-2"])

    assert _resolve_dialogue_shot_ids(benchmark, ()) == ("shot-2",)


def test_gpu_declared_dialogue_scope_rejects_conflicting_caller_scope():
    benchmark = _benchmark(dialogue_shot_ids=["shot-2"])

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="caller dialogue scope conflicts",
    ):
        _resolve_dialogue_shot_ids(benchmark, ("shot-1",))


def test_gpu_declared_dialogue_scope_rejects_unknown_rendered_shot():
    benchmark = _benchmark(dialogue_shot_ids=["shot-3"])

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="unknown rendered shot",
    ):
        _resolve_dialogue_shot_ids(benchmark, ())


def test_gpu_declared_dialogue_scope_requires_id_list():
    benchmark = _benchmark(dialogue_shot_ids=None)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="declared dialogue scope without dialogue shot IDs",
    ):
        _resolve_dialogue_shot_ids(benchmark, ())


def test_legacy_benchmark_keeps_explicit_dialogue_scope_compatibility():
    benchmark = _benchmark(declared=None, dialogue_shot_ids=None)

    assert _resolve_dialogue_shot_ids(benchmark, ("shot-1",)) == ("shot-1",)
