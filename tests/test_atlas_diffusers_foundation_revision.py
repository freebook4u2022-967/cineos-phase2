"""Regression coverage for immutable Diffusers foundation revision routing."""

import pytest

from cineos.atlas.diffusers_video import (
    DiffusersVideoError,
    DiffusersVideoRenderer,
    FoundationProvenance,
)


class MinimalPipeline:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


def _renderer(tmp_path, foundation, factory):
    return DiffusersVideoRenderer(
        foundation,
        output_dir=tmp_path,
        pipeline_factory=factory,
        video_exporter=lambda *_args, **_kwargs: None,
    )


def test_pinned_foundation_revision_is_forwarded_to_pipeline_factory(tmp_path):
    calls = []
    pipeline = MinimalPipeline()

    def factory(model_id, **options):
        calls.append((model_id, options))
        return pipeline

    renderer = _renderer(
        tmp_path,
        FoundationProvenance(model_id="declared/model", revision="immutable-sha"),
        factory,
    )

    renderer.load_model(device="cpu")

    assert calls == [("declared/model", {"revision": "immutable-sha"})]
    assert pipeline.devices == ["cpu"]


def test_matching_explicit_revision_remains_supported(tmp_path):
    calls = []
    pipeline = MinimalPipeline()

    def factory(model_id, **options):
        calls.append((model_id, options))
        return pipeline

    renderer = _renderer(
        tmp_path,
        FoundationProvenance(model_id="declared/model", revision="immutable-sha"),
        factory,
    )

    renderer.load_model(device="cpu", revision="immutable-sha")

    assert calls == [("declared/model", {"revision": "immutable-sha"})]


def test_mismatched_revision_fails_before_pipeline_load(tmp_path):
    calls = []

    def factory(model_id, **options):
        calls.append((model_id, options))
        return MinimalPipeline()

    renderer = _renderer(
        tmp_path,
        FoundationProvenance(model_id="declared/model", revision="immutable-sha"),
        factory,
    )

    with pytest.raises(
        DiffusersVideoError,
        match="does not match declared foundation revision",
    ):
        renderer.load_model(device="cpu", revision="different-revision")

    assert calls == []


def test_unpinned_foundation_keeps_explicit_revision_backward_compatible(tmp_path):
    calls = []
    pipeline = MinimalPipeline()

    def factory(model_id, **options):
        calls.append((model_id, options))
        return pipeline

    renderer = _renderer(
        tmp_path,
        FoundationProvenance(model_id="declared/model"),
        factory,
    )

    renderer.load_model(device="cpu", revision="caller-selected-revision")

    assert calls == [("declared/model", {"revision": "caller-selected-revision"})]
