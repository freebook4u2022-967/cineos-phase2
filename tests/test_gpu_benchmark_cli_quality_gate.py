from types import SimpleNamespace

import pytest

from cineos.atlas import gpu_benchmark_cli as cli
from cineos.atlas.siglip2_video_scorer import SigLIP2VideoScorerError


def _quality_gated_receipt(*, production_quality_evidence: bool = True):
    return SimpleNamespace(
        production_gpu_evidence=True,
        production_quality_evidence=production_quality_evidence,
        evidence_tier=(
            "production-gpu-quality-gated"
            if production_quality_evidence
            else "production-gpu"
        ),
    )


def test_production_cli_routes_real_render_through_quality_retry_gate(
    monkeypatch, tmp_path
):
    evaluator = object()
    observed = {}

    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: evaluator,
    )

    def fake_quality_retry(benchmark_id, requests, profile, **kwargs):
        observed["benchmark_id"] = benchmark_id
        observed["requests"] = requests
        observed["profile"] = profile
        observed.update(kwargs)
        return _quality_gated_receipt()

    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        fake_quality_retry,
    )

    receipt = cli.run_production_benchmark(
        "bench-quality",
        (),
        output_dir=tmp_path,
        reference_manifest="approved-references.json",
    )

    assert receipt.production_quality_evidence is True
    assert observed["benchmark_id"] == "bench-quality"
    assert observed["quality_evaluator"] is evaluator
    assert observed["reference_manifest"] == "approved-references.json"
    assert observed["profile"] is cli.WAN22_TI2V_5B_PROFILE


def test_production_cli_rejects_receipt_without_production_quality_evidence(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: object(),
    )
    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        lambda *args, **kwargs: _quality_gated_receipt(
            production_quality_evidence=False
        ),
    )

    with pytest.raises(
        cli.GPUProductionBenchmarkCLIError,
        match="artifact-bound production QC evidence",
    ):
        cli.run_production_benchmark(
            "bench-no-qc",
            (),
            output_dir=tmp_path,
            reference_manifest="approved-references.json",
        )


def test_production_quality_evaluator_fails_closed_when_pinned_qc_unavailable(
    monkeypatch,
):
    loader = object()
    monkeypatch.setattr(
        cli,
        "_production_reference_loader",
        lambda requests, reference_manifest: loader,
    )
    monkeypatch.setattr(
        cli,
        "_production_multi_reference_adapter",
        lambda requests: None,
    )

    class MissingPinnedQC:
        def __init__(self, reference_loader, *, device):
            assert reference_loader is loader
            assert device == "cuda"
            raise SigLIP2VideoScorerError("pinned snapshot unavailable")

    monkeypatch.setattr(cli, "SigLIP2FeatureVideoScorer", MissingPinnedQC)

    with pytest.raises(
        cli.GPUProductionBenchmarkCLIError,
        match="cannot initialize pinned production visual QC",
    ):
        cli._production_quality_evaluator((), "approved-references.json")
