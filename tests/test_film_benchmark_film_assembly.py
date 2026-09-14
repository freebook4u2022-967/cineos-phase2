from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.film.benchmark_film_assembly import build_production_shot_evidence
from cineos.film.exceptions import AssemblyError


def _sha(index: int) -> str:
    return f"{index:064x}"


def _benchmark(*, tamper_report_output: bool = False, tamper_measurement: bool = False):
    receipts = []
    reports = []
    for index in range(1, 6):
        shot_id = f"shot-{index}"
        output_sha = _sha(index)
        receipts.append(
            SimpleNamespace(
                output_sha256=output_sha,
                result=SimpleNamespace(
                    shot_id=shot_id,
                    output_path=f"/tmp/{shot_id}.mp4",
                ),
            )
        )
        report_output = _sha(99) if tamper_report_output and index == 3 else output_sha
        measurement_output = _sha(98) if tamper_measurement and index == 3 else output_sha
        reports.append(
            {
                "accepted": True,
                "production_measurement_evidence": True,
                "shot_id": shot_id,
                "output_sha256": report_output,
                "measurement": {
                    "schema": "cineos-sequence-quality-measurement/0.1",
                    "observer_id": "test-observer",
                    "artifact_sha256": measurement_output,
                },
            }
        )
    return SimpleNamespace(
        evidence_tier="production-gpu-quality-gated",
        production_gpu_evidence=True,
        production_quality_evidence=True,
        shot_receipts=tuple(receipts),
        quality_reports=tuple(reports),
    )


def test_build_production_shot_evidence_binds_qc_to_exact_render() -> None:
    records = build_production_shot_evidence(_benchmark())

    assert len(records) == 5
    assert [record["shot_id"] for record in records] == [
        "shot-1",
        "shot-2",
        "shot-3",
        "shot-4",
        "shot-5",
    ]
    assert all(record["accepted"] is True for record in records)
    assert all(record["decision"] == "accept" for record in records)
    assert all(record["production_gpu_evidence"] is True for record in records)
    assert len({record["evidence_sha256"] for record in records}) == 5


def test_build_production_shot_evidence_rejects_substituted_quality_report() -> None:
    with pytest.raises(AssemblyError, match="QC report is bound to another render"):
        build_production_shot_evidence(_benchmark(tamper_report_output=True))


def test_build_production_shot_evidence_rejects_substituted_measurement() -> None:
    with pytest.raises(AssemblyError, match="QC measurement is bound to another render"):
        build_production_shot_evidence(_benchmark(tamper_measurement=True))


def test_build_production_shot_evidence_requires_quality_gated_tier() -> None:
    benchmark = _benchmark()
    benchmark.evidence_tier = "production-gpu-execution"

    with pytest.raises(AssemblyError, match="production-gpu-quality-gated"):
        build_production_shot_evidence(benchmark)
