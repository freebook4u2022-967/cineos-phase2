from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from cineos.film.benchmark_film_assembly import build_production_shot_evidence
from cineos.film.exceptions import AssemblyError


def _sha(index: int) -> str:
    return f"{index:064x}"


def _chain(receipts: list[SimpleNamespace]) -> str:
    digest = hashlib.sha256()
    for receipt in receipts:
        digest.update(receipt.result.request_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(receipt.output_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _benchmark(
    *,
    tamper_report_output: bool = False,
    tamper_measurement: bool = False,
    tamper_chain: bool = False,
    tamper_total_bytes: bool = False,
    tamper_profile: bool = False,
    tamper_origin: bool = False,
):
    profile_id = "wan22-a14b-production"
    origin = "external-open-pretrained"
    receipts = []
    reports = []
    for index in range(1, 6):
        shot_id = f"shot-{index}"
        output_sha = _sha(index)
        receipts.append(
            SimpleNamespace(
                profile_id=(
                    "substituted-profile"
                    if tamper_profile and index == 3
                    else profile_id
                ),
                origin=(
                    "substituted-origin" if tamper_origin and index == 3 else origin
                ),
                output_sha256=output_sha,
                output_bytes=1000 + index,
                result=SimpleNamespace(
                    shot_id=shot_id,
                    output_path=f"/tmp/{shot_id}.mp4",
                    request_hash=_sha(100 + index),
                ),
            )
        )
        report_output = _sha(99) if tamper_report_output and index == 3 else output_sha
        measurement_output = (
            _sha(98) if tamper_measurement and index == 3 else output_sha
        )
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
    chain_sha = _sha(500) if tamper_chain else _chain(receipts)
    total_output_bytes = sum(receipt.output_bytes for receipt in receipts)
    if tamper_total_bytes:
        total_output_bytes += 1

    return SimpleNamespace(
        evidence_tier="production-gpu-quality-gated",
        production_gpu_evidence=True,
        production_quality_evidence=True,
        profile_id=profile_id,
        origin=origin,
        chain_sha256=chain_sha,
        total_output_bytes=total_output_bytes,
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
    with pytest.raises(
        AssemblyError, match="QC measurement is bound to another render"
    ):
        build_production_shot_evidence(_benchmark(tamper_measurement=True))


def test_build_production_shot_evidence_requires_quality_gated_tier() -> None:
    benchmark = _benchmark()
    benchmark.evidence_tier = "production-gpu-execution"

    with pytest.raises(AssemblyError, match="production-gpu-quality-gated"):
        build_production_shot_evidence(benchmark)


def test_build_production_shot_evidence_rejects_stale_aggregate_chain() -> None:
    with pytest.raises(AssemblyError, match="aggregate chain does not match"):
        build_production_shot_evidence(_benchmark(tamper_chain=True))


def test_build_production_shot_evidence_rejects_stale_aggregate_byte_count() -> None:
    with pytest.raises(
        AssemblyError, match="aggregate output byte count does not match"
    ):
        build_production_shot_evidence(_benchmark(tamper_total_bytes=True))


def test_build_production_shot_evidence_rejects_profile_substitution() -> None:
    with pytest.raises(AssemblyError, match="profile does not match aggregate profile"):
        build_production_shot_evidence(_benchmark(tamper_profile=True))


def test_build_production_shot_evidence_rejects_origin_substitution() -> None:
    with pytest.raises(AssemblyError, match="origin does not match aggregate origin"):
        build_production_shot_evidence(_benchmark(tamper_origin=True))
