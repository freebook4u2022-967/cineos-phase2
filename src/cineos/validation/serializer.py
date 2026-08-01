"""Deterministic validation report serialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .base import ValidationResult, ValidationStatus
from .report import ValidationReport


def report_to_dict(
    report: ValidationReport, *, include_hash: bool = True
) -> dict[str, Any]:
    payload = asdict(report)
    payload["overall_status"] = report.overall_status.value
    for source, result in zip(payload["results"], report.results, strict=True):
        source["status"] = result.status.value
    if not include_hash:
        payload["content_hash"] = ""
    return payload


def canonical_json(report: ValidationReport, *, include_hash: bool = True) -> str:
    return json.dumps(
        report_to_dict(report, include_hash=include_hash),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def content_hash(report: ValidationReport) -> str:
    return hashlib.sha256(
        canonical_json(report, include_hash=False).encode()
    ).hexdigest()


def save(report: ValidationReport, path: str | Path) -> None:
    report.content_hash = content_hash(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(report) + "\n", encoding="utf-8")


def report_from_dict(payload: dict[str, Any]) -> ValidationReport:
    results = [
        ValidationResult(
            category=item["category"],
            status=ValidationStatus(item["status"]),
            score=item.get("score"),
            checks=item.get("checks", {}),
            warnings=item.get("warnings", []),
            failures=item.get("failures", []),
            metadata=item.get("metadata", {}),
        )
        for item in payload.get("results", [])
    ]
    fields = dict(payload)
    fields["overall_status"] = ValidationStatus(fields["overall_status"])
    fields["results"] = results
    return ValidationReport(**fields)


def load(path: str | Path) -> ValidationReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("validation report must be a JSON object")
    report = report_from_dict(payload)
    if report.content_hash and report.content_hash != content_hash(report):
        raise ValueError("validation report content hash mismatch")
    return report
