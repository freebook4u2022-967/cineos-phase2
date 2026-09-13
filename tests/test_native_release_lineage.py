from __future__ import annotations

from dataclasses import replace

import pytest

from cineos.native_video.audited_release import AuditedProductionRelease
from cineos.native_video.release_lineage import (
    GENESIS_PREVIOUS_RELEASE,
    ProductionReleaseLineageEntry,
    create_release_lineage_entry,
    verify_release_lineage,
)
from cineos.native_video.release_receipt import ProductionReleaseError


def _release(
    *, seed: str, model: str = "c", runtime: str = "d"
) -> AuditedProductionRelease:
    return AuditedProductionRelease(
        release_bundle_sha256=seed * 64,
        audit_record_sha256="b" * 64,
        movie_sha256="e" * 64,
        model_fingerprint=model * 64,
        runtime_fingerprint=runtime * 64,
    )


def test_genesis_and_same_composition_release_form_gap_free_chain() -> None:
    first = create_release_lineage_entry(_release(seed="a"))
    second = create_release_lineage_entry(_release(seed="f"), previous=first)

    assert first.sequence == 0
    assert first.previous_release_fingerprint == GENESIS_PREVIOUS_RELEASE
    assert second.sequence == 1
    assert second.previous_release_fingerprint == first.release_fingerprint
    assert second.migration_fingerprint == GENESIS_PREVIOUS_RELEASE
    assert verify_release_lineage([first, second]) == (first, second)


def test_model_or_runtime_upgrade_requires_explicit_migration_fingerprint() -> None:
    first = create_release_lineage_entry(_release(seed="a"))
    upgraded = _release(seed="f", model="1", runtime="2")

    with pytest.raises(ProductionReleaseError, match="migration_fingerprint"):
        create_release_lineage_entry(upgraded, previous=first)

    second = create_release_lineage_entry(
        upgraded,
        previous=first,
        migration_fingerprint="9" * 64,
    )
    assert second.migration_fingerprint == "9" * 64
    verify_release_lineage([first, second])


def test_same_composition_rejects_spurious_migration() -> None:
    first = create_release_lineage_entry(_release(seed="a"))

    with pytest.raises(ProductionReleaseError, match="without model/runtime"):
        create_release_lineage_entry(
            _release(seed="f"),
            previous=first,
            migration_fingerprint="9" * 64,
        )


def test_verifier_rejects_history_rewrite_or_sequence_gap() -> None:
    first = create_release_lineage_entry(_release(seed="a"))
    second = create_release_lineage_entry(_release(seed="f"), previous=first)

    rewritten = replace(second, previous_release_fingerprint="7" * 64)
    with pytest.raises(ProductionReleaseError, match="previous-release mismatch"):
        verify_release_lineage([first, rewritten])

    skipped = replace(second, sequence=3)
    with pytest.raises(ProductionReleaseError, match="sequence gap"):
        verify_release_lineage([first, skipped])


def test_verifier_rejects_duplicate_release_fingerprint() -> None:
    first = create_release_lineage_entry(_release(seed="a"))
    duplicate = ProductionReleaseLineageEntry(
        sequence=1,
        release_fingerprint=first.release_fingerprint,
        previous_release_fingerprint=first.release_fingerprint,
        model_fingerprint=first.model_fingerprint,
        runtime_fingerprint=first.runtime_fingerprint,
        migration_fingerprint=GENESIS_PREVIOUS_RELEASE,
    )

    with pytest.raises(ProductionReleaseError, match="duplicate release"):
        verify_release_lineage([first, duplicate])


def test_lineage_digest_fields_fail_closed() -> None:
    with pytest.raises(ProductionReleaseError, match="release_fingerprint"):
        ProductionReleaseLineageEntry(
            sequence=0,
            release_fingerprint="bad",
            previous_release_fingerprint=GENESIS_PREVIOUS_RELEASE,
            model_fingerprint="c" * 64,
            runtime_fingerprint="d" * 64,
            migration_fingerprint=GENESIS_PREVIOUS_RELEASE,
        )
