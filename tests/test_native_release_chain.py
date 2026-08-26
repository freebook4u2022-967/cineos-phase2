import json

import pytest

from cineos.native_video.release_chain import (
    GENESIS_PREVIOUS_SHA256,
    ReleaseChainEntry,
    ReleaseChainError,
    append_release,
    load_release_chain,
    save_release_chain,
    verify_release_chain,
)


def _sha(char: str) -> str:
    return char * 64


def test_append_release_builds_verified_lineage() -> None:
    first = append_release(
        (),
        release_id="film-v1",
        receipt_sha256=_sha("a"),
        native_model_manifest_sha256=_sha("b"),
    )
    assert first[0].previous_entry_sha256 == GENESIS_PREVIOUS_SHA256

    second = append_release(
        first,
        release_id="film-v2",
        receipt_sha256=_sha("c"),
        native_model_manifest_sha256=_sha("d"),
    )
    assert second[1].previous_entry_sha256 == second[0].entry_sha256
    verify_release_chain(second)


def test_release_chain_rejects_duplicate_release_ids() -> None:
    first = ReleaseChainEntry("same", _sha("a"), _sha("b"))
    second = ReleaseChainEntry(
        "same",
        _sha("c"),
        _sha("d"),
        previous_entry_sha256=first.entry_sha256,
    )
    with pytest.raises(ReleaseChainError, match="duplicate release_id"):
        verify_release_chain((first, second))


def test_release_chain_rejects_broken_predecessor() -> None:
    first = ReleaseChainEntry("v1", _sha("a"), _sha("b"))
    second = ReleaseChainEntry("v2", _sha("c"), _sha("d"), previous_entry_sha256=_sha("e"))
    with pytest.raises(ReleaseChainError, match="predecessor mismatch"):
        verify_release_chain((first, second))


def test_release_chain_round_trip_and_tamper_detection(tmp_path) -> None:
    entries = append_release(
        (),
        release_id="film-v1",
        receipt_sha256=_sha("a"),
        native_model_manifest_sha256=_sha("b"),
    )
    path = save_release_chain(entries, tmp_path / "release-chain.json")
    assert load_release_chain(path) == entries

    document = json.loads(path.read_text(encoding="utf-8"))
    document["entries"][0]["release_id"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReleaseChainError, match="integrity hash mismatch"):
        load_release_chain(path)


def test_release_chain_rejects_malformed_digest() -> None:
    with pytest.raises(ReleaseChainError, match="64-character SHA-256"):
        ReleaseChainEntry("v1", "not-a-digest", _sha("b"))
