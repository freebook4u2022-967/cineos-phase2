from __future__ import annotations

import os

from cineos.native_video import production_readiness as production_readiness_module


def test_readiness_hash_rejects_same_size_post_hash_mutation_with_restored_mtime(
    tmp_path, monkeypatch
):
    """Post-hash writes must fail closed even when size and mtime are preserved."""
    path = tmp_path / "release-audit.json"
    original = b"original\n"
    replacement = b"mutated!\n"
    assert len(original) == len(replacement)
    path.write_bytes(original)
    initial = path.stat()

    real_lstat = production_readiness_module.Path.lstat
    calls = 0

    def mutate_before_final_lstat(candidate):
        nonlocal calls
        if candidate == path:
            calls += 1
            if calls == 2:
                path.write_bytes(replacement)
                os.utime(
                    path,
                    ns=(initial.st_atime_ns, initial.st_mtime_ns),
                )
        return real_lstat(candidate)

    monkeypatch.setattr(
        production_readiness_module.Path,
        "lstat",
        mutate_before_final_lstat,
    )

    digest, blocker = production_readiness_module._sha256_regular_evidence_file(
        path,
        key="release_audit",
    )

    assert calls == 2
    assert digest is None
    assert blocker == (
        "readiness evidence artifact changed during verification: release_audit"
    )
