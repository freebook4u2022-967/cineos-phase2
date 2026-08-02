import pytest

from cineos.benchmarks import (
    BenchmarkRunner,
    Metric,
    MetricStatus,
    alpha_suite,
    create_baseline,
)
from cineos.benchmarks.exceptions import BenchmarkError
from cineos.benchmarks.serializer import dumps
from cineos.release.manifest import ReleaseManifest, load_manifest, save_manifest
from cineos.release.packaging import checksum, verify_checksums
from cineos.release.validator import evaluate_gates


def test_suite_and_serialization_are_deterministic(tmp_path):
    suite = alpha_suite()
    assert len(suite.cases) == 13
    assert suite.content_hash == alpha_suite().content_hash
    report = BenchmarkRunner().run(suite, tmp_path, mandatory_only=True)
    assert dumps(report) == dumps(report)
    assert report.passed


def test_baseline_is_never_overwritten(tmp_path):
    report = BenchmarkRunner().run(alpha_suite(), tmp_path / "run", mandatory_only=True)
    path = tmp_path / "baseline.json"
    create_baseline(report, path)
    with pytest.raises(BenchmarkError):
        create_baseline(report, path)


def test_metric_state_requires_a_value():
    with pytest.raises(ValueError):
        Metric("peak_ram", status=MetricStatus.MEASURED)


def test_manifest_hash_and_checksums(tmp_path):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"wheel")
    manifest = ReleaseManifest(
        "0.1.0-alpha.1",
        "abcdef123",
        "2026-08-02T00:00:00Z",
        "3.12",
        ("Linux",),
        checksums={artifact.name: checksum(artifact)},
    )
    path = save_manifest(manifest, tmp_path / "release.json")
    assert load_manifest(path).content_hash == manifest.content_hash
    assert verify_checksums(manifest.checksums, tmp_path) == ()
    artifact.write_bytes(b"changed")
    assert verify_checksums(manifest.checksums, tmp_path) == (artifact.name,)


def test_release_gates_fail_closed():
    manifest = ReleaseManifest("0.1.0-alpha.1", "abcdef123", "now", "3.12", ("Linux",))
    assert not evaluate_gates(manifest, {}).approved
