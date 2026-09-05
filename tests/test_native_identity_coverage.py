import pytest

from cineos.native_image.identity_coverage import IdentityCoverageAnalyzer
from cineos.native_image.training import NativeDatasetManifest, NativeTrainingSample


def _sample(sample_id, character, views=(), variations=()):
    return NativeTrainingSample(
        sample_id=sample_id,
        image_path=f"{sample_id}.ppm",
        character_reference_paths=(f"refs/{character}.ppm",),
        caption="character reference coverage",
        identity_tags=(character,),
        continuity_tags=("scene-1",),
        metadata={"identity_views": views, "identity_variations": variations},
    )


def test_complete_character_reference_set_scores_one():
    manifest = NativeDatasetManifest("identity", "1")
    manifest.samples = [
        _sample("a", "arif", ("front",), ("expression",)),
        _sample("b", "arif", ("side",), ("lighting",)),
        _sample("c", "arif", ("three_quarter",), ("costume",)),
        _sample("d", "arif", ("full_body",), ()),
    ]
    result = IdentityCoverageAnalyzer().analyze(manifest).characters[0]
    assert result.coverage_score == pytest.approx(1.0)
    assert result.missing_views == ()
    assert result.missing_variations == ()


def test_incomplete_reference_set_reports_missing_coverage():
    manifest = NativeDatasetManifest("identity", "1")
    manifest.samples = [_sample("a", "hana", ("front",), ())]
    result = IdentityCoverageAnalyzer().analyze(manifest).characters[0]
    assert "side" in result.missing_views
    assert "lighting" in result.missing_variations
    assert result.coverage_score < 0.5


def test_report_averages_multiple_character_scores():
    manifest = NativeDatasetManifest("identity", "1")
    manifest.samples = [
        _sample("a", "arif", ("front",), ("expression",)),
        _sample("b", "hana", ("front", "side"), ("lighting",)),
    ]
    report = IdentityCoverageAnalyzer().analyze(manifest)
    assert len(report.characters) == 2
    assert 0.0 < report.mean_coverage_score < 1.0
