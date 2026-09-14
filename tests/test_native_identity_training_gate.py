import pytest

from cineos.native_image.identity_training_gate import IdentityTrainingGate
from cineos.native_image.training import NativeDatasetManifest, NativeTrainingSample


def _sample(sample_id, views=(), variations=()):
    return NativeTrainingSample(
        sample_id=sample_id,
        image_path=f"{sample_id}.ppm",
        character_reference_paths=("refs/arif.ppm",),
        caption="Arif reference",
        identity_tags=("arif",),
        continuity_tags=("scene-1",),
        metadata={"identity_views": views, "identity_variations": variations},
    )


def test_gate_allows_complete_identity_coverage():
    manifest = NativeDatasetManifest("set", "1")
    manifest.samples = [
        _sample("a", ("front",), ("expression",)),
        _sample("b", ("side",), ("lighting",)),
        _sample("c", ("three_quarter",), ("costume",)),
        _sample("d", ("full_body",), ()),
    ]
    decision = IdentityTrainingGate().evaluate(manifest)
    assert decision.allowed
    assert decision.recommendations == ()


def test_gate_blocks_incomplete_character_and_recommends_shots():
    manifest = NativeDatasetManifest("set", "1")
    manifest.samples = [_sample("a", ("front",), ())]
    decision = IdentityTrainingGate().evaluate(manifest)
    assert not decision.allowed
    recommendation = decision.recommendations[0]
    assert recommendation.character_id == "arif"
    assert "capture side reference" in recommendation.required_shots
    assert "capture lighting variation" in recommendation.required_shots


def test_require_raises_for_production_ineligible_manifest():
    manifest = NativeDatasetManifest("set", "1")
    manifest.samples = [_sample("a", ("front",), ())]
    with pytest.raises(ValueError, match="arif"):
        IdentityTrainingGate().require(manifest)
