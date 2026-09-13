from cineos.native_image.trainable_model import (
    NativeTrainableModel,
    NativeTrainingLoop,
    SGDOptimizer,
)
from cineos.native_image.training import NativeDatasetManifest, NativeTrainingSample


def _sample(sample_id: str, target: float) -> NativeTrainingSample:
    return NativeTrainingSample(
        sample_id=sample_id,
        image_path=f"frames/{sample_id}.ppm",
        character_reference_paths=("refs/hero-front.png",),
        caption="hero in a night interior",
        scene_description="dim apartment with practical light",
        identity_tags=("hero",),
        continuity_tags=("same wardrobe",),
        metadata={"training_target": target},
    )


def test_training_step_updates_parameters():
    loop = NativeTrainingLoop(
        model=NativeTrainableModel(),
        optimizer=SGDOptimizer(learning_rate=0.1),
    )
    before = (
        loop.model.parameters.identity_scale,
        loop.model.parameters.scene_scale,
        loop.model.parameters.bias,
    )
    result = loop.train_sample(_sample("one", 0.8))
    after = (
        loop.model.parameters.identity_scale,
        loop.model.parameters.scene_scale,
        loop.model.parameters.bias,
    )

    assert result.step == 1
    assert result.loss >= 0.0
    assert after != before


def test_repeated_training_reduces_loss_on_same_sample():
    loop = NativeTrainingLoop(optimizer=SGDOptimizer(learning_rate=0.05))
    sample = _sample("repeat", 0.75)
    losses = [loop.train_sample(sample).loss for _ in range(20)]

    assert losses[-1] < losses[0]
    assert loop.step == 20


def test_epoch_checkpoint_and_resume(tmp_path):
    dataset = NativeDatasetManifest(name="tiny", version="0.1")
    dataset.add(_sample("a", 0.4))
    dataset.add(_sample("b", -0.2))
    loop = NativeTrainingLoop(optimizer=SGDOptimizer(learning_rate=0.03))
    results = loop.train_epoch(dataset)
    checkpoint = loop.save_checkpoint(tmp_path / "trainable.json")

    resumed = NativeTrainingLoop.load_checkpoint(checkpoint)
    resumed_result = resumed.train_sample(_sample("c", 0.6))

    assert len(results) == 2
    assert resumed.step == 3
    assert resumed_result.step == 3
    assert resumed.optimizer.learning_rate == 0.03
