"""Manifest-driven real dataset pipeline for CINEOS distributed training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .neural_backend import NeuralModelConfig, _load_torch
from .neural_data import ApprovedManifestPreprocessor
from .training import NativeDatasetManifest


@dataclass(frozen=True, slots=True)
class RealTrainingRecord:
    sample_id: str
    character_id: str
    identity_features: tuple[float, ...]
    scene_features: tuple[float, ...]
    target_latent: tuple[float, ...]


class RealManifestTorchDataset:
    """Torch dataset backed by approved CINEOS manifest files only."""

    def __init__(
        self,
        manifest: NativeDatasetManifest,
        dataset_root: str | Path,
        config: NeuralModelConfig,
    ) -> None:
        self.torch = _load_torch()
        preprocessor = ApprovedManifestPreprocessor(config, Path(dataset_root))
        prepared = preprocessor.prepare_dataset(manifest)
        if not prepared:
            raise ValueError("real training manifest must contain at least one sample")
        characters = {
            sample.sample_id: sample.identity_tags[0]
            for sample in manifest.samples
            if sample.identity_tags
        }
        self.records = tuple(
            RealTrainingRecord(
                sample_id=item.sample_id,
                character_id=characters.get(item.sample_id, item.sample_id),
                identity_features=item.identity_features,
                scene_features=item.scene_features,
                target_latent=item.target_latent,
            )
            for item in prepared
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        torch = self.torch
        identity = torch.tensor(item.identity_features, dtype=torch.float32)
        scene = torch.tensor(item.scene_features, dtype=torch.float32)
        target = torch.tensor(item.target_latent, dtype=torch.float32)
        source = torch.zeros_like(target)
        return identity, scene, source, target, item.character_id


def build_distributed_real_loader(
    dataset: RealManifestTorchDataset,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    shuffle: bool = True,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world_size")
    torch = dataset.torch
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
    return loader, sampler
