"""Executable PyTorch DDP training entrypoint for CINEOS research jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from .identity_loss import IdentityLossConfig, TorchIdentityConsistencyLoss
from .identity_training import TorchIdentityProjection
from .neural_backend import NeuralModelConfig, TorchCineosFlowModel, _load_torch
from .real_dataset import RealManifestTorchDataset
from .torch_distributed import DistributedRuntimeConfig, TorchDistributedRuntime
from .training import NativeDatasetManifest, NativeTrainingSample


@dataclass(frozen=True, slots=True)
class DDPTrainConfig:
    samples: int = 32
    batch_size: int = 4
    steps: int = 2
    feature_dim: int = 8
    embedding_dim: int = 8
    latent_dim: int = 16
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    identity_loss_weight: float = 1.0


class SyntheticFlowDataset:
    def __init__(self, torch, config: DDPTrainConfig) -> None:
        generator = torch.Generator().manual_seed(1234)
        self.identity = torch.randn(config.samples, config.feature_dim, generator=generator)
        self.scene = torch.randn(config.samples, config.feature_dim, generator=generator)
        self.source = torch.randn(config.samples, config.latent_dim, generator=generator)
        self.target = torch.randn(config.samples, config.latent_dim, generator=generator)

    def __len__(self) -> int:
        return len(self.identity)

    def __getitem__(self, index: int):
        return self.identity[index], self.scene[index], self.source[index], self.target[index]


def load_dataset_manifest(path: str | Path) -> NativeDatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [
        NativeTrainingSample(
            sample_id=item["sample_id"],
            image_path=item["image_path"],
            character_reference_paths=tuple(item["character_reference_paths"]),
            caption=item["caption"],
            scene_description=item.get("scene_description", ""),
            identity_tags=tuple(item.get("identity_tags", ())),
            continuity_tags=tuple(item.get("continuity_tags", ())),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload.get("samples", ())
    ]
    return NativeDatasetManifest(
        name=payload["name"],
        version=payload["version"],
        samples=samples,
        schema=payload.get("schema", "cineos-native-training-dataset/0.1"),
    )


def load_identity_anchors(path: str | Path) -> dict[str, tuple[float, ...]]:
    """Load normalized character anchors from a compact JSON identity bank.

    Parsing and validation remain available without PyTorch so manifests and
    identity metadata can be inspected by lightweight tooling and CI jobs.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source = payload.get("characters", payload)
    if not isinstance(source, dict) or not source:
        raise ValueError("identity bank must contain at least one character")
    anchors: dict[str, tuple[float, ...]] = {}
    width: int | None = None
    for character_id, raw in source.items():
        vector = raw.get("vector") if isinstance(raw, dict) else raw
        values = tuple(float(value) for value in vector)
        if not values:
            raise ValueError(f"invalid identity anchor: {character_id}")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ValueError(f"invalid identity anchor: {character_id}")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("identity anchors must share the same dimension")
        anchors[str(character_id)] = tuple(value / norm for value in values)
    return anchors


def validate_identity_coverage(dataset, anchors: dict[str, tuple[float, ...]]) -> dict[str, int]:
    """Validate anchor coverage for every real training record before training starts.

    Identity supervision must fail fast rather than discovering a missing
    character anchor after DDP workers have initialized and consumed batches.
    The returned counts are persisted in checkpoints for auditability.
    """
    counts: dict[str, int] = {}
    for record in dataset.records:
        character_id = str(record.character_id)
        counts[character_id] = counts.get(character_id, 0) + 1
    missing = sorted(character_id for character_id in counts if character_id not in anchors)
    if missing:
        raise ValueError(f"identity anchors missing for: {', '.join(missing)}")
    return counts


def _anchors_for_batch(torch, character_ids, anchors, device):
    missing = sorted({character_id for character_id in character_ids if character_id not in anchors})
    if missing:
        raise ValueError(f"identity anchors missing for: {', '.join(missing)}")
    return torch.tensor([anchors[item] for item in character_ids], dtype=torch.float32, device=device)


def _load_resume(torch, path: Path, model, projection, optimizer, device) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    if projection is not None and payload.get("identity_projection") is not None:
        projection.load_state_dict(payload["identity_projection"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("steps", 0)), int(payload.get("sampler_epoch", 0))


def run_ddp_training(
    config: DDPTrainConfig,
    checkpoint: str | Path,
    *,
    resume: bool = False,
    manifest: str | Path | None = None,
    dataset_root: str | Path | None = None,
    identity_bank: str | Path | None = None,
) -> str | None:
    torch = _load_torch()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    runtime = TorchDistributedRuntime(
        DistributedRuntimeConfig(rank=rank, world_size=world_size, backend=backend)
    )
    runtime.initialize()
    try:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        model_config = NeuralModelConfig(
            feature_dim=config.feature_dim,
            embedding_dim=config.embedding_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            device=str(device),
        )
        model = TorchCineosFlowModel(model_config)
        ddp = runtime.wrap_model(
            model.module, device_id=local_rank if torch.cuda.is_available() else None
        )

        anchors = load_identity_anchors(identity_bank) if identity_bank is not None else None
        projection_ddp = None
        identity_loss = None
        parameters = list(ddp.parameters())
        if anchors is not None:
            identity_dim = len(next(iter(anchors.values())))
            projection = TorchIdentityProjection(config.latent_dim, identity_dim, str(device))
            projection_ddp = runtime.wrap_model(
                projection.module,
                device_id=local_rank if torch.cuda.is_available() else None,
            )
            parameters.extend(projection_ddp.parameters())
            identity_loss = TorchIdentityConsistencyLoss(
                IdentityLossConfig(weight=config.identity_loss_weight)
            )

        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
        checkpoint_path = Path(checkpoint)
        completed_steps, sampler_epoch = (0, 0)
        if resume:
            completed_steps, sampler_epoch = _load_resume(
                torch,
                checkpoint_path,
                ddp.module,
                projection_ddp.module if projection_ddp is not None else None,
                optimizer,
                device,
            )

        identity_sample_counts = None
        if manifest is not None:
            if dataset_root is None:
                raise ValueError("--dataset-root is required when --manifest is used")
            dataset = RealManifestTorchDataset(
                load_dataset_manifest(manifest), dataset_root, model_config
            )
            real_mode = True
            if anchors is not None:
                identity_sample_counts = validate_identity_coverage(dataset, anchors)
        else:
            if anchors is not None:
                raise ValueError("--identity-bank requires --manifest real dataset mode")
            dataset = SyntheticFlowDataset(torch, config)
            real_mode = False
        sampler = runtime.distributed_sampler(dataset, shuffle=True)
        sampler.set_epoch(sampler_epoch)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler
        )

        last_flow_loss = None
        last_identity_loss = None
        target_steps = completed_steps + config.steps
        for batch in loader:
            identity, scene, source, target = batch[:4]
            identity = identity.to(device)
            scene = scene.to(device)
            source = source.to(device)
            target = target.to(device)
            time = torch.full((identity.shape[0], 1), 0.5, device=device)
            interpolated = 0.5 * source + 0.5 * target
            target_velocity = target - source
            optimizer.zero_grad(set_to_none=True)
            predicted = ddp(identity, scene, interpolated, time)
            flow_loss = torch.nn.functional.mse_loss(predicted, target_velocity)
            loss = flow_loss
            identity_component = None
            if anchors is not None and projection_ddp is not None and identity_loss is not None:
                character_ids = tuple(batch[4])
                anchor_tensor = _anchors_for_batch(torch, character_ids, anchors, device)
                predicted_identity = projection_ddp(predicted)
                identity_component = identity_loss(predicted_identity, anchor_tensor)
                loss = loss + identity_component
            loss.backward()
            optimizer.step()
            last_flow_loss = float(flow_loss.detach().cpu())
            last_identity_loss = (
                float(identity_component.detach().cpu()) if identity_component is not None else None
            )
            completed_steps += 1
            if completed_steps >= target_steps:
                break

        sampler_epoch += 1
        runtime.barrier()
        if runtime.is_rank_zero():
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "cineos-ddp-training/0.5",
                    "dataset_mode": "real" if real_mode else "synthetic",
                    "manifest": str(manifest) if manifest is not None else None,
                    "identity_bank": str(identity_bank) if identity_bank is not None else None,
                    "identity_sample_counts": identity_sample_counts,
                    "model": ddp.module.state_dict(),
                    "identity_projection": (
                        projection_ddp.module.state_dict() if projection_ddp is not None else None
                    ),
                    "optimizer": optimizer.state_dict(),
                    "world_size": world_size,
                    "steps": completed_steps,
                    "sampler_epoch": sampler_epoch,
                    "last_flow_loss": last_flow_loss,
                    "last_identity_loss": last_identity_loss,
                },
                checkpoint_path,
            )
            return str(checkpoint_path)
        return None
    finally:
        runtime.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="CINEOS distributed training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-root")
    parser.add_argument("--identity-bank")
    args = parser.parse_args()
    run_ddp_training(
        DDPTrainConfig(steps=args.steps),
        args.checkpoint,
        resume=args.resume,
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        identity_bank=args.identity_bank,
    )


if __name__ == "__main__":
    main()
