"""Executable PyTorch DDP training entrypoint for CINEOS research jobs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from .neural_backend import NeuralModelConfig, TorchCineosFlowModel, _load_torch
from .torch_distributed import DistributedRuntimeConfig, TorchDistributedRuntime


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
        return (
            self.identity[index],
            self.scene[index],
            self.source[index],
            self.target[index],
        )


def run_ddp_training(config: DDPTrainConfig, checkpoint: str | Path) -> str | None:
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
        model = TorchCineosFlowModel(
            NeuralModelConfig(
                feature_dim=config.feature_dim,
                embedding_dim=config.embedding_dim,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                device=str(device),
            )
        )
        ddp = runtime.wrap_model(model.module, device_id=local_rank if torch.cuda.is_available() else None)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=config.learning_rate)
        dataset = SyntheticFlowDataset(torch, config)
        sampler = runtime.distributed_sampler(dataset, shuffle=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)

        completed_steps = 0
        for identity, scene, source, target in loader:
            identity = identity.to(device)
            scene = scene.to(device)
            source = source.to(device)
            target = target.to(device)
            time = torch.full((identity.shape[0], 1), 0.5, device=device)
            interpolated = 0.5 * source + 0.5 * target
            target_velocity = target - source
            optimizer.zero_grad(set_to_none=True)
            predicted = ddp(identity, scene, interpolated, time)
            loss = torch.nn.functional.mse_loss(predicted, target_velocity)
            loss.backward()
            optimizer.step()
            completed_steps += 1
            if completed_steps >= config.steps:
                break

        runtime.barrier()
        if runtime.is_rank_zero():
            destination = Path(checkpoint)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "cineos-ddp-smoke/0.1",
                    "model": ddp.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "world_size": world_size,
                    "steps": completed_steps,
                },
                destination,
            )
            return str(destination)
        return None
    finally:
        runtime.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="CINEOS DDP smoke training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    run_ddp_training(DDPTrainConfig(steps=args.steps), args.checkpoint)


if __name__ == "__main__":
    main()
