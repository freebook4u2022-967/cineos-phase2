"""Neural latent decoding and visual comparison artifacts for CINEOS experiments.

The decoder checkpoint contract in this module is intentionally independent from
film orchestration.  Training can therefore promote a decoder artifact into the
native video runtime without serialising arbitrary Python objects or coupling the
renderer to a particular trainer implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from .neural_backend import NeuralModelConfig, _load_torch

_DECODER_CHECKPOINT_SCHEMA = "cineos-torch-rgb-decoder-checkpoint/0.1"


@dataclass(frozen=True, slots=True)
class DecodedRGBFrame:
    width: int
    height: int
    rgb: bytes

    def save_ppm(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            f"P6\n{self.width} {self.height}\n255\n".encode("ascii") + self.rgb
        )
        return destination


@dataclass(slots=True)
class TorchLatentRGBDecoder:
    """Small trainable decoder turning native neural latents into RGB pixels.

    Checkpoints contain only a schema marker, primitive configuration and tensor
    state.  They are the stable hand-off between decoder training and the native
    film renderer; callers never need to pickle the decoder object itself.
    """

    config: NeuralModelConfig
    width: int = 32
    height: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("decoder dimensions must be positive")
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        output_dim = self.width * self.height * 3
        self.network = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, output_dim),
            nn.Sigmoid(),
        ).to(self.device_object)

    def decode(self, latent):
        return self.network(latent.to(self.device_object))

    def decode_frame(self, latent) -> DecodedRGBFrame:
        pixels = self.decode(latent).detach().cpu().reshape(-1)
        rgb = bytes(int(max(0.0, min(1.0, float(value))) * 255) for value in pixels)
        return DecodedRGBFrame(self.width, self.height, rgb)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist deployable decoder weights and architecture metadata."""
        torch = _load_torch()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": _DECODER_CHECKPOINT_SCHEMA,
                "config": asdict(self.config),
                "width": self.width,
                "height": self.height,
                "network": self.network.state_dict(),
            },
            destination,
        )
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> Self:
        """Restore a decoder from the explicit CINEOS checkpoint schema."""
        torch = _load_torch()
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"decoder checkpoint does not exist: {checkpoint}")
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            payload = torch.load(checkpoint, map_location=device)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _DECODER_CHECKPOINT_SCHEMA
        ):
            raise ValueError("unsupported torch RGB decoder checkpoint schema")
        raw_config = payload.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError("decoder checkpoint is missing model configuration")
        decoder = cls(
            NeuralModelConfig(**raw_config),
            width=int(payload["width"]),
            height=int(payload["height"]),
            device=device,
        )
        network_state = payload.get("network")
        if not isinstance(network_state, dict):
            raise ValueError("decoder checkpoint is missing network weights")
        decoder.network.load_state_dict(network_state)
        decoder.network.eval()
        return decoder


@dataclass(frozen=True, slots=True)
class ImageComparisonArtifacts:
    reconstruction_path: str
    generated_path: str


def save_latent_comparison(
    decoder: TorchLatentRGBDecoder,
    target_latent,
    generated_latent,
    output_dir: str | Path,
) -> ImageComparisonArtifacts:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reconstruction = decoder.decode_frame(target_latent).save_ppm(
        destination / "reconstruction.ppm"
    )
    generated = decoder.decode_frame(generated_latent).save_ppm(
        destination / "generated.ppm"
    )
    return ImageComparisonArtifacts(str(reconstruction), str(generated))
