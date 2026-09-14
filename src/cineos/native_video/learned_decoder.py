"""Checkpoint-backed learned RGB decoder for the native CINEOS video runtime.

This module is the deployment bridge between neural decoder training and film
rendering.  It intentionally accepts the provider-neutral ``Tensor`` used by the
temporal runtime and delegates only pixel reconstruction to CINEOS-owned trained
weights.  No external image/video generation provider is involved.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from cineos.hardware.torch_device import resolve_torch_device
from cineos.native_image.neural_backend import _load_torch
from cineos.native_image.neural_decoder import TorchLatentRGBDecoder
from cineos.native_image.tensor_model import Tensor


def _checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class CheckpointLatentRGBDecoder:
    """Native renderer adapter backed by a trained CINEOS decoder checkpoint.

    Width, height and latent dimensionality are strict deployment invariants.  A
    mismatched temporal model or renderer configuration fails closed instead of
    silently resizing, truncating or padding learned representations.  ``auto``
    deploys to CUDA when PyTorch can execute CUDA workloads, then MPS, then CPU;
    explicit accelerator requests never silently fall back.
    """

    checkpoint_path: str | Path
    device: str = "auto"
    decoder: TorchLatentRGBDecoder = field(init=False, repr=False)
    decoder_id: str = field(init=False)

    def __post_init__(self) -> None:
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"decoder checkpoint does not exist: {checkpoint}")
        self.device = resolve_torch_device(self.device)
        self.decoder = TorchLatentRGBDecoder.load_checkpoint(
            checkpoint,
            device=self.device,
        )
        fingerprint = _checkpoint_fingerprint(checkpoint)
        self.decoder_id = f"cineos-torch-rgb-decoder/0.1@sha256:{fingerprint}"

    @property
    def latent_dim(self) -> int:
        return self.decoder.config.latent_dim

    @property
    def width(self) -> int:
        return self.decoder.width

    @property
    def height(self) -> int:
        return self.decoder.height

    def decode(self, latent: Tensor, *, width: int, height: int) -> bytes:
        if latent.shape != (self.latent_dim,):
            raise ValueError(
                "learned RGB decoder latent shape mismatch: "
                f"expected {(self.latent_dim,)}, got {latent.shape}"
            )
        if width != self.width or height != self.height:
            raise ValueError(
                "learned RGB decoder resolution mismatch: "
                f"checkpoint is {self.width}x{self.height}, requested {width}x{height}"
            )
        torch = _load_torch()
        latent_tensor = torch.tensor(
            latent.values,
            dtype=torch.float32,
            device=self.decoder.device_object,
        )
        with torch.inference_mode():
            frame = self.decoder.decode_frame(latent_tensor)
        expected = width * height * 3
        if len(frame.rgb) != expected:
            raise ValueError(
                "learned RGB decoder produced "
                f"{len(frame.rgb)} bytes; expected {expected}"
            )
        return frame.rgb
