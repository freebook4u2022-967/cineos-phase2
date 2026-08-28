"""Trainable neural encoders for CINEOS real-data experiments."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_pixels import decode_rgb_image
from .neural_backend import NeuralModelConfig, _load_torch

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


def _image_file_to_tensor(path: str | Path, config: NeuralModelConfig, *, device: Any):
    """Decode real RGB pixels into the neural backend's deterministic input tensor."""
    torch = _load_torch()
    decoded = decode_rgb_image(path, image_size=config.image_size)
    return torch.tensor(decoded.values, dtype=torch.float32, device=device)


def _decoded_pixel_digest(values: tuple[float, ...]) -> bytes:
    """Return a container-independent digest for normalized decoded RGB pixels.

    Approved character references can arrive as different PNG/JPEG files that
    decode to the same resized RGB evidence. Identity aggregation must not silently
    overweight a duplicated pose simply because the same visual reference was
    supplied more than once. Quantizing the normalized values back to their exact
    8-bit channel representation makes the digest independent of container bytes
    and stable across platforms.
    """

    payload = bytes(
        max(0, min(255, round((value + 1.0) * 127.5))) for value in values
    )
    return hashlib.blake2b(payload, digest_size=16).digest()


def _stable_token_id(token: str, vocabulary_size: int) -> int:
    if vocabulary_size <= 1:
        raise ValueError("vocabulary_size must be greater than 1")
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return 1 + (int.from_bytes(digest, "big") % (vocabulary_size - 1))


def _token_ids(text: str, *, vocabulary_size: int, max_tokens: int) -> tuple[int, ...]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    tokens = _TOKEN_RE.findall(text.casefold())
    if not tokens:
        return (0,)
    return tuple(
        _stable_token_id(token, vocabulary_size) for token in tokens[:max_tokens]
    )


@dataclass
class TorchImageLatentEncoder:
    """Small VAE-style image encoder producing mean/logvar latent parameters."""

    config: NeuralModelConfig
    device: str = "cpu"

    def __post_init__(self) -> None:
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        self.input_dim = self.config.image_size * self.config.image_size * 3
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_dim),
            nn.SiLU(),
        ).to(self.device_object)
        self.mean_head = nn.Linear(
            self.config.hidden_dim,
            self.config.latent_dim,
        ).to(self.device_object)
        self.logvar_head = nn.Linear(
            self.config.hidden_dim,
            self.config.latent_dim,
        ).to(self.device_object)

    def encode_file(self, path: str | Path):
        features = _image_file_to_tensor(
            path,
            self.config,
            device=self.device_object,
        )
        hidden = self.backbone(features)
        return self.mean_head(hidden), self.logvar_head(hidden)

    def sample(self, mean, logvar, *, deterministic: bool = False):
        torch = _load_torch()
        if deterministic:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + (torch.randn_like(std) * std)


@dataclass
class TorchCharacterReferenceEncoder:
    """Aggregate approved character image pixels into trainable identity features.

    Aggregation is deliberately order-invariant and content-deduplicated. Repeated
    files that decode to identical resized RGB pixels contribute only once, so an
    accidental duplicate approved reference cannot bias identity conditioning
    toward one pose, crop or expression.
    """

    config: NeuralModelConfig
    device: str = "cpu"

    def __post_init__(self) -> None:
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        self.input_dim = self.config.image_size * self.config.image_size * 3
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.feature_dim),
        ).to(self.device_object)

    def encode_files(self, paths: tuple[str | Path, ...]):
        torch = _load_torch()
        if not paths:
            raise ValueError("character reference encoder requires references")

        encoded = []
        seen_pixels: set[bytes] = set()
        for path in paths:
            decoded = decode_rgb_image(path, image_size=self.config.image_size)
            digest = _decoded_pixel_digest(decoded.values)
            if digest in seen_pixels:
                continue
            seen_pixels.add(digest)
            features = torch.tensor(
                decoded.values,
                dtype=torch.float32,
                device=self.device_object,
            )
            encoded.append(self.network(features))

        if not encoded:
            raise RuntimeError("character reference encoder produced no unique evidence")
        return torch.stack(encoded, dim=0).mean(dim=0)


@dataclass
class TorchSceneTextEncoder:
    """Trainable token encoder for scene, continuity and directing language.

    The encoder is intentionally CINEOS-owned and dependency-light. It uses stable
    hashed token IDs feeding a learned embedding table, then mean-pools tokens and
    projects them into the native feature space. This is materially stronger than
    repeating raw UTF-8 bytes while remaining checkpointable and trainable from
    CINEOS datasets. A future pretrained language backbone can replace this module
    behind the same ``encode`` contract.
    """

    config: NeuralModelConfig
    device: str = "cpu"
    vocabulary_size: int = 8192
    max_tokens: int = 128

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 1:
            raise ValueError("vocabulary_size must be greater than 1")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        embedding_dim = max(8, int(self.config.embedding_dim))
        self.embedding = nn.Embedding(
            self.vocabulary_size,
            embedding_dim,
            padding_idx=0,
        ).to(self.device_object)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.feature_dim),
        ).to(self.device_object)

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Return deterministic token IDs for checkpoint-safe training/debugging."""
        return _token_ids(
            text,
            vocabulary_size=self.vocabulary_size,
            max_tokens=self.max_tokens,
        )

    def encode(self, caption: str, scene_description: str, continuity: tuple[str, ...]):
        torch = _load_torch()
        segments = (
            f"[caption] {caption}",
            f"[scene] {scene_description}",
            *(f"[continuity] {item}" for item in continuity),
        )
        ids = self.tokenize(" ".join(segments))
        token_tensor = torch.tensor(ids, dtype=torch.long, device=self.device_object)
        pooled = self.embedding(token_tensor).mean(dim=0)
        return self.network(pooled)
