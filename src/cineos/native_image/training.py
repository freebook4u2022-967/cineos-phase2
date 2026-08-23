"""Training contracts for future CINEOS-owned learned frame models."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

DATASET_MANIFEST_SCHEMA = "cineos-native-training-dataset/0.1"
CHECKPOINT_MANIFEST_SCHEMA = "cineos-native-model-checkpoint/0.1"


@dataclass(frozen=True, slots=True)
class NativeTrainingSample:
    """One supervised/multimodal frame-training example."""

    sample_id: str
    image_path: str
    character_reference_paths: tuple[str, ...]
    caption: str
    scene_description: str
    identity_tags: tuple[str, ...] = ()
    continuity_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("training sample requires sample_id")
        if not self.image_path.strip():
            raise ValueError("training sample requires image_path")
        if not self.character_reference_paths:
            raise ValueError("training sample requires character references")
        if not self.caption.strip():
            raise ValueError("training sample requires caption")


@dataclass(slots=True)
class NativeDatasetManifest:
    """Versioned dataset manifest for reproducible CINEOS training runs."""

    name: str
    version: str
    samples: list[NativeTrainingSample] = field(default_factory=list)
    schema: str = DATASET_MANIFEST_SCHEMA

    def add(self, sample: NativeTrainingSample) -> None:
        if any(existing.sample_id == sample.sample_id for existing in self.samples):
            raise ValueError(f"duplicate training sample_id: {sample.sample_id}")
        self.samples.append(sample)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sample_count"] = len(self.samples)
        payload["content_hash"] = self.content_hash()
        return payload

    def content_hash(self) -> str:
        payload = {
            "schema": self.schema,
            "name": self.name,
            "version": self.version,
            "samples": [asdict(sample) for sample in self.samples],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination


@dataclass(frozen=True, slots=True)
class NativeCheckpointManifest:
    """Metadata contract for a learned CINEOS model checkpoint."""

    model_name: str
    model_version: str
    dataset_hash: str
    training_step: int
    component_files: dict[str, str]
    metrics: dict[str, float] = field(default_factory=dict)
    schema: str = CHECKPOINT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.training_step < 0:
            raise ValueError("training_step must be non-negative")
        required = {"identity_encoder", "scene_encoder", "sampler", "decoder"}
        missing = required - set(self.component_files)
        if missing:
            raise ValueError(
                f"checkpoint missing required components: {sorted(missing)}"
            )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination


class LearnedIdentityEncoder(Protocol):
    def encode_references(self, references: tuple[str, ...]) -> Any: ...


class LearnedSceneEncoder(Protocol):
    def encode_scene(self, caption: str, scene_description: str) -> Any: ...


class LearnedLatentSampler(Protocol):
    def sample(self, identity_state: Any, scene_state: Any, *, seed: int) -> Any: ...


class LearnedRGBDecoder(Protocol):
    def decode(self, latent: Any, *, width: int, height: int) -> bytes: ...
