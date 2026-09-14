"""Build and quality-gate CINEOS real training manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .training import NativeDatasetManifest, NativeTrainingSample


@dataclass(frozen=True, slots=True)
class DatasetQualityPolicy:
    minimum_bytes: int = 16
    require_identity_tags: bool = True
    require_continuity_tags: bool = True


@dataclass(frozen=True, slots=True)
class DatasetQualityIssue:
    sample_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    manifest: NativeDatasetManifest
    rejected: tuple[DatasetQualityIssue, ...]


class RealTrainingDatasetBuilder:
    def __init__(
        self, dataset_root: str | Path, policy: DatasetQualityPolicy | None = None
    ) -> None:
        self.root = Path(dataset_root)
        self.policy = policy or DatasetQualityPolicy()

    def build(
        self,
        name: str,
        version: str,
        samples: list[NativeTrainingSample],
    ) -> DatasetBuildResult:
        manifest = NativeDatasetManifest(name=name, version=version)
        rejected: list[DatasetQualityIssue] = []
        seen_hashes: set[str] = set()
        for sample in samples:
            reason = self._validate(sample, seen_hashes)
            if reason is not None:
                rejected.append(DatasetQualityIssue(sample.sample_id, reason))
                continue
            image = self.root / sample.image_path
            seen_hashes.add(self._digest(image))
            manifest.add(sample)
        return DatasetBuildResult(manifest, tuple(rejected))

    def _validate(
        self, sample: NativeTrainingSample, seen_hashes: set[str]
    ) -> str | None:
        image = self.root / sample.image_path
        if not image.is_file():
            return "training image missing"
        if image.stat().st_size < self.policy.minimum_bytes:
            return "training image too small"
        if any(
            not (self.root / path).is_file()
            for path in sample.character_reference_paths
        ):
            return "character reference missing"
        if self.policy.require_identity_tags and not sample.identity_tags:
            return "identity metadata missing"
        if self.policy.require_continuity_tags and not sample.continuity_tags:
            return "continuity metadata missing"
        digest = self._digest(image)
        if digest in seen_hashes:
            return "duplicate training image"
        return None

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
