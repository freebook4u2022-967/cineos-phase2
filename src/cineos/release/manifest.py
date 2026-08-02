"""Content-addressed release manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .exceptions import ReleaseError
from .versioning import is_semantic_version


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    release_version: str
    commit_sha: str
    build_timestamp: str
    python_version: str
    supported_operating_systems: tuple[str, ...]
    package_dependencies: tuple[str, ...] = ()
    optional_dependency_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    supported_renderer_plugins: tuple[str, ...] = ("preview",)
    known_limitations: tuple[str, ...] = ()
    benchmark_summary: dict[str, object] = field(default_factory=dict)
    test_summary: dict[str, object] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    licensing_metadata: dict[str, str] = field(
        default_factory=lambda: {"license": "MIT"}
    )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def validate(self) -> tuple[str, ...]:
        errors = []
        if not is_semantic_version(self.release_version):
            errors.append("release version is not semantic")
        if len(self.commit_sha) < 7:
            errors.append("commit SHA is invalid")
        if not self.build_timestamp:
            errors.append("build timestamp is required")
        if not self.supported_operating_systems:
            errors.append("supported operating systems are required")
        return tuple(errors)


def save_manifest(manifest: ReleaseManifest, path: Path) -> Path:
    payload = {**asdict(manifest), "content_hash": manifest.content_hash}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_manifest(path: Path) -> ReleaseManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = raw.pop("content_hash", None)
    raw["supported_operating_systems"] = tuple(raw["supported_operating_systems"])
    for key in (
        "package_dependencies",
        "supported_renderer_plugins",
        "known_limitations",
    ):
        raw[key] = tuple(raw.get(key, ()))
    raw["optional_dependency_groups"] = {
        k: tuple(v) for k, v in raw.get("optional_dependency_groups", {}).items()
    }
    manifest = ReleaseManifest(**raw)
    if manifest.validate():
        raise ReleaseError("; ".join(manifest.validate()))
    if expected != manifest.content_hash:
        raise ReleaseError("release manifest content hash mismatch")
    return manifest
