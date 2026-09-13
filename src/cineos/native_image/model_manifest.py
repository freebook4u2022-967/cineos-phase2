"""Versioned native-model manifests and safe activation/rollback support.

This module keeps CINEOS-owned learned artifacts evolvable without silently loading
incompatible checkpoints. It intentionally uses only standard-library primitives
so the compatibility gate is available in training, rendering, and release tools.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

NATIVE_MODEL_MANIFEST_SCHEMA = "cineos-native-model-manifest/1"
NATIVE_MODEL_REGISTRY_SCHEMA = "cineos-native-model-registry/1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)


class ModelManifestError(ValueError):
    """Raised when a model manifest or registry violates its contract."""


@dataclass(frozen=True, slots=True)
class NativeModelComponent:
    """One independently versioned artifact used by native inference."""

    name: str
    version: str
    artifact_sha256: str
    contract_version: int = 1

    def validate(self) -> None:
        if not self.name.strip():
            raise ModelManifestError("component name must not be empty")
        if _SEMVER_RE.fullmatch(self.version) is None:
            raise ModelManifestError(f"invalid component version: {self.version}")
        if _SHA256_RE.fullmatch(self.artifact_sha256) is None:
            raise ModelManifestError(
                f"component {self.name} artifact_sha256 must be lowercase SHA-256"
            )
        if self.contract_version < 1:
            raise ModelManifestError("component contract_version must be >= 1")


@dataclass(frozen=True, slots=True)
class NativeModelManifest:
    """Immutable identity and compatibility contract for a native model."""

    model_id: str
    model_version: str
    runtime_contract_version: int
    components: tuple[NativeModelComponent, ...]
    metadata: dict[str, str] = field(default_factory=dict)
    schema: str = NATIVE_MODEL_MANIFEST_SCHEMA

    def validate(self) -> None:
        if self.schema != NATIVE_MODEL_MANIFEST_SCHEMA:
            raise ModelManifestError(f"unsupported manifest schema: {self.schema}")
        if not self.model_id.strip():
            raise ModelManifestError("model_id must not be empty")
        if _SEMVER_RE.fullmatch(self.model_version) is None:
            raise ModelManifestError(f"invalid model version: {self.model_version}")
        if self.runtime_contract_version < 1:
            raise ModelManifestError("runtime_contract_version must be >= 1")
        if not self.components:
            raise ModelManifestError("at least one model component is required")
        names: set[str] = set()
        for component in self.components:
            component.validate()
            if component.name in names:
                raise ModelManifestError(f"duplicate component: {component.name}")
            names.add(component.name)

    def canonical_payload(self) -> dict[str, object]:
        self.validate()
        components = [asdict(component) for component in self.components]
        components.sort(key=lambda item: str(item["name"]))
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "runtime_contract_version": self.runtime_contract_version,
            "components": components,
            "metadata": dict(sorted(self.metadata.items())),
        }

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
        *,
        verify_hash: bool = True,
    ) -> NativeModelManifest:
        try:
            raw_components = payload["components"]
            if not isinstance(raw_components, list):
                raise ModelManifestError("components must be a list")
            components = tuple(
                NativeModelComponent(
                    name=str(item["name"]),
                    version=str(item["version"]),
                    artifact_sha256=str(item["artifact_sha256"]),
                    contract_version=int(item.get("contract_version", 1)),
                )
                for item in raw_components
                if isinstance(item, dict)
            )
            raw_metadata = dict(payload.get("metadata", {}))
            manifest = cls(
                model_id=str(payload["model_id"]),
                model_version=str(payload["model_version"]),
                runtime_contract_version=int(payload["runtime_contract_version"]),
                components=components,
                metadata={str(k): str(v) for k, v in raw_metadata.items()},
                schema=str(payload.get("schema", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelManifestError("malformed native model manifest") from exc
        manifest.validate()
        if verify_hash:
            expected = payload.get("manifest_sha256")
            if not isinstance(expected, str) or expected != manifest.manifest_sha256:
                raise ModelManifestError("native model manifest hash mismatch")
        return manifest

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_hash: bool = True,
    ) -> NativeModelManifest:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelManifestError("unable to read native model manifest") from exc
        if not isinstance(payload, dict):
            raise ModelManifestError("native model manifest root must be an object")
        return cls.from_dict(payload, verify_hash=verify_hash)


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    compatible: bool
    reason: str


def check_runtime_compatibility(
    manifest: NativeModelManifest,
    *,
    runtime_contract_version: int,
    supported_component_contracts: dict[str, int],
) -> RuntimeCompatibility:
    """Fail closed when a learned artifact requires a newer runtime contract."""

    manifest.validate()
    if manifest.runtime_contract_version > runtime_contract_version:
        return RuntimeCompatibility(
            False,
            "model runtime contract is newer than this CINEOS runtime",
        )
    for component in manifest.components:
        supported = supported_component_contracts.get(component.name)
        if supported is None:
            return RuntimeCompatibility(
                False,
                f"unsupported component: {component.name}",
            )
        if component.contract_version > supported:
            reason = (
                f"component {component.name} requires contract "
                f"{component.contract_version}, runtime supports {supported}"
            )
            return RuntimeCompatibility(False, reason)
    return RuntimeCompatibility(True, "compatible")


@dataclass(slots=True)
class NativeModelRegistry:
    """Persistent activation history with compatibility-gated rollback."""

    path: Path
    runtime_contract_version: int
    supported_component_contracts: dict[str, int]

    def _empty_state(self) -> dict[str, object]:
        return {
            "schema": NATIVE_MODEL_REGISTRY_SCHEMA,
            "active_manifest_sha256": None,
            "history": [],
            "manifests": {},
        }

    def _load_state(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty_state()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelManifestError("unable to read native model registry") from exc
        if (
            not isinstance(state, dict)
            or state.get("schema") != NATIVE_MODEL_REGISTRY_SCHEMA
        ):
            raise ModelManifestError("unsupported native model registry schema")
        if not isinstance(state.get("history"), list) or not isinstance(
            state.get("manifests"), dict
        ):
            raise ModelManifestError("malformed native model registry")
        return state

    def _save_state(self, state: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def activate(self, manifest: NativeModelManifest) -> str:
        compatibility = check_runtime_compatibility(
            manifest,
            runtime_contract_version=self.runtime_contract_version,
            supported_component_contracts=self.supported_component_contracts,
        )
        if not compatibility.compatible:
            raise ModelManifestError(
                "refusing incompatible model activation: " + compatibility.reason
            )
        state = self._load_state()
        digest = manifest.manifest_sha256
        manifests = dict(state["manifests"])
        manifests[digest] = manifest.to_dict()
        history = [str(value) for value in state["history"]]
        active = state.get("active_manifest_sha256")
        if active is not None and active != digest:
            history.append(str(active))
        state["manifests"] = manifests
        state["history"] = history
        state["active_manifest_sha256"] = digest
        self._save_state(state)
        return digest

    def active(self) -> NativeModelManifest | None:
        state = self._load_state()
        digest = state.get("active_manifest_sha256")
        if digest is None:
            return None
        payload = dict(state["manifests"]).get(str(digest))
        if not isinstance(payload, dict):
            raise ModelManifestError("active manifest is missing from registry")
        return NativeModelManifest.from_dict(payload)

    def rollback(self) -> NativeModelManifest:
        state = self._load_state()
        history = [str(value) for value in state["history"]]
        if not history:
            raise ModelManifestError(
                "no previous native model is available for rollback"
            )
        target_digest = history.pop()
        payload = dict(state["manifests"]).get(target_digest)
        if not isinstance(payload, dict):
            raise ModelManifestError("rollback manifest is missing from registry")
        target = NativeModelManifest.from_dict(payload)
        compatibility = check_runtime_compatibility(
            target,
            runtime_contract_version=self.runtime_contract_version,
            supported_component_contracts=self.supported_component_contracts,
        )
        if not compatibility.compatible:
            raise ModelManifestError(
                "refusing incompatible rollback: " + compatibility.reason
            )
        state["history"] = history
        state["active_manifest_sha256"] = target_digest
        self._save_state(state)
        return target


def component_contract_map(
    components: Iterable[NativeModelComponent],
) -> dict[str, int]:
    """Build a runtime support map while rejecting duplicate component names."""

    result: dict[str, int] = {}
    for component in components:
        component.validate()
        if component.name in result:
            raise ModelManifestError(f"duplicate component: {component.name}")
        result[component.name] = component.contract_version
    return result
