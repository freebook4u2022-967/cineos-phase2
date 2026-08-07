from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceManifestEntry:
    asset_id: str
    path: str
    sha256: str
    consumed_by_backend: bool = False
