"""Canonical JSON encoding and SHA-256 content hashing."""

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Encode JSON with stable ordering and no insignificant whitespace."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any) -> str:
    """Return the SHA-256 digest of a value's canonical JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_hashes(payload: dict[str, Any]) -> dict[str, str]:
    """Hash each package section and the complete unhashed package payload."""

    section_names = (
        "project_metadata",
        "scene_manifest",
        "shot_manifest",
        "character_manifest",
        "location_manifest",
        "asset_manifest",
        "timeline_manifest",
    )
    hashes = {name: content_hash(payload[name]) for name in section_names}
    hashes["package"] = content_hash(payload)
    return hashes
