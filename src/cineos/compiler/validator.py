"""Structural and content-integrity validation for Film Packages."""

from typing import Any

from .hashing import build_hashes
from .manifest import FILM_PACKAGE_VERSION, FilmPackage
from .serializer import package_to_dict


class PackageValidationError(ValueError):
    """Raised when a Film Package is malformed or has been modified."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def validation_errors(package: FilmPackage) -> list[str]:
    """Return all detectable package structure and integrity errors."""

    if not isinstance(package, FilmPackage):
        return ["package must be a FilmPackage"]
    errors: list[str] = []
    if package.version != FILM_PACKAGE_VERSION:
        errors.append(f"unsupported Film Package version: {package.version!r}")

    scene_ids = _ids(package.scene_manifest, "scene_id", "scene", errors)
    shot_ids = _ids(package.shot_manifest, "shot_id", "shot", errors)
    asset_ids = _ids(package.asset_manifest, "asset_id", "asset", errors)
    _check_duplicates(scene_ids, "scene", errors)
    _check_duplicates(shot_ids, "shot", errors)
    _check_duplicates(asset_ids, "asset", errors)

    timeline_scenes = package.timeline_manifest.get("scene_order")
    if timeline_scenes != scene_ids:
        errors.append("timeline scene order does not match scene manifest")
    shot_order = package.timeline_manifest.get("shot_order")
    if not isinstance(shot_order, dict):
        errors.append("timeline shot_order must be an object")
    else:
        expected = {
            scene_id: [
                entry.get("shot_id")
                for entry in package.shot_manifest
                if isinstance(entry, dict) and entry.get("scene_id") == scene_id
            ]
            for scene_id in scene_ids
        }
        if shot_order != expected:
            errors.append("timeline shot order does not match shot manifest")

    payload = package_to_dict(package)
    payload.pop("content_hashes")
    try:
        expected_hashes = build_hashes(payload)
    except (TypeError, ValueError):
        errors.append("package contains values that cannot be serialized as JSON")
    else:
        if package.content_hashes != expected_hashes:
            errors.append("content hashes do not match package contents")
    return errors


def _ids(entries: Any, key: str, kind: str, errors: list[str]) -> list[str]:
    if not isinstance(entries, list):
        errors.append(f"{kind} manifest must be a list")
        return []
    identifiers: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get(key), str):
            errors.append(f"{kind} manifest entry {index} has an invalid {key}")
        else:
            identifiers.append(entry[key])
    return identifiers


def _check_duplicates(values: list[str], kind: str, errors: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if not value:
            errors.append(f"{kind} ID cannot be empty")
        elif value in seen:
            errors.append(f"duplicate {kind} ID: {value}")
        seen.add(value)


def verify(package: FilmPackage) -> bool:
    """Raise on invalid content and return ``True`` for a valid package."""

    errors = validation_errors(package)
    if errors:
        raise PackageValidationError(errors)
    return True


class PackageValidator:
    """Object-oriented package validation façade."""

    validate = staticmethod(validation_errors)
    verify = staticmethod(verify)
    is_valid = staticmethod(lambda package: not validation_errors(package))
