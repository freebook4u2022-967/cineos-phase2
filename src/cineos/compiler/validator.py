"""Structural and integrity validation for Film Packages."""

from __future__ import annotations

import re

from .hashing import CanonicalizationError, content_hash
from .manifest import FILM_PACKAGE_VERSION, FilmPackage

_SECTIONS = (
    "project_metadata",
    "scene_manifest",
    "shot_manifest",
    "character_manifest",
    "location_manifest",
    "asset_manifest",
    "timeline_manifest",
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class PackageValidationError(ValueError):
    """Raised when a Film Package is malformed or has failed integrity checks."""


def package_hash(package: FilmPackage) -> str:
    """Hash package content and version, excluding the hashes themselves."""
    data = {"version": package.version}
    data.update({name: getattr(package, name) for name in _SECTIONS})
    return content_hash(data)


def validate(package: FilmPackage) -> None:
    """Raise :class:`PackageValidationError` if *package* is not valid."""
    if not isinstance(package, FilmPackage):
        raise PackageValidationError("expected a FilmPackage")
    if package.version != FILM_PACKAGE_VERSION:
        raise PackageValidationError(
            f"unsupported Film Package version: {package.version!r}"
        )
    if not isinstance(package.project_metadata, dict):
        raise PackageValidationError("project_metadata must be an object")
    for name in _SECTIONS[1:-1]:
        if not isinstance(getattr(package, name), list):
            raise PackageValidationError(f"{name} must be an array")
    if not isinstance(package.content_hashes, dict):
        raise PackageValidationError("content_hashes must be an object")
    expected_names = {*_SECTIONS, "package"}
    if set(package.content_hashes) != expected_names:
        raise PackageValidationError("content_hashes has missing or unknown sections")
    try:
        for name in _SECTIONS:
            digest = package.content_hashes[name]
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise PackageValidationError(f"invalid hash for {name}")
            if digest != content_hash(getattr(package, name)):
                raise PackageValidationError(f"content hash mismatch for {name}")
        if package.content_hashes["package"] != package_hash(package):
            raise PackageValidationError("content hash mismatch for package")
    except CanonicalizationError as error:
        raise PackageValidationError(str(error)) from error


def verify(package: FilmPackage) -> bool:
    """Return whether *package* has valid structure, version, and hashes."""
    try:
        validate(package)
    except (PackageValidationError, TypeError, KeyError):
        return False
    return True
