"""Version-aware registry for persistent CineDNA profiles."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from .exceptions import DuplicateProfileError, ProfileNotFoundError
from .profile import CharacterDNA
from .serializer import profile_from_dict, profile_to_dict
from .validator import CineDNAValidator


class CineDNARegistry:
    def __init__(self) -> None:
        self._profiles: dict[UUID, list[CharacterDNA]] = {}

    def register(self, profile: CharacterDNA) -> CharacterDNA:
        CineDNAValidator().raise_for_errors(profile)
        versions = self._profiles.setdefault(profile.character_uuid, [])
        if any(item.profile_version == profile.profile_version for item in versions):
            raise DuplicateProfileError(
                f"profile {profile.character_uuid} version "
                f"{profile.profile_version} exists"
            )
        versions.append(profile)
        versions.sort(key=lambda item: _version_key(item.profile_version))
        return profile

    def retrieve(
        self, character_uuid: UUID | str, version: str | None = None
    ) -> CharacterDNA:
        try:
            profiles = self._profiles[UUID(str(character_uuid))]
        except (KeyError, ValueError) as error:
            raise ProfileNotFoundError(str(character_uuid)) from error
        if version is None:
            return profiles[-1]
        for profile in profiles:
            if profile.profile_version == str(version):
                return profile
        raise ProfileNotFoundError(f"{character_uuid} version {version}")

    get = retrieve
    resolve = retrieve
    resolve_by_character_uuid = retrieve

    def update(self, profile: CharacterDNA) -> CharacterDNA:
        """Register a new version, retaining prior immutable revisions."""

        if profile.character_uuid not in self._profiles:
            raise ProfileNotFoundError(str(profile.character_uuid))
        return self.register(profile)

    def version(self, character_uuid: UUID | str) -> tuple[CharacterDNA, ...]:
        self.retrieve(character_uuid)
        return tuple(self._profiles[UUID(str(character_uuid))])

    versions = version

    def list(self) -> list[CharacterDNA]:
        return [self._profiles[key][-1] for key in sorted(self._profiles, key=str)]

    def validate(self, character_uuid: UUID | str | None = None) -> list[str]:
        profiles = (
            [self.retrieve(character_uuid)]
            if character_uuid is not None
            else self.list()
        )
        validator = CineDNAValidator()
        errors: list[str] = []
        for profile in profiles:
            errors.extend(
                f"profile {profile.character_uuid}: {error}"
                for error in validator.validate(profile)
            )
        return errors

    def to_dict(self) -> dict[str, object]:
        profiles = [
            profile
            for key in sorted(self._profiles, key=str)
            for profile in self._profiles[key]
        ]
        return {
            "format": "cineos-cinedna-registry-v1",
            "profiles": [profile_to_dict(item) for item in profiles],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> CineDNARegistry:
        if (
            value.get("format", "cineos-cinedna-registry-v1")
            != "cineos-cinedna-registry-v1"
        ):
            raise ValueError("unsupported CineDNA registry format")
        registry = cls()
        for item in value.get("profiles", []):  # type: ignore[union-attr]
            registry.register(profile_from_dict(item))  # type: ignore[arg-type]
        return registry

    @classmethod
    def load(cls, path: str | Path) -> CineDNARegistry:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("CineDNA registry JSON must contain an object")
        return cls.from_dict(value)

    def __len__(self) -> int:
        return len(self._profiles)


def _version_key(version: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in version.split(".")
    )
