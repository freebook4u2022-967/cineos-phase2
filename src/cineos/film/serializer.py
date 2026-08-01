"""Canonical persistence for resumable builds."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .build import BuildStatus, FilmBuild
from .shot_state import ShotState


def build_to_dict(build: FilmBuild) -> dict[str, Any]:
    value = asdict(build)
    value["status"] = str(build.status)
    value["content_hash"] = build.content_hash
    return value


def dumps(build: FilmBuild) -> str:
    return json.dumps(build_to_dict(build), sort_keys=True, separators=(",", ":"))


def save(build: FilmBuild, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(build) + "\n", encoding="utf-8")
    return path


def load(source: str | Path | Mapping[str, Any]) -> FilmBuild:
    if isinstance(source, Mapping):
        value = dict(source)
    else:
        raw = (
            Path(source).read_text(encoding="utf-8")
            if Path(source).is_file()
            else str(source)
        )
        value = json.loads(raw)
    expected = value.pop("content_hash", None)
    value["status"] = BuildStatus(value.get("status", "created"))
    value["shot_states"] = [ShotState(**item) for item in value.get("shot_states", [])]
    build = FilmBuild(**value)
    if expected and expected != build.content_hash:
        raise ValueError("FilmBuild content hash mismatch")
    return build
