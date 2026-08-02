"""Complete audio deliverable export with checksums."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .project import AudioProject
from .report import production_report
from .serializer import project_to_dict


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


class AudioExporter:
    def export(
        self,
        project: AudioProject,
        output_dir: str | Path,
        *,
        mixed_audio: str | Path | None = None,
        stems: dict[str, str | Path] | None = None,
    ) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        if mixed_audio and Path(mixed_audio).is_file():
            target = root / f"mixed{Path(mixed_audio).suffix}"
            shutil.copyfile(mixed_audio, target)
            outputs["mixed_audio"] = target
        stem_root = root / "stems"
        for name, source in sorted((stems or {}).items()):
            if Path(source).is_file():
                stem_root.mkdir(exist_ok=True)
                target = stem_root / f"{name}{Path(source).suffix}"
                shutil.copyfile(source, target)
                outputs[f"stem_{name}"] = target
        documents = {
            "audio_project": ("audio-project.json", project_to_dict(project)),
            "cue_sheet": (
                "cue-sheet.json",
                {
                    "dialogue": [asdict(item) for item in project.dialogue_tracks],
                    "ambience": [asdict(item) for item in project.ambience_tracks],
                    "effects": [asdict(item) for item in project.effects_tracks],
                    "music": [asdict(item) for item in project.music_tracks],
                },
            ),
            "subtitles": ("subtitle-timing.json", project.subtitle_metadata),
            "lip_sync": (
                "lip-sync.json",
                [
                    {**asdict(item), "content_hash": item.content_hash}
                    for item in project.lip_sync_metadata
                ],
            ),
            "report": ("audio-production-report.json", production_report(project)),
        }
        for key, (name, value) in documents.items():
            target = root / name
            _write_json(target, value)
            outputs[key] = target
        checksums = {
            path.relative_to(root)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in outputs.values()
        }
        checksum_path = root / "checksums.json"
        _write_json(checksum_path, checksums)
        outputs["checksums"] = checksum_path
        return outputs


export = AudioExporter().export
