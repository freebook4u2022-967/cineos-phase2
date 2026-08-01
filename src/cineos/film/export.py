"""Export all auditable build artifacts and checksums."""

import json
import shutil
from pathlib import Path

from .build import FilmBuild
from .report import build_report
from .serializer import build_to_dict
from .validator import file_hash


def export_artifacts(
    build: FilmBuild, output_dir: str | Path, final_mp4: str | Path | None = None
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "build": build_to_dict(build),
        "shot_manifest": [shot.shot_id for shot in build.shot_states],
        "validation_reports": build.validation_states,
        "recovery_history": build.recovery_states,
        "renderer_metadata": {
            "renderer_id": build.renderer_id,
            **build.metadata.get("renderer", {}),
        },
        "final_timeline": [shot.selected_output for shot in build.shot_states],
        "build_summary": build_report(build),
    }
    outputs: dict[str, str] = {}
    for name, value in artifacts.items():
        path = root / f"{name}.json"
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        outputs[name] = str(path)
    if final_mp4:
        target = root / "final.mp4"
        if Path(final_mp4).resolve() != target.resolve():
            shutil.copy2(final_mp4, target)
        outputs["final_mp4"] = str(target)
    checksums = {name: file_hash(path) for name, path in sorted(outputs.items())}
    checksum_path = root / "checksum_manifest.json"
    checksum_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["checksum_manifest"] = str(checksum_path)
    build.output_files.update(outputs)
    return outputs
