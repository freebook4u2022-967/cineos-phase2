from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

from .package import ColabRenderPackage
from .serializer import dump_json


def export_package(
    package: ColabRenderPackage, output: Path, source_root: Path | None = None
) -> Path:
    def build(root: Path):
        (root / "references").mkdir(parents=True)
        (root / "audio").mkdir()
        for manifest, folder in (
            (package.approved_reference_manifest, "references"),
            (package.dialogue_audio_manifest, "audio"),
        ):
            for item in manifest:
                src = Path(item.get("path", ""))
                if src.is_file():
                    shutil.copy2(src, root / folder / src.name)
        data = package.to_dict()
        data["checksums"] = {}
        dump_json(data, root / "package.json")
        data["checksums"]["package.json"] = hashlib.sha256(
            (root / "package.json").read_bytes()
        ).hexdigest()
        dump_json(data, root / "package.json")

    if output.suffix.lower() == ".zip":
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "mission-one"
            root.mkdir()
            build(root)
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(root))
    else:
        output.mkdir(parents=True, exist_ok=True)
        build(output)
    return output
