from __future__ import annotations

import json
import subprocess
from pathlib import Path


def verify_results(source: Path) -> dict:
    data = json.loads(source.read_text())
    expected = data.get("expected_shots") or [
        x.get("shot_id") for x in data.get("shots", [])
    ]
    rendered = {
        x.get("shot_id") for x in data.get("shots", []) if x.get("success", True)
    }
    missing = [x for x in expected if x not in rendered]
    return {
        "valid": not missing and bool(data.get("shots")),
        "missing_shots": missing,
        "reference_conditioning": "unsupported",
        "warnings": [
            "CogVideoX-2B text-to-video does not consume packaged reference images",
            "Lip-sync is approximate unless measured",
        ],
    }


def assemble(render_dir: Path, output: Path, fps: int | None = None) -> Path:
    shots = sorted(render_dir.glob("shot-*.mp4")) or sorted(render_dir.glob("*.mp4"))
    if len(shots) != 3:
        raise ValueError(f"expected three rendered shots, found {len(shots)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    listing = render_dir / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in shots))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    if not output.exists() or output.stat().st_size == 0:
        raise ValueError("FFmpeg produced an empty film")
    return output
