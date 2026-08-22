import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cineos.mission_one.validator import assembly_ready
from cineos.renderers.colab.exceptions import AssemblyBlockedError
from cineos.renderers.colab.result import RenderContentStatus, ShotRenderResult
from cineos.renderers.colab.serializer import dump_json
from cineos.renderers.colab.verifier import (
    assemble,
    validate_rendered_shot,
    verify_results,
)

FFMPEG = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg unavailable")


def make_video(path: Path, source: str, duration: float = 1.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-t",
            str(duration),
            "-r",
            "8",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@FFMPEG
def test_black_mp4_detection(tmp_path):
    path = tmp_path / "black.mp4"
    make_video(path, "color=black:s=64x64")
    assert (
        validate_rendered_shot(path).content_status
        == RenderContentStatus.BLACK_FRAME_FAILURE
    )


@FFMPEG
def test_valid_mp4_detection(tmp_path):
    path = tmp_path / "valid.mp4"
    make_video(path, "testsrc2=s=64x64:r=8")
    result = validate_rendered_shot(path)
    assert result.success and result.content_status == RenderContentStatus.VALID


def test_tiny_file_rejected(tmp_path):
    path = tmp_path / "tiny.mp4"
    path.write_bytes(b"not video")
    assert (
        validate_rendered_shot(path).content_status == RenderContentStatus.EMPTY_OUTPUT
    )


@FFMPEG
def test_frozen_frame_detection(tmp_path):
    path = tmp_path / "still.mp4"
    make_video(path, "color=white:s=64x64")
    assert (
        validate_rendered_shot(path).content_status
        == RenderContentStatus.FROZEN_FRAME_FAILURE
    )


def test_assembly_gate_and_valid_three_shots(tmp_path):
    shots = [
        {"shot_id": f"shot-{n}", "success": True, "content_status": "valid"}
        for n in range(1, 4)
    ]
    assert assembly_ready(shots, [f"shot-{n}" for n in range(1, 4)])
    shots[1].update(success=False, content_status="black_frame_failure")
    dump_json(
        {"expected_shots": [f"shot-{n}" for n in range(1, 4)], "shots": shots},
        tmp_path / "render-results.json",
    )
    with pytest.raises(AssemblyBlockedError):
        assemble(tmp_path, tmp_path / "final-film.mp4")


def test_retry_metadata_and_deterministic_serialization(tmp_path):
    result = ShotRenderResult(
        "shot-1",
        "shot-1.mp4",
        False,
        "black_frame_failure",
        retry_attempted=True,
        retry_reason="black_frame_failure",
        original_settings={"frames": 49},
        retry_settings={"frames": 33},
    )
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    dump_json(result.to_dict(), first)
    dump_json(result.to_dict(), second)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["retry_reason"] == "black_frame_failure"


def test_verification_report_counts_failures(tmp_path):
    report = tmp_path / "render-results.json"
    dump_json(
        {
            "expected_shots": ["shot-1"],
            "shots": [
                {
                    "shot_id": "shot-1",
                    "success": False,
                    "content_status": "black_frame_failure",
                }
            ],
        },
        report,
    )
    verified = verify_results(report)
    assert not verified["valid"] and verified["black_frame_failures"] == 1


def test_notebook_uses_t4_safe_path_and_smoke_mode():
    text = Path("colab/CINEOS-Mission-One.ipynb").read_text()
    assert '.to(\\"cuda\\")' not in text
    assert "enable_sequential_cpu_offload" in text
    assert 'Generator(device=\\"cpu\\")' in text
    assert "MISSION_ONE_SMOKE_MODE = False" in text
    assert "[:1] if MISSION_ONE_SMOKE_MODE" in text
