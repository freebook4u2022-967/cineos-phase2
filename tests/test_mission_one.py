import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from cineos.mission_one.brief import DirectedSceneBrief
from cineos.mission_one.compiler import compile_scene
from cineos.mission_one.performance import compile_performance
from cineos.renderers.colab.exporter import export_package
from cineos.renderers.colab.notebook import load_package
from cineos.renderers.colab.verifier import assemble, verify_results
from cineos.studio.controller import StudioController

EXAMPLE = Path("examples/mission_one/creative-brief.json")


def brief():
    return DirectedSceneBrief.from_dict(json.loads(EXAMPLE.read_text()))


def test_deterministic_serialization_compilation_and_chronology():
    first = brief()
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        brief().to_dict(), sort_keys=True
    )
    one = compile_scene(first).to_dict()
    two = compile_scene(brief()).to_dict()
    assert one == two
    assert one["prompts"][0].index("First,") < one["prompts"][0].index("Camera:")


def test_dialogue_metadata_and_continuity_propagation():
    scene = brief()
    package = compile_scene(scene)
    dialogue = package.shots[1].dialogue
    assert dialogue["text"] == "That signal knows my name."
    assert dialogue["audio_strategy"] == "separate_audio"
    performance = compile_performance(scene, scene.shots[1]).to_dict()
    assert performance["CONTINUITY"]["previous_shot"]["previous_shot_id"] == "shot-1"


def test_export_and_notebook_loading(tmp_path):
    package = compile_scene(brief())
    archive = export_package(package, tmp_path / "package.zip")
    with zipfile.ZipFile(archive) as value:
        value.extractall(tmp_path / "loaded")
    loaded = load_package(tmp_path / "loaded/package.json")
    assert [shot.shot_id for shot in loaded.shots] == ["shot-1", "shot-2", "shot-3"]
    notebook = json.loads(Path("colab/CINEOS-Mission-One.ipynb").read_text())
    assert "package.json" in json.dumps(notebook)


def test_output_verification_detects_missing_shot(tmp_path):
    path = tmp_path / "render-results.json"
    path.write_text(
        json.dumps(
            {
                "expected_shots": ["shot-1", "shot-2", "shot-3"],
                "shots": [{"shot_id": "shot-1"}],
            }
        )
    )
    result = verify_results(path)
    assert not result["valid"]
    assert result["missing_shots"] == ["shot-2", "shot-3"]


def test_ffmpeg_assembly_with_fake_rendered_mp4s(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed in this test environment")
    for number in range(1, 4):
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=.1",
                "-pix_fmt",
                "yuv420p",
                str(tmp_path / f"shot-{number}.mp4"),
            ],
            check=True,
        )
    output = assemble(tmp_path, tmp_path / "final-film.mp4")
    assert output.stat().st_size > 0


def test_studio_controller_flow(tmp_path):
    controller = StudioController()
    controller.load_mission_one_brief(EXAMPLE)
    controller.update_mission_one_shot("shot-2", action="presses the glowing control")
    package = controller.compile_mission_one(tmp_path / "package.json")
    assert len(package.shots) == 3
