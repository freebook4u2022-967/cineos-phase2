"""Unit and integration coverage for the CINEOS command line."""

import json
from pathlib import Path

from cineos.cli.main import main
from cineos.compiler import load


def _project(path: Path, *, duration: float = 1.0) -> Path:
    path.write_text(
        json.dumps(
            {
                "title": "CLI film",
                "author": "Test",
                "scenes": [
                    {
                        "scene_id": "scene-1",
                        "title": "Opening",
                        "duration": duration,
                        "shots": [{"shot_id": "shot-1", "duration": 1.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_version_supports_json(capsys) -> None:
    assert main(["--json", "version"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "0.1.0"


def test_usage_errors_support_json(capsys) -> None:
    assert main(["compile", "project.json", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert error["exit_code"] == 2
    assert "--output" in error["error"]


def test_validate_rejects_invalid_project(tmp_path, capsys) -> None:
    project = _project(tmp_path / "project.json", duration=2.0)
    assert main(["validate", str(project)]) == 4
    assert "does not match shot duration" in capsys.readouterr().err


def test_compile_creates_verified_package(tmp_path) -> None:
    project = _project(tmp_path / "project.json")
    destination = tmp_path / "film-package.json"
    assert main(["compile", str(project), "--output", str(destination)]) == 0
    assert load(destination).project_metadata["title"] == "CLI film"


def test_demo_runs_integrated_preview_pipeline(tmp_path) -> None:
    destination = tmp_path / "demo"
    assert main(["demo", "--output-dir", str(destination)]) == 0
    assert (destination / "film-package.json").is_file()
    assert (destination / "renders" / "demo-shot.preview.json").is_file()
    assert (destination / "demo.mp4").read_bytes().startswith(b"CINEOS-PREVIEW-MOVIE\n")


def test_render_and_assemble_are_deterministic(tmp_path) -> None:
    project = _project(tmp_path / "project.json")
    package = tmp_path / "film-package.json"
    render_dir = tmp_path / "renders"
    movie = tmp_path / "movie.mp4"

    assert main(["compile", str(project), "--output", str(package)]) == 0
    assert main(["render", str(package), "--output-dir", str(render_dir)]) == 0
    assert main(["assemble", str(render_dir), "--output", str(movie)]) == 0
    first = movie.read_bytes()

    assert main(["render", str(package), "--output-dir", str(render_dir)]) == 0
    assert main(["assemble", str(render_dir), "--output", str(movie)]) == 0
    assert movie.read_bytes() == first
