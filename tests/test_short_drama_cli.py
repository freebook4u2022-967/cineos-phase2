import json

from cineos.short_drama.entrypoint import main


def test_drama_create_writes_full_production_bundle(tmp_path):
    premise = "A man receives a message from his wife who died three years ago."
    assert (
        main(
            [
                "drama",
                "create",
                premise,
                "--duration",
                "180",
                "--genre",
                "mystery",
                "--tone",
                "tense and intimate",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert (tmp_path / "drama-package.json").is_file()
    assert (tmp_path / "assets.json").is_file()
    assert (tmp_path / "film-package.json").is_file()


def test_drama_create_supports_json_output(tmp_path, capsys):
    premise = "A woman finds tomorrow's newspaper on her doorstep."
    assert (
        main(
            [
                "--json",
                "drama",
                "create",
                premise,
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["continuity_status"] == "pass"
    assert payload["film_package"].endswith("film-package.json")


def test_existing_cli_commands_are_preserved(capsys):
    assert main(["--json", "version"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "0.1.0-alpha.1"
