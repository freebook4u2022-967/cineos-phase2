import json

from cineos.assets.storage import load as load_asset_registry
from cineos.cinedna import CineDNARegistry
from cineos.short_drama import DramaBrief, ShortDramaOrchestrator
from cineos.short_drama.character_approval import approve_character_files
from cineos.short_drama.integration import write_production_artifacts
from cineos.short_drama.entrypoint import main


def _project(tmp_path):
    plan = ShortDramaOrchestrator().plan(
        DramaBrief(
            premise="A man receives a message from his wife who died three years ago.",
            duration_seconds=180,
            genre="mystery",
        )
    )
    return write_production_artifacts(plan, tmp_path)


def _identity(path):
    path.write_text(
        json.dumps(
            {
                "display_name": "Protagonist",
                "face": {
                    "age_range": "adult",
                    "facial_feature_descriptors": {"identity": "locked"},
                    "invariants": ["preserve approved facial identity"],
                },
                "body": {
                    "height": "project-defined",
                    "build": "project-defined",
                    "silhouette_constraints": ["preserve approved body silhouette"],
                },
                "constraints": {
                    "identity_lock": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_approval_promotes_pending_character_to_valid_cinedna(tmp_path):
    artifacts = _project(tmp_path)
    identity = _identity(tmp_path / "identity.json")
    profiles = tmp_path / "cinedna.json"

    result = approve_character_files(
        artifacts["asset_registry"],
        "Protagonist",
        "references/protagonist-front.png",
        identity,
        profiles_path=profiles,
    )

    assets = load_asset_registry(artifacts["asset_registry"])
    character = assets.retrieve(result["character_id"])
    assert character.metadata["cinedna"]["status"] == "approved"
    assert character.references[0].approval_status == "approved"

    registry = CineDNARegistry.load(profiles)
    profile = registry.retrieve(result["character_id"])
    assert profile.display_name == "Protagonist"
    assert profile.approved_reference_ids
    assert profile.face_profile.reference_asset_ids == profile.approved_reference_ids


def test_character_approval_requires_explicit_face_and_body(tmp_path):
    artifacts = _project(tmp_path)
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"face": {}}), encoding="utf-8")

    try:
        approve_character_files(
            artifacts["asset_registry"],
            "Protagonist",
            "references/protagonist-front.png",
            identity,
            profiles_path=tmp_path / "cinedna.json",
        )
    except ValueError as error:
        assert "body" in str(error)
    else:
        raise AssertionError("approval accepted identity without explicit body data")


def test_character_approval_is_available_from_cineos_drama_cli(tmp_path, capsys):
    artifacts = _project(tmp_path)
    identity = _identity(tmp_path / "identity.json")
    profiles = tmp_path / "cinedna.json"

    assert (
        main(
            [
                "--json",
                "drama",
                "character",
                "approve",
                "Protagonist",
                "--assets",
                str(artifacts["asset_registry"]),
                "--reference",
                "references/protagonist-front.png",
                "--identity",
                str(identity),
                "--profiles",
                str(profiles),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["character_name"] == "Protagonist"
    assert profiles.is_file()
