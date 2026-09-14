from __future__ import annotations

import pytest

from cineos.film.first_film import DirectorCharacter, FastTrackAutoDirector


@pytest.fixture
def cast() -> tuple[DirectorCharacter, ...]:
    return (
        DirectorCharacter(character_id="hero", name="Hero"),
        DirectorCharacter(character_id="partner", name="Partner"),
    )


def test_default_director_preserves_three_shot_backwards_compatibility(
    cast: tuple[DirectorCharacter, ...],
) -> None:
    package = FastTrackAutoDirector().direct("A difficult rescue.", cast)

    assert [shot["shot_id"] for shot in package.shot_manifest] == [
        "shot-001",
        "shot-002",
        "shot-003",
    ]
    assert [shot["beat"] for shot in package.shot_manifest] == [
        "setup",
        "escalation",
        "payoff",
    ]


def test_director_can_generate_seedance_style_five_connected_shots(
    cast: tuple[DirectorCharacter, ...],
) -> None:
    package = FastTrackAutoDirector(shot_count=5).direct(
        "A difficult rescue.",
        cast,
    )

    assert len(package.shot_manifest) == 5
    assert package.timeline_manifest["shot_order"]["scene-001"] == [
        "shot-001",
        "shot-002",
        "shot-003",
        "shot-004",
        "shot-005",
    ]
    assert len({shot["beat"] for shot in package.shot_manifest}) == 5
    assert {shot["continuity_key"] for shot in package.shot_manifest} == {
        "scene-001:hero:partner"
    }
    assert all(
        shot["character_ids"] == ("hero", "partner") for shot in package.shot_manifest
    )


def test_director_can_generate_ten_connected_shots_without_duplicate_beats(
    cast: tuple[DirectorCharacter, ...],
) -> None:
    package = FastTrackAutoDirector(shot_count=10).direct(
        "A difficult rescue.",
        cast,
    )

    assert len(package.shot_manifest) == 10
    assert len({shot["beat"] for shot in package.shot_manifest}) == 10
    assert package.shot_manifest[0]["beat"] == "setup"
    assert package.shot_manifest[-1]["beat"] == "payoff"


@pytest.mark.parametrize("shot_count", [2, 11, 3.5, True])
def test_director_rejects_invalid_connected_shot_count(shot_count: object) -> None:
    with pytest.raises(ValueError, match="shot_count"):
        FastTrackAutoDirector(shot_count=shot_count)  # type: ignore[arg-type]
