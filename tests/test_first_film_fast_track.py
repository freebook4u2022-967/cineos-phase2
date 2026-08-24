from pathlib import Path

from cineos.film.audio import mux_primary_audio
from cineos.film.first_film import DirectorCharacter, FastTrackAutoDirector, FirstFilmRunner


class _Renderer:
    pass


def test_fast_track_director_locks_character_ids_across_all_shots():
    director = FastTrackAutoDirector(shot_duration=3.0)
    package = director.direct(
        "A fugitive returns home before dawn.",
        [
            DirectorCharacter("arif", "Arif", ("refs/arif-front.png",)),
            DirectorCharacter("hana", "Hana", ("refs/hana-front.png",)),
        ],
    )
    assert len(package.shot_manifest) == 3
    assert package.timeline_manifest["shot_order"]["scene-001"] == [
        "shot-001",
        "shot-002",
        "shot-003",
    ]
    for shot in package.shot_manifest:
        assert shot["character_ids"] == ("arif", "hana")
        assert shot["duration"] == 3.0
        assert shot["continuity_key"].endswith("arif:hana")


def test_first_film_runner_dry_run_reaches_renderable_plan(tmp_path):
    runner = FirstFilmRunner(_Renderer(), renderer_id="native-test")
    build = runner.run(
        "A survivor discovers a signal in an abandoned city.",
        [DirectorCharacter("lead", "Lead")],
        tmp_path,
        dry_run=True,
    )
    assert build.renderer_id == "native-test"
    assert build.metadata["dry_run"]["shot_count"] == 3
    assert build.metadata["dry_run"]["renderer_compatible"] is True
    assert build.metadata["first_film"]["character_ids"] == ["lead"]
    assert build.metadata["first_film"]["critical_path"][-1] == "audio_mux"


def test_director_rejects_duplicate_character_ids():
    director = FastTrackAutoDirector()
    try:
        director.direct(
            "Premise",
            [DirectorCharacter("hero", "A"), DirectorCharacter("hero", "B")],
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate IDs should be rejected")


def test_audio_mux_has_nonblocking_silent_fallback(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"placeholder-film")
    destination = tmp_path / "first-film.mp4"
    result = mux_primary_audio(source, [], destination)
    assert result == destination
    assert destination.read_bytes() == source.read_bytes()
