from cineos.short_drama.first_film import ShortDramaFirstFilmRunner


class _Renderer:
    pass


def test_full_short_drama_director_reaches_film_render_plan(tmp_path):
    runner = ShortDramaFirstFilmRunner(_Renderer(), renderer_id="native-test")
    build = runner.run(
        "A courier must deliver a mysterious package before sunrise.",
        tmp_path,
        duration_seconds=30,
        dry_run=True,
    )
    assert build.renderer_id == "native-test"
    assert build.metadata["auto_director"]["engine"] == "ShortDramaOrchestrator"
    assert build.metadata["auto_director"]["shot_count"] > 0
    assert build.metadata["dry_run"]["shot_count"] > 0
    assert build.metadata["dry_run"]["renderer_compatible"] is True
