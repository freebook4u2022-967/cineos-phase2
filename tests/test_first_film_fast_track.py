from cineos.film.audio import mux_primary_audio
from cineos.film.build import BuildStatus, FilmBuild
from cineos.film.first_film import (
    DirectorCharacter,
    FastTrackAutoDirector,
    FirstFilmRunner,
)


class _Renderer:
    pass


class _FinalReport:
    def __init__(self, decision, directives=()):
        self.decision = decision
        self.directives = tuple(directives)

    def as_dict(self):
        return {"decision": self.decision, "directives": list(self.directives)}


class _FinalEvaluator:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def evaluate(self, movie_path, plan):
        self.calls.append((movie_path, tuple(plan)))
        return self.report


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
    assert build.metadata["first_film"]["critical_path"][-1] == "final_film_qc"
    assert build.metadata["first_film"]["runtime_checkpointing"] is False
    assert build.metadata["first_film"]["final_film_qc_enabled"] is False


def test_first_film_runner_binds_provider_neutral_runtime_hooks():
    runtime_state = {"kind": "test-runtime", "accepted": ["shot-001"]}

    def provider():
        return runtime_state

    def restorer(payload):
        runtime_state.update(payload)

    def started(planned, scene_index, attempt):
        del planned, scene_index, attempt

    def accepted(planned, scene_index, attempt):
        del planned, scene_index, attempt

    def rejected(planned, scene_index, attempt):
        del planned, scene_index, attempt

    runner = FirstFilmRunner(
        _Renderer(),
        orchestrator_kwargs={
            "checkpoint_state_provider": provider,
            "checkpoint_state_restorer": restorer,
            "shot_attempt_start": started,
            "shot_attempt_accepted": accepted,
            "shot_attempt_rejected": rejected,
        },
    )
    assert runner.orchestrator.checkpoint_state_provider is provider
    assert runner.orchestrator.checkpoint_state_restorer is restorer
    assert runner.orchestrator.shot_attempt_start is started
    assert runner.orchestrator.shot_attempt_accepted is accepted
    assert runner.orchestrator.shot_attempt_rejected is rejected


def test_first_film_dry_run_persists_runtime_checkpoint(tmp_path):
    checkpoint = tmp_path / "film-checkpoint.json"
    runner = FirstFilmRunner(
        _Renderer(),
        orchestrator_kwargs={
            "checkpoint_state_provider": lambda: {
                "kind": "test-runtime",
                "accepted": [],
            }
        },
    )
    build = runner.run(
        "A courier reaches the final checkpoint before sunrise.",
        [DirectorCharacter("courier", "Courier")],
        tmp_path / "output",
        dry_run=True,
        checkpoint_path=checkpoint,
    )
    assert checkpoint.exists()
    assert build.metadata["first_film"]["runtime_checkpointing"] is True


def test_first_film_runner_rejects_runtime_hooks_that_override_runner_policy():
    try:
        FirstFilmRunner(
            _Renderer(),
            orchestrator_kwargs={"max_recovery_attempts": 99},
        )
    except ValueError as error:
        assert "cannot override runner policy" in str(error)
    else:
        raise AssertionError("runner policy override should be rejected")


def test_first_film_production_mode_requires_final_film_evaluator():
    try:
        FirstFilmRunner(_Renderer(), require_final_film_evaluation=True)
    except ValueError as error:
        assert "requires a final_film_evaluator" in str(error)
    else:
        raise AssertionError("production mode must fail closed without final-film QC")


def test_final_film_qc_rejects_assembled_movie_and_persists_evidence(tmp_path):
    evaluator = _FinalEvaluator(_FinalReport("reject", ("temporal drift",)))
    runner = FirstFilmRunner(
        _Renderer(),
        final_film_evaluator=evaluator,
        require_final_film_evaluation=True,
    )
    package = FastTrackAutoDirector().direct(
        "A fugitive returns home before dawn.",
        [DirectorCharacter("arif", "Arif")],
    )
    build = FilmBuild("project", package.package_id, "native-test")
    build.transition(BuildStatus.COMPLETED)
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")

    runner._evaluate_final_movie(build, movie, package)

    assert build.status == BuildStatus.FAILED
    assert evaluator.calls
    assert len(evaluator.calls[0][1]) == 3
    assert build.metadata["final_film_qc"]["decision"] == "reject"
    assert build.metadata["final_film_qc"]["evidence"]["directives"] == [
        "temporal drift"
    ]
    assert "temporal drift" in build.failures[-1]


def test_final_film_qc_warning_preserves_deliverable_with_warning(tmp_path):
    evaluator = _FinalEvaluator(_FinalReport("warn", ("minor edit discontinuity",)))
    runner = FirstFilmRunner(_Renderer(), final_film_evaluator=evaluator)
    package = FastTrackAutoDirector().direct(
        "A courier reaches the final checkpoint before sunrise.",
        [DirectorCharacter("courier", "Courier")],
    )
    build = FilmBuild("project", package.package_id, "native-test")
    build.transition(BuildStatus.COMPLETED)
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")

    runner._evaluate_final_movie(build, movie, package)

    assert build.status == BuildStatus.COMPLETED_WITH_WARNINGS
    assert "minor edit discontinuity" in build.warnings[-1]


def test_final_film_qc_invalid_decision_fails_closed(tmp_path):
    evaluator = _FinalEvaluator(_FinalReport("maybe"))
    runner = FirstFilmRunner(_Renderer(), final_film_evaluator=evaluator)
    package = FastTrackAutoDirector().direct(
        "A final signal appears over the city.",
        [DirectorCharacter("lead", "Lead")],
    )
    build = FilmBuild("project", package.package_id, "native-test")
    build.transition(BuildStatus.COMPLETED)
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")

    runner._evaluate_final_movie(build, movie, package)

    assert build.status == BuildStatus.FAILED
    assert "invalid decision" in build.failures[-1]


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
