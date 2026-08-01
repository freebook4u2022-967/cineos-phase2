"""Studio controller, state, model, worker, and offscreen smoke tests."""

from pathlib import Path

import pytest

from cineos.core import Scene, Shot
from cineos.studio import StudioController, StudioState
from cineos.studio.models import QueueState, RenderQueueItem, ReviewResult


def test_project_open_save_edit_and_compile(tmp_path: Path) -> None:
    controller = StudioController()
    controller.new_project("Short", "Director")
    controller.update_metadata(language="en", duration_target=2.0, fps=25.0)
    controller.add_scene(Scene("s1", "Opening"))
    controller.add_shot("s1", Shot("sh1", action="Fade in", duration=2.0))
    controller.move_shot("s1", 0, 0)

    destination = controller.save(tmp_path / "short.cineos.json")
    assert not controller.state.dirty
    assert controller.validate() == []
    assert controller.compile().project_metadata["fps"] == 25.0

    reopened = StudioController()
    project = reopened.open(destination)
    assert project.scenes[0].shots[0].action == "Fade in"
    assert reopened.state.language == "en"


def test_state_selection_queue_and_review_models() -> None:
    state = StudioState(selected_asset_id="character-1", selected_shot_id="shot-1")
    item = RenderQueueItem("job-1", "shot-1", "mock")
    item.start()
    state.queue.append(item)
    state.reviews["shot-1"] = ReviewResult("shot-1", approved=True)
    assert item.state is QueueState.RUNNING
    assert item.attempts == 1
    assert state.reviews["shot-1"].approved


def test_queue_rejects_invalid_progress() -> None:
    with pytest.raises(ValueError, match="progress"):
        RenderQueueItem("job", "shot", "mock", progress=2)


def test_worker_and_offscreen_application(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from cineos.studio.app import create_application
    from cineos.studio.main_window import MainWindow
    from cineos.studio.workers import BackgroundWorker

    app = create_application([])
    worker = BackgroundWorker(
        lambda *, cancel_event, progress: progress(1, "done") or "ok"
    )
    results: list[object] = []
    worker.signals.result.connect(results.append)
    worker.run()
    window = MainWindow(StudioController())
    assert window.windowTitle() == "CINEOS Studio Alpha"
    assert results == ["ok"]
    window.close()
    app.processEvents()
