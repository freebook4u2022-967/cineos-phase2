import json
from threading import Event, Thread

import pytest

from cineos.runtime import (
    AtlasRuntime,
    EventBus,
    JobState,
    RenderJob,
    RenderQueue,
    RuntimeConfig,
    RuntimeEvent,
    RuntimeState,
    load_config,
)


def test_queue_is_fifo_and_assigns_unique_job_ids() -> None:
    queue = RenderQueue()
    first = RenderJob(lambda context: None)
    second = RenderJob(lambda context: None)
    queue.put(first)
    queue.put(second)

    assert first.id != second.id
    assert queue.get(timeout=0) is first
    queue.task_done()
    assert queue.get(timeout=0) is second
    queue.task_done()


def test_runtime_tracks_progress_result_and_events() -> None:
    observed: list[RuntimeEvent] = []

    def task(context):
        context.report_progress(0.5)
        return "done"

    with AtlasRuntime() as runtime:
        runtime.events.subscribe("*", observed.append)
        job = RenderJob(task, name="infrastructure-only")
        assert runtime.submit(job) == job.id
        assert runtime.run_next(timeout=0).result(timeout=1) == "done"

    assert job.state == JobState.COMPLETED
    assert job.progress == 1.0
    assert job.result == "done"
    assert job.attempts == 1
    assert runtime.state == RuntimeState.CLOSED
    assert [event.name for event in observed] == [
        "job.queued",
        "job.started",
        "job.progress",
        "job.completed",
    ]


def test_failed_job_is_retried_automatically() -> None:
    calls = 0

    def flaky(context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return 42

    with AtlasRuntime() as runtime:
        job = RenderJob(flaky, max_retries=1)
        runtime.submit(job)
        with pytest.raises(RuntimeError, match="temporary"):
            runtime.run_next(timeout=0).result(timeout=1)
        runtime.run_next(timeout=0).result(timeout=1)

    assert job.state == JobState.COMPLETED
    assert job.attempts == 2
    assert job.result == 42


def test_queued_job_can_be_cancelled() -> None:
    with AtlasRuntime() as runtime:
        job = RenderJob(lambda context: pytest.fail("cancelled task ran"))
        runtime.submit(job)
        assert runtime.cancel(job.id)
        assert not runtime.cancel(job.id)

    assert job.state == JobState.CANCELLED


def test_running_job_supports_cooperative_cancellation() -> None:
    started = Event()

    def work(context):
        started.set()
        while True:
            context.raise_if_cancelled()

    with AtlasRuntime() as runtime:
        job = RenderJob(work)
        runtime.submit(job)
        future = runtime.run_next(timeout=0)
        assert started.wait(1)
        assert runtime.cancel(job.id)
        with pytest.raises(Exception, match="cancelled"):
            future.result(timeout=1)

    assert job.state == JobState.CANCELLED


def test_queue_accepts_producers_from_multiple_threads() -> None:
    queue = RenderQueue()
    jobs = [RenderJob(lambda context: None) for _ in range(20)]
    producers = [Thread(target=queue.put, args=(job,)) for job in jobs]
    for producer in producers:
        producer.start()
    for producer in producers:
        producer.join()

    assert queue.pending == 20
    assert {queue.get(timeout=0).id for _ in jobs} == {job.id for job in jobs}


def test_event_callback_errors_are_isolated(caplog) -> None:
    bus = EventBus()

    def broken_callback(event):
        raise RuntimeError("callback problem")

    bus.subscribe("test", broken_callback)
    bus.emit(RuntimeEvent("test"))

    assert "runtime event callback failed" in caplog.text


def test_load_config_combines_json_and_environment(tmp_path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"workers": 2, "log_level": "DEBUG"}))

    config = load_config(path, {"CINEOS_RUNTIME_WORKERS": "3"})

    assert config == RuntimeConfig(workers=3, log_level="DEBUG")


def test_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RuntimeConfig(workers=0)
