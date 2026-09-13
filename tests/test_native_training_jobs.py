from cineos.native_image.training_jobs import (
    RetryPolicy,
    TrainingJobOrchestrator,
    TrainingJobStore,
)


def test_training_job_completes_and_records_checkpoint(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    orchestrator = TrainingJobOrchestrator(store)
    job = orchestrator.submit("job-1")
    completed = orchestrator.run(job, lambda _: "/checkpoints/best.pt")
    assert completed.state == "completed"
    assert completed.checkpoint_path == "/checkpoints/best.pt"
    assert store.load().state == "completed"


def test_training_job_retries_failed_worker(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    orchestrator = TrainingJobOrchestrator(store, RetryPolicy(max_attempts=2))
    calls = []

    def worker(job):
        calls.append(job.attempts)
        if len(calls) == 1:
            raise RuntimeError("temporary GPU failure")
        return "/checkpoints/recovered.pt"

    completed = orchestrator.run(orchestrator.submit("job-2"), worker)
    assert completed.state == "completed"
    assert completed.attempts == 2
    assert calls == [1, 2]


def test_training_job_stays_failed_after_retry_budget(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    orchestrator = TrainingJobOrchestrator(store, RetryPolicy(max_attempts=2))

    def worker(_):
        raise RuntimeError("GPU unavailable")

    failed = orchestrator.run(orchestrator.submit("job-3"), worker)
    assert failed.state == "failed"
    assert failed.attempts == 2
    assert "GPU unavailable" in failed.error


def test_training_job_can_be_cancelled_before_worker_runs(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    orchestrator = TrainingJobOrchestrator(store)
    cancelled = orchestrator.cancel(orchestrator.submit("job-4"))
    result = orchestrator.run(cancelled, lambda _: "/should-not-run.pt")
    assert result.state == "cancelled"
    assert result.attempts == 0
