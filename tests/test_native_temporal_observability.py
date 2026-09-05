from __future__ import annotations

import json

from cineos.native_image.tensor_model import Tensor
from cineos.native_video.observability import (
    TEMPORAL_EVENT_SCHEMA,
    InMemoryTemporalObserver,
    JsonlTemporalObserver,
    TemporalRuntimeEvent,
)
from cineos.native_video.runtime import NativeTemporalRuntime
from cineos.native_video.temporal_model import NativeTemporalModel, TemporalFrameInput


def _frame(index: int, motion: float = 0.05) -> TemporalFrameInput:
    return TemporalFrameInput(
        shot_id="shot-observe",
        frame_index=index,
        identity=Tensor((0.1,) * 8, (8,)),
        scene=Tensor((0.2,) * 8, (8,)),
        motion=Tensor((motion,) * 4, (4,)),
    )


def test_runtime_emits_versioned_event_for_accepted_candidate() -> None:
    model = NativeTemporalModel.initialized()
    observer = InMemoryTemporalObserver()
    runtime = NativeTemporalRuntime.default(model=model, observer=observer)
    state = model.initial_state("shot-observe")

    result = runtime.generate_frame(_frame(0), state)

    events = observer.snapshot()
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "candidate_accepted"
    assert event.schema_version == TEMPORAL_EVENT_SCHEMA
    assert event.shot_id == result.report.shot_id
    assert event.frame_index == 0
    assert event.attempt == 1
    assert event.decision == result.report.decision
    assert event.occurred_at


def test_jsonl_observer_writes_machine_readable_audit_record(tmp_path) -> None:
    path = tmp_path / "temporal-events.jsonl"
    observer = JsonlTemporalObserver(path)
    event = TemporalRuntimeEvent(
        event_type="candidate_rejected",
        shot_id="shot-002",
        frame_index=3,
        attempt=2,
        decision="retry",
        continuity_delta=0.9,
        threshold=0.75,
    )

    observer.record(event)

    records = path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    payload = json.loads(records[0])
    assert payload["schema_version"] == TEMPORAL_EVENT_SCHEMA
    assert payload["shot_id"] == "shot-002"
    assert payload["attempt"] == 2
    assert payload["continuity_delta"] == 0.9


def test_observer_failure_is_fail_open_and_counted() -> None:
    class BrokenObserver:
        def record(self, event: TemporalRuntimeEvent) -> None:
            del event
            raise OSError("telemetry unavailable")

    model = NativeTemporalModel.initialized()
    runtime = NativeTemporalRuntime.default(model=model, observer=BrokenObserver())
    state = model.initial_state("shot-observe")

    result = runtime.generate_frame(_frame(0), state)

    assert result.report.accepted is True
    assert state.last_frame_index == 0
    assert state.metadata["temporal_observer_errors"] == 1
