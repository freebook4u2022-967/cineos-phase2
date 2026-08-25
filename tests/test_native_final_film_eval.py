from __future__ import annotations

import pytest

from cineos.native_video.final_eval import (
    SceneBoundaryEvalPolicy,
    SceneBoundarySample,
    TemporalFilmEvalPolicy,
    evaluate_sampled_frames,
    evaluate_scene_boundaries,
)


def _frame(value: int, size: int = 16) -> bytes:
    return bytes([value]) * size


def test_final_film_eval_accepts_non_black_motion_evidence() -> None:
    report = evaluate_sampled_frames((_frame(40), _frame(50), _frame(61), _frame(73)))

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.black_frame_ratio == 0.0
    assert report.frozen_transition_ratio == 0.0
    assert 10.0 < report.mean_interframe_mad < 12.0


def test_final_film_eval_rejects_black_regions() -> None:
    report = evaluate_sampled_frames(
        (_frame(0), _frame(0), _frame(0), _frame(70)),
        TemporalFilmEvalPolicy(max_black_ratio=0.25, max_frozen_ratio=1.0),
    )

    assert report.decision == "reject"
    assert report.accepted is False
    assert report.black_frame_ratio == pytest.approx(0.75)
    assert any("black" in directive for directive in report.directives)


def test_final_film_eval_rejects_frozen_sequence() -> None:
    report = evaluate_sampled_frames(
        (_frame(100), _frame(100), _frame(100), _frame(100))
    )

    assert report.decision == "reject"
    assert report.frozen_transition_ratio == 1.0
    assert any("frozen" in directive for directive in report.directives)


def test_final_film_eval_warns_on_excessive_hard_cuts() -> None:
    report = evaluate_sampled_frames(
        (_frame(20), _frame(100), _frame(20), _frame(100)),
        TemporalFilmEvalPolicy(max_frozen_ratio=1.0),
    )

    assert report.decision == "warn"
    assert report.accepted is True
    assert report.hard_cut_transition_ratio == 1.0
    assert any("hard-cut" in directive for directive in report.directives)


def test_final_film_eval_requires_consistent_decoded_frame_size() -> None:
    with pytest.raises(ValueError, match="same non-zero size"):
        evaluate_sampled_frames((b"abc", b"abcd"))


def test_scene_boundary_eval_accepts_intentional_hard_cut() -> None:
    report = evaluate_scene_boundaries(
        (
            SceneBoundarySample(
                from_scene_id="scene-01",
                to_scene_id="scene-02",
                outgoing_frame=_frame(35),
                incoming_frame=_frame(130),
                transition="cut",
            ),
        )
    )

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.reject_count == 0
    assert report.boundaries[0].boundary_mad == pytest.approx(95.0)


def test_scene_boundary_eval_rejects_large_match_cut_drift() -> None:
    report = evaluate_scene_boundaries(
        (
            SceneBoundarySample(
                from_scene_id="scene-01",
                to_scene_id="scene-01b",
                outgoing_frame=_frame(50),
                incoming_frame=_frame(120),
                transition="match",
            ),
        )
    )

    boundary = report.boundaries[0]
    assert report.decision == "reject"
    assert report.accepted is False
    assert boundary.decision == "reject"
    assert any("match boundary" in item for item in boundary.directives)


def test_scene_boundary_eval_warns_on_moderate_match_drift() -> None:
    report = evaluate_scene_boundaries(
        (
            SceneBoundarySample(
                from_scene_id="scene-04",
                to_scene_id="scene-04b",
                outgoing_frame=_frame(80),
                incoming_frame=_frame(105),
                transition="match",
            ),
        )
    )

    assert report.decision == "warn"
    assert report.warn_count == 1
    assert report.boundaries[0].accepted is True


def test_scene_boundary_eval_rejects_black_edge_for_any_transition() -> None:
    report = evaluate_scene_boundaries(
        (
            SceneBoundarySample(
                from_scene_id="scene-09",
                to_scene_id="scene-10",
                outgoing_frame=_frame(0),
                incoming_frame=_frame(150),
                transition="cut",
            ),
        )
    )

    assert report.decision == "reject"
    assert any("near-black" in item for item in report.boundaries[0].directives)


def test_scene_boundary_eval_rejects_abrupt_fade() -> None:
    report = evaluate_scene_boundaries(
        (
            SceneBoundarySample(
                from_scene_id="scene-11",
                to_scene_id="scene-12",
                outgoing_frame=_frame(30),
                incoming_frame=_frame(150),
                transition="fade",
            ),
        ),
        SceneBoundaryEvalPolicy(fade_reject_mad=100.0),
    )

    assert report.decision == "reject"
    assert any("fade boundary" in item for item in report.boundaries[0].directives)


def test_scene_boundary_eval_warns_on_effectively_frozen_cut() -> None:
    report = evaluate_scene_boundaries(
        (
            SceneBoundarySample(
                from_scene_id="scene-20",
                to_scene_id="scene-21",
                outgoing_frame=_frame(100),
                incoming_frame=_frame(100),
                transition="cut",
            ),
        )
    )

    assert report.decision == "warn"
    assert any("effectively frozen" in item for item in report.boundaries[0].directives)


def test_scene_boundary_sample_validates_edit_contract() -> None:
    with pytest.raises(ValueError, match="transition must be"):
        SceneBoundarySample(
            from_scene_id="scene-a",
            to_scene_id="scene-b",
            outgoing_frame=_frame(80),
            incoming_frame=_frame(90),
            transition="dissolve",
        )
