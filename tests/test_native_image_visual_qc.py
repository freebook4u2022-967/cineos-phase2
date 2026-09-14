from cineos.native_image.visual_qc import (
    MultiAxisVisualQCGate,
    VisualContinuityObservation,
    build_rerender_directives,
)


def test_clean_visual_continuity_is_accepted():
    report = MultiAxisVisualQCGate().evaluate(
        VisualContinuityObservation(
            shot_id="shot-010",
            scores={
                "face_identity": 0.97,
                "body_shape": 0.94,
                "wardrobe": 0.96,
                "hair": 0.93,
                "props": 0.91,
                "environment": 0.95,
                "lighting": 0.90,
                "screen_direction": 0.92,
            },
        )
    )

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.should_rerender is False


def test_noncritical_continuity_drift_warns_without_forcing_rerender():
    report = MultiAxisVisualQCGate().evaluate(
        VisualContinuityObservation(
            shot_id="shot-011",
            scores={
                "face_identity": 0.96,
                "body_shape": 0.93,
                "wardrobe": 0.81,
                "environment": 0.88,
            },
        )
    )

    assert report.decision == "warn"
    assert report.warning_axes == ("wardrobe",)
    assert report.should_rerender is False


def test_critical_identity_failure_rejects_and_builds_rerender_directives():
    report = MultiAxisVisualQCGate().evaluate(
        VisualContinuityObservation(
            shot_id="shot-012",
            scores={
                "face_identity": 0.55,
                "body_shape": 0.92,
                "wardrobe": 0.95,
                "screen_direction": 0.90,
            },
        )
    )

    assert report.decision == "reject"
    assert report.failed_axes == ("face_identity",)
    assert report.should_rerender is True
    assert build_rerender_directives(report) == (
        "preserve face identity",
        "rerender shot from last accepted continuity state",
    )


def test_unknown_visual_qc_axis_is_rejected():
    try:
        VisualContinuityObservation(
            shot_id="shot-013",
            scores={"unknown_axis": 0.9},
        )
    except ValueError as exc:
        assert "unknown visual QC axes" in str(exc)
    else:
        raise AssertionError("unknown visual QC axis should fail validation")
