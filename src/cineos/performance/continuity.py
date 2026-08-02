from dataclasses import dataclass, field


@dataclass(slots=True)
class ContinuityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self):
        return not self.errors


def validate_continuity(previous, current):
    report = ContinuityReport()
    checks = (
        ("gaze", "eye_line_tracks"),
        ("gesture", "gesture_tracks"),
        ("posture", "body_performance_tracks"),
        ("emotional", "emotional_arc"),
        ("prop-hand", "gesture_tracks"),
        ("screen-direction", "eye_line_tracks"),
    )
    for label, key in checks:
        expected = previous.continuity_outputs.get(label)
        actual = current.continuity_inputs.get(label)
        if expected is not None and actual != expected:
            report.errors.append(f"{label} continuity mismatch")
    for beat in current.performance_beats:
        if beat.reaction and beat.start_time < 0:
            report.errors.append("dialogue reaction timing is invalid")
    return report
