from .brief import DirectedSceneBrief

VALID_RENDER_STATUSES = {
    "valid",
    "black_frame_failure",
    "frozen_frame_failure",
    "empty_output",
    "decode_failure",
    "render_exception",
    "manual_review_required",
}


def assembly_ready(shots: list[dict], expected_shots: list[str] | None = None) -> bool:
    """Return true only when every mandatory shot explicitly passed validation."""
    expected = expected_shots or [shot.get("shot_id", "") for shot in shots]
    valid = {
        shot.get("shot_id")
        for shot in shots
        if shot.get("success") and shot.get("content_status") == "valid"
    }
    return bool(expected) and all(shot_id in valid for shot_id in expected)


def validate_brief(brief: DirectedSceneBrief) -> list[str]:
    errors = []
    if len(brief.character_ids) != 1:
        errors.append("Mission One requires exactly one character")
    if len(brief.shots) != 3:
        errors.append("Mission One requires exactly three shots")
    if not 18 <= brief.target_duration <= 24:
        errors.append("target duration must be 18–24 seconds")
    if (
        brief.shots
        and abs(sum(s.duration for s in brief.shots) - brief.target_duration) > 0.01
    ):
        errors.append("shot durations must equal target duration")
    for shot in brief.shots:
        if not 6 <= shot.duration <= 8:
            errors.append(f"{shot.shot_id}: duration must be 6–8 seconds")
        if not shot.action:
            errors.append(f"{shot.shot_id}: one primary action is required")
    return errors
