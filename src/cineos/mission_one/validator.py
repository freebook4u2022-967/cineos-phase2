from .brief import DirectedSceneBrief


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
