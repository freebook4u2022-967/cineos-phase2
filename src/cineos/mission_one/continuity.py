from .brief import DirectedSceneBrief


def propagate_continuity(brief: DirectedSceneBrief) -> None:
    for previous, current in zip(brief.shots, brief.shots[1:]):
        current.previous_shot_continuity = {
            **brief.continuity_locks,
            "previous_shot_id": previous.shot_id,
            "wardrobe_state": previous.wardrobe_state,
            "environment_state": previous.environment_state,
            "prop_state": previous.prop_state,
            **current.previous_shot_continuity,
        }
