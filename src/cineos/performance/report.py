def inspect_plan(plan):
    return {
        "performance_id": plan.performance_id,
        "shot_id": plan.shot_id,
        "scene_id": plan.scene_id,
        "characters": len(plan.character_ids),
        "dialogue_cues": len(plan.dialogue_cue_ids),
        "beats": len(plan.performance_beats),
        "facial_tracks": len(plan.facial_performance_tracks),
        "lip_sync_tracks": len(plan.lip_sync_tracks),
        "gesture_tracks": len(plan.gesture_tracks),
        "body_tracks": len(plan.body_performance_tracks),
        "eye_line_tracks": len(plan.eye_line_tracks),
        "content_hash": plan.content_hash,
        "lost_capabilities": plan.lost_capabilities,
    }
