def subtitle_entry(shot_id: str, text: str, start: float, duration: float) -> dict:
    return {
        "shot_id": shot_id,
        "text": text,
        "start": start,
        "end": start + duration,
        "lip_sync": "approximate_unless_measured",
    }
