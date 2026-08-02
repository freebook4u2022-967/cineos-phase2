def validate_timeline(items):
    errors = []
    for item in items:
        if item.start_time < 0 or item.end_time < item.start_time:
            errors.append(
                f"invalid timing for {getattr(item, 'character_id', 'track')}"
            )
    return errors
