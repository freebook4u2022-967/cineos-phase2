# Quality path

## Current level

Mission One provides basic real AI video generation with limited action obedience, limited identity consistency, and approximate lip-sync. It is not Seedance-level output. Scores are never fabricated: action, camera, continuity, identity, and temporal stability require measurements or visual review.

## Measurable improvement path

1. Adopt a stronger image-to-video renderer when reference input is truly supported.
2. Add pose and motion conditioning and facial performance control.
3. Add a measured lip-sync renderer and identity adapter.
4. Expand shot validation, then automatic rerender against explicit failures.
5. Add upscaling, temporal repair, color, and sound finishing.

Each upgrade must record backend capability, measured evidence, and limitations rather than silently lowering constraints.
