# Identity and continuity validation

`cineos.validation` checks a completed shot independently of its renderer. It
compares approved conditioning with extracted keyframes in five separate areas:
character identity invariants, wardrobe, props/vehicles, environment, and
across-frame temporal continuity. Scores are aids for production review—not
perfect identity recognition—and the system never identifies real people.

## Workflow

```bash
cineos validate-render shot.mp4 --shot shot-01 \
  --conditioning shot-01.conditioning.json --output report.json
cineos validation show report.json
cineos validation compare previous.mp4 current.mp4
```

Add `--json` before or after the top-level command for machine-readable output.
Reports contain stable category results, warnings and failures, a rerender
recommendation for failed shots, and a SHA-256 content hash. JSON keys are
sorted deterministically. Timestamps and UUIDs intentionally differ per run.

## Backends and optional plugins

`ValidatorBackend` is the only model-facing interface. A backend may implement
face-embedding **comparison against approved fictional-character references**,
image similarity, object presence, temporal consistency, optical flow, or
perceptual hashing. It returns `None` for unsupported capabilities. No CV
library or proprietary service is required; `FakeValidatorBackend` provides a
deterministic test implementation. Unsupported checks remain visible rather
than being treated as successful.

FFmpeg is used to extract sampled PNG keyframes when available. If it is absent,
the container path is passed to a backend so validation can degrade gracefully.
Missing references require manual review. Failed results are never
automatically approved.

## Atlas Runtime

Use `AtlasRuntime.run_with_validation` to invoke a validation callback after
each handler finishes. For structured renderer values,
`ValidationPipeline.validate_render_result` attaches the report and
`mark_for_rerender` flag to `RenderResult.renderer_metadata`. Progress callbacks
receive `(category, fraction)` updates through completion.
