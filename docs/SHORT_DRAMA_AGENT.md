# CINEOS Short Drama Agent

## Product goal

CINEOS Short Drama Agent is an independent, provider-neutral short-drama creation system. It owns story reasoning, character state, directing decisions, continuity and downstream film planning. External renderers are optional adapters, not the product core.

## Sprint 1 foundation

Sprint 1 established the orchestration boundary:

`DramaBrief -> Story -> Screenplay -> Direction -> Shots -> Continuity -> DramaPlan`

## Sprint 2 creative brain

Sprint 2 turns the skeleton into a richer deterministic creative brain while keeping the contracts local and renderer-independent.

Current creative pipeline:

`DramaBrief -> DramaBrain -> CharacterBrain -> ScreenwriterAgent -> DirectorDecisionEngine -> ShotPlanner -> SceneStateEngine -> ContinuitySupervisor -> DramaPlan`

### Drama Brain

Expands one premise into genre/theme, hook, stakes, reversal, climax, resolution and an emotional curve.

### Character Brain

Creates persistent character profiles containing role, motivation, fear, secret, relationships, knowledge, emotion, physical state, wardrobe and props.

Each generated character is now linked to a deterministic canonical CINEOS character asset. The asset is explicitly marked `pending-approved-reference` for CineDNA. CINEOS does not fabricate face/body identity data or claim a CineDNA profile exists before approved references and explicit identity metadata are supplied.

### Scene State Engine

Carries world and character state across scenes. State changes must be explicit; wardrobe, props, physical condition, knowledge and environment otherwise persist.

### Director Decision Engine

Creates explicit story-first camera and performance decisions for every dramatic beat: shot size, lens, movement, blocking rule, performance intention and lighting intention.

## Production bridge

`DramaPlan` now compiles into the existing CINEOS `MovieProject` and deterministic `FilmPackage` contracts. Character assets are exported through the canonical asset registry and remain ready for the existing CineDNA approval workflow.

The installed command is routed through the Short Drama Agent entrypoint while preserving all existing commands:

```bash
cineos drama create "A man receives a message from his wife who died three years ago." \
  --duration 180 \
  --genre mystery \
  --tone "tense and intimate" \
  --output-dir output/drama
```

The command writes:

- `drama-package.json` — creative brain output and continuity state
- `assets.json` — canonical production assets and CineDNA readiness metadata
- `film-package.json` — verified output of the existing CINEOS Film Compiler

The original `cineos validate`, `compile`, `render`, `film`, `nova`, `audio`, `performance`, benchmark and release commands continue to delegate to the existing CLI.

## Benchmark premise

The regression benchmark remains:

> A man receives a message from his wife who died three years ago.

For a 180-second mystery plan the system must produce five dramatic beats/scenes, character state, five director decisions, five timed shots, a state timeline, a passing continuity report, canonical character assets and a verified Film Package.

## Architectural rule

No renderer-specific prompt format, proprietary video API or external video model is permitted inside the Short Drama Agent core. Future learned CINEOS models can replace deterministic brains behind the same contracts.

## Next targets

1. Add an explicit Character Approval workflow that attaches approved references and completes CineDNA identity metadata.
2. Add pluggable learned creative-brain adapters behind provider-neutral schemas.
3. Expand screenplay generation from beat intent into dialogue, subtext and performance-aware scene text.
4. Add a Short Drama quality benchmark covering hook strength, pacing, continuity and character-state preservation.
5. Begin Atlas Native Renderer research without coupling it to the creative brain.
