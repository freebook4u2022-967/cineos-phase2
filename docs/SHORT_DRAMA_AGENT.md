# CINEOS Short Drama Agent

## Product goal

CINEOS Short Drama Agent is an independent, provider-neutral short-drama creation system. It owns story reasoning, character state, directing decisions, continuity and downstream film planning. External renderers are optional adapters, not the product core.

## Sprint 1 foundation

Sprint 1 established the orchestration boundary:

`DramaBrief -> Story -> Screenplay -> Direction -> Shots -> Continuity -> DramaPlan`

## Sprint 2 creative brain

Sprint 2 turns the skeleton into a richer deterministic creative brain while keeping the contracts local and renderer-independent.

Current pipeline:

`DramaBrief -> DramaBrain -> CharacterBrain -> ScreenwriterAgent -> DirectorDecisionEngine -> ShotPlanner -> SceneStateEngine -> ContinuitySupervisor -> DramaPlan`

### Drama Brain

Expands one premise into genre/theme, hook, stakes, reversal, climax, resolution and an emotional curve.

### Character Brain

Creates persistent character profiles containing role, motivation, fear, secret, relationships, knowledge, emotion, physical state, wardrobe and props. These profiles are designed to link to CineDNA in a later integration sprint.

### Scene State Engine

Carries world and character state across scenes. State changes must be explicit; wardrobe, props, physical condition, knowledge and environment otherwise persist.

### Director Decision Engine

Creates explicit story-first camera and performance decisions for every dramatic beat: shot size, lens, movement, blocking rule, performance intention and lighting intention.

### Drama package adapter

`cineos.short_drama.cli.create_drama_plan()` and `write_drama_plan()` expose a JSON-safe package boundary. The top-level `cineos drama create` parser wiring is the next CLI integration step; the creative engine itself does not depend on the command line.

## Benchmark premise

The regression benchmark remains:

> A man receives a message from his wife who died three years ago.

For a 180-second mystery plan the system must produce five dramatic beats/scenes, character state, five director decisions, five timed shots, a state timeline and a passing continuity report.

## Architectural rule

No renderer-specific prompt format, proprietary video API or external video model is permitted inside the Short Drama Agent core. Future learned CINEOS models can replace deterministic brains behind the same contracts.

## Next integration targets

1. Wire `cineos drama create` into the top-level CLI.
2. Link CharacterProfile identities to CineDNA/assets.
3. Compile DramaPlan scenes/shots into MovieProject and FilmPackage.
4. Add pluggable learned creative-brain adapters behind provider-neutral schemas.
5. Begin Atlas Native Renderer research without coupling it to the creative brain.
