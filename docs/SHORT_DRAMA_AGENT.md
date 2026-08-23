# CINEOS Short Drama Agent

## Sprint 1 goal

Turn one creative brief into a deterministic, renderer-independent short-drama plan while preserving the existing CINEOS core boundaries.

Pipeline:

`DramaBrief -> StoryArchitect -> ScreenwriterAgent -> DirectorAgent -> ShotPlanner -> ContinuitySupervisor -> DramaPlan`

The five agents are provider-neutral contracts. Sprint 1 deliberately does not call an LLM or renderer; future adapters can supply model-backed intelligence without coupling story logic to Atlas or a specific video provider.

## Acceptance target

A three-minute mystery premise can be transformed into an ordered story structure, screenplay beats, direction metadata, shot plan, and continuity report through one `ShortDramaOrchestrator.plan()` call.

## Next sprint

1. Add structured character and world state linked to CineDNA/assets.
2. Add model-backed story/screenplay adapters with deterministic schemas.
3. Compile planned shots into existing CINEOS project/FilmPackage contracts.
4. Add renderer prompt adapters rather than renderer-specific data to the core.
5. Add post-render QC/retry policy using existing validation infrastructure.
