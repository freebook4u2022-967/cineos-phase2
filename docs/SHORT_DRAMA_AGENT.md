# CINEOS Short Drama Agent

## Product goal

CINEOS Short Drama Agent is an independent, provider-neutral short-drama creation system. It owns story reasoning, character state, directing decisions, continuity, character identity approval and downstream film planning. External renderers are optional adapters, not the product core.

## Current pipeline

`DramaBrief -> DramaBrain -> CharacterBrain -> ScreenwriterAgent -> DirectorDecisionEngine -> ShotPlanner -> SceneStateEngine -> ContinuitySupervisor -> DramaPlan -> MovieProject -> FilmPackage`

## Sprint 3 character identity approval

Short Drama character assets are initially created with CineDNA status `pending-approved-reference`. CINEOS does not infer or fabricate a person's face/body identity from a filename or image path.

A character becomes CineDNA-ready only after an approved reference and explicit identity JSON are supplied:

`cineos drama character approve Protagonist --assets assets.json --reference references/protagonist-front.png --identity identity.json --profiles cinedna.json`

The identity JSON must explicitly contain `face` and `body` objects. The approval workflow then:

1. resolves the canonical character asset,
2. records the reference as approved,
3. stores the explicit CineDNA identity data,
4. builds and validates a CineDNA profile,
5. persists the updated asset registry and CineDNA registry.

This provides a stable identity boundary for future native rendering and character-consistency systems.

## Short-drama creation

`cineos drama create "A man receives a message from his wife who died three years ago." --duration 180 --genre mystery --output-dir output/drama`

The command writes:

- `drama-package.json`
- `assets.json`
- `film-package.json`

## Architectural rule

No renderer-specific prompt format, proprietary video API or external video model is permitted inside the Short Drama Agent core. Future learned CINEOS models and Atlas Native Renderer components must integrate behind provider-neutral contracts.

## Next targets

1. Multi-reference character approval and reference ranking.
2. Character consistency conditioning from approved CineDNA.
3. Richer screenplay dialogue, subtext and performance generation.
4. Short Drama quality benchmark and automatic QC.
5. Atlas Native Renderer research and native shot generation.
