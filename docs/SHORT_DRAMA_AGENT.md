# CINEOS Short Drama Agent

## Product goal

CINEOS Short Drama Agent is an independent, provider-neutral short-drama creation system. It owns story reasoning, character state, directing decisions, continuity, character identity approval, native-generation contracts and downstream film planning. External renderers are optional adapters or reference benchmarks, not the product core.

## Current pipeline

`DramaBrief -> DramaBrain -> CharacterBrain -> ScreenwriterAgent -> DirectorDecisionEngine -> ShotPlanner -> SceneStateEngine -> ContinuitySupervisor -> DramaPlan -> MovieProject -> FilmPackage -> ApprovedReferences -> CineDNA -> IdentityLock -> NativeShotRequest -> AtlasNative -> NativeConditioning -> TemporalIdentityMemory -> VisualQC -> AutomaticRerender -> NativeFrameRuntime -> CINEOSLatentFrameModel -> RGBFrame -> TrainingDataset -> TrainableModel -> Optimizer -> Checkpoint -> NativeTemporalModel -> TemporalQC -> TransactionalRetryCommit -> SceneContinuityMemory -> FilmOrchestrator -> FinalAssembly`

The architecture intentionally separates film orchestration from model-specific state. Native temporal continuity plugs into complete-film execution through versioned checkpoint-state and transactional shot-attempt hooks; the film layer does not import native tensor or device semantics.

## Character identity approval

Short Drama character assets are initially created with CineDNA status `pending-approved-reference`. CINEOS does not infer or fabricate a person's face/body identity from a filename or image path.

A character becomes CineDNA-ready only after an approved reference and explicit identity JSON are supplied:

`cineos drama character approve Protagonist --assets assets.json --reference references/protagonist-front.png --identity identity.json --profiles cinedna.json`

The identity JSON must explicitly contain `face` and `body` objects. The approval workflow then:

1. resolves the canonical character asset,
2. records the reference as approved,
3. stores the explicit CineDNA identity data,
4. builds and validates a CineDNA profile,
5. persists the updated asset registry and CineDNA registry.

Multi-reference identity lock, identity-bank training, coverage checks and benchmark gates now sit downstream of this approval boundary so future learned models remain tied to explicitly approved identity evidence.

## Short-drama creation

`cineos drama create "A man receives a message from his wife who died three years ago." --duration 180 --genre mystery --output-dir output/drama`

The command writes:

- `drama-package.json`
- `assets.json`
- `film-package.json`

## Complete-film runtime integration

`FirstFilmRunner` accepts provider-neutral `orchestrator_kwargs`. A native temporal runtime binds its durable scene continuity through `NativeFilmContinuityBridge.orchestrator_kwargs()`.

This gives complete-film execution five explicit boundaries:

1. checkpoint accepted native continuity state,
2. restore accepted continuity state on resume,
3. start each render attempt from the last durable accepted anchor,
4. commit continuity only after whole-shot QC accepts the attempt,
5. discard rejected attempt state without poisoning later shots.

`FirstFilmRunner.run(...)` also accepts `resume` and `checkpoint_path`, so long-running film generation can persist and restore native runtime state without coupling the orchestration layer to a particular renderer generation.

## Native-model status

The repository contains CINEOS-owned trainable-model, latent-frame, neural encoder/decoder, temporal runtime, dataset, checkpoint, distributed-training and evaluation contracts plus executable prototypes that produce real RGB pixels. These establish a real native learning path rather than a proprietary-provider wrapper.

The repository must not claim production visual quality or frontier-model parity until trained weights, GPU-scale experiments and benchmark evidence support those claims. Third-party engines remain optional adapters/reference benchmarks only.

## Architectural rules

1. No renderer-specific proprietary API belongs inside the Short Drama Agent core.
2. Rejected visual/temporal candidates must never advance durable continuity state.
3. Runtime checkpoints store only accepted state at transaction boundaries.
4. Model/checkpoint/dataset contracts are versioned so future learned components can evolve safely.
5. External renderers cannot be presented as the CINEOS native renderer.
6. Production completion requires end-to-end validation, not merely a mock or dry-run path.

## Next targets

1. Bind native temporal state directly into the complete native render execution path and verify resume across real shot retries.
2. Strengthen the automatic pixel observer and scene-boundary QC with generated-frame evidence.
3. Continue the approved-reference identity encoder and scene/text encoder quality path.
4. Advance diffusion/flow-matching latent objectives and neural RGB/VAE decoder quality.
5. Validate GPU training runners, mixed precision, distributed checkpointing and sequence-level temporal training.
6. Add final-film temporal evaluation, audio/dialogue continuity and scene-boundary regression benchmarks.
7. Run full current-head CI and Benchmark smoke before merge; do not claim current-head success without GitHub evidence.
