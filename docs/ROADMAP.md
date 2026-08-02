# CINEOS Phase 2 Roadmap

- **NOVA Director Alpha:** deterministic planning, critique, targeted revision,
  CLI workflow, and Studio controller integration are implemented.

The roadmap communicates direction rather than delivery dates. Work advances
only when the preceding stage has documented, tested acceptance criteria.

## 1. Foundation

- Establish Python 3.12 packaging and the `cineos` namespace.
- Adopt pytest, Ruff, Black, contribution policies, and security guidance.
- Define subsystem ownership and dependency rules.

## 2. Contracts and schemas

- Identify core project, asset, scene, and job concepts with stakeholders.
- Specify versioned serialization formats and validation behavior.
- Define capability interfaces at subsystem boundaries.
- Add compatibility and migration tests for accepted contracts.

## 3. Reference workflow

- Implement a minimal end-to-end project workflow against stable contracts.
- Add deterministic compilation and actionable diagnostics.
- Record provenance and lifecycle behavior for assets.
- Validate integrations without coupling shared code to product shells.

## 4. Production readiness

- Establish performance baselines with reproducible benchmarks.
- Add threat models, failure recovery, observability, and release automation.
- Document deployment and hardware compatibility expectations.
- Publish support and compatibility policies for stable releases.

## Explicitly deferred

A renderer is not part of the foundation and will require a separately reviewed
design. No AI model is selected, implied, or represented by a placeholder in
this roadmap.

- [x] First backend-neutral complete short-film assembly pipeline

## Studio Alpha

- [x] Native cross-platform PySide6 workspace shell
- [x] Project persistence, metadata, scene/shot ordering, validation and compilation
- [x] Asset, timeline, renderer, queue, review, recovery, and export surfaces
- [x] Thread-pool worker, cancellation, settings persistence, and offscreen launch support
- [ ] Post-alpha rich editors and additional real renderer integrations

Studio Alpha intentionally excludes collaboration, cloud billing, model training,
and nonlinear timeline editing and does not claim production readiness.
