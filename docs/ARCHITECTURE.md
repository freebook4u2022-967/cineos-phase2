# CINEOS Phase 2 Architecture

## Purpose

Phase 2 establishes a modular foundation for cinematic production workflows.
This document defines boundaries and dependency rules before subsystem
implementations are introduced.

## Principles

1. **Contracts before implementations.** Public interfaces and data ownership
   must be explicit before integrations are built.
2. **One-way dependencies.** Product-facing tools depend on shared contracts;
   shared contracts do not depend on product-facing tools.
3. **Reproducibility.** Compilation, tests, and benchmarks must be deterministic
   when given the same declared inputs.
4. **Replaceable subsystems.** Integration occurs through versioned interfaces,
   not knowledge of another subsystem's internals.
5. **Secure defaults.** External input is untrusted, privileges are minimized,
   and credentials never enter project artifacts.

## Repository boundaries

- `src/cineos/` contains the shared Python namespace, stable value types, and
  cross-subsystem interfaces. It must remain independent of product shells.
- `atlas/` owns asset identity, metadata, and provenance concerns.
- `nova/` is reserved for creative workflow coordination.
- `compiler/` is reserved for deterministic validation and transformation of
  project descriptions into runtime-ready representations.
- `studio/` is reserved for user-facing production tools and orchestration.
- `renderer/` marks the rendering boundary. No renderer is implemented in this
  foundation.
- `hardware/` is reserved for explicit hardware capability and integration
  adapters.
- `benchmarks/` holds documented, reproducible performance workloads rather
  than product code.
- `scripts/` holds thin development and automation entry points. Reusable logic
  belongs in a tested package.

## Dependency direction

Subsystems may depend on contracts exposed by `cineos`, but `cineos` must not
import subsystem implementations. Studio-level orchestration may compose
subsystems through those contracts. Hardware-specific behavior stays behind
adapters, and benchmark code remains outside runtime dependencies.

Dependencies between top-level subsystems require an explicit interface and an
architecture review. Cycles between subsystems are prohibited.

## Data and interface evolution

Persistent and exchanged data formats will be versioned. Readers should reject
unsupported versions with actionable errors; migrations must be explicit and
tested. Public interfaces follow semantic versioning once they are declared
stable.

## Current status

Only packaging, quality tooling, documentation, and a minimal importable Python
namespace are established. Runtime pipelines, renderer behavior, and AI models
are deliberately outside the current foundation.
