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
- `src/cineos/atlas/` owns renderer contracts and the renderer-independent
  runtime that schedules Film Package tasks.
- `nova/` is reserved for creative workflow coordination.
- `src/cineos/compiler/` owns deterministic validation and transformation of
  core project descriptions into versioned Film Packages.
- `src/cineos/plugins/` owns generic extension metadata, discovery, lifecycle,
  dependency, and compatibility contracts. It has no renderer dependency.
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

## Core project model

`cineos.core` is the operating-system-level domain model and has no renderer or
AI dependencies. `MovieProject` owns production settings and collections of
typed assets and scenes. A `Scene` owns ordered `Shot` values; asset references
are stored as stable IDs rather than object pointers so projects remain easy to
serialize and validate.

`AssetRegistry` allocates project-local IDs and separates characters,
environments, and props. `Timeline` records the canonical scene order and each
scene's shot order. It also checks that a scene's declared duration equals the
sum of its shots. `ProjectValidator` is the cross-object integrity boundary: it
checks empty and duplicate IDs, asset references, durations, and agreement
between the timeline and project collections.

The core deliberately does not generate creative content, compile scenes, or
render frames. Atlas, NOVA, compiler, studio, and renderer integrations consume
the core through explicit interfaces rather than adding their behavior to these
domain values.

## Film Compiler

`cineos.compiler` is a one-way consumer of `cineos.core`. It first applies the
core `ProjectValidator`, then copies the project into a `FilmPackage`; it never
mutates the source project. The package schema includes project metadata plus
scene, shot, character, location, complete asset, and timeline manifests. It is
renderer-independent and contains no generated frames or model output.

The format has an explicit version. Serialization uses canonical JSON with
sorted object keys and fixed separators. SHA-256 hashes cover every manifest
and the complete unhashed package payload, allowing both deterministic builds
and detection of modified content. Loading verifies version, manifest identity
and ordering, structure, and all hashes before returning a package. Unsupported
versions or invalid packages fail with actionable validation errors.

The compiler has no Atlas or NOVA dependency. Future consumers may read the
Film Package contract but must not introduce rendering behavior into compiler
modules.

## Atlas Runtime

`cineos.atlas.runtime` is a one-way consumer of the versioned Film Package
contract. It verifies every package before deriving immutable task views in the
explicit timeline order. Runtime jobs expose pending, running, completed,
failed, and cancelled states together with progress and per-shot results.

Atlas Runtime dispatches tasks only through an application-supplied callable.
It does not select or implement a renderer, allocate GPU resources, load an AI
model, or interpret task results. This keeps orchestration deterministic and
makes local, remote, and future execution integrations replaceable. The
existing renderer SDK remains an optional boundary; runtime modules do not
depend on a concrete backend.

## Plugin Framework

`cineos.plugins` is an optional extension boundary shared by hosts rather than
an Atlas or renderer extension system. Plugins declare immutable metadata: a
unique name, their own semantic version, the supported plugin API version, and
other plugin names on which they depend. Compatibility is checked before
registration, and a host rejects plugins targeting a different API major
version.

`PluginManager` owns discovery and state. It discovers packaging entry points
from the `cineos.plugins` group, but also accepts explicit instances so embedded
and test hosts do not depend on installed package metadata. Loading initializes
a plugin; enabling activates it; disabling leaves it loaded; unloading releases
it. Operations are idempotent, dependencies transition first, cycles and
missing dependencies fail explicitly, and an active dependency cannot be
disabled or unloaded out from under a dependent plugin.

Lifecycle hooks receive an opaque, host-owned context. The framework neither
defines rendering operations nor imports compiler, Atlas Runtime, hardware, or
product shells. A plugin may integrate with those systems only through contracts
provided by its host, preserving dependency direction and renderer independence.

## Data and interface evolution

Persistent and exchanged data formats will be versioned. Readers should reject
unsupported versions with actionable errors; migrations must be explicit and
tested. Public interfaces follow semantic versioning once they are declared
stable.

## Current status

Packaging, quality tooling, the core project model, deterministic Film Package
compilation, Atlas Runtime package task orchestration, and the generic plugin
framework are established. Rendering, GPU integrations, and AI models remain
deliberately outside the current foundation.
