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
- `src/cineos/assets/` owns globally unique production-asset identity,
  references, revision snapshots, relationships, validation, and portable
  storage. The core project consumes its registry through an explicit field.
- `nova/` is reserved for creative workflow coordination.
- `src/cineos/compiler/` owns deterministic validation and transformation of
  core project descriptions into versioned Film Packages.
- `studio/` is reserved for user-facing production tools and orchestration.
- `renderer/` marks the rendering boundary. No renderer is implemented in this
  foundation.
- `hardware/` is reserved for explicit hardware capability and integration
  adapters.
- `src/cineos/hardware/` owns immutable diagnostic values, dependency-optional
  OS/GPU probes, deterministic reports, and conservative renderer guidance.
- `benchmarks/` holds documented, reproducible performance workloads rather
  than product code.
- `scripts/` holds thin development and automation entry points. Reusable logic
  belongs in a tested package.
- `src/cineos/cli/` is the product shell. It converts JSON at the boundary and
  composes core validation, Film Compiler, Atlas Runtime, and renderer contracts;
  domain rules remain in their owning packages.
- `src/cineos/plugins/` owns optional-extension identity, compatibility,
  discovery, and lifecycle contracts. It depends on no product subsystem.

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

## Command-line integration

The CLI is a one-way consumer of the core, compiler, and Atlas public APIs.
`validate` and `compile` deserialize project JSON into the Core Project Model;
validation and compilation are delegated to their existing services. `render`
loads a verified Film Package and sends timeline tasks through Atlas Runtime to
the deterministic preview handler. `assemble` consumes that handler's manifest.
`demo` composes all of those stages with a built-in minimal project.

At the render boundary the CLI discovers plugins through `PluginManager` and
activates them with an immutable context containing the renderer registry and
Atlas runtime. Lifecycle ordering remains deterministic, and all plugins are
deactivated after execution, including when rendering fails.

The preview format deliberately contains no generated imagery: it exists to
exercise orchestration reproducibly without GPU support or an AI model. Stable
exit codes and optional JSON messages make the shell suitable for CI and other
automation. Files are written in package/timeline order with canonical JSON so
the command layer does not weaken deterministic build guarantees.

The `hardware-report` command is a one-way consumer of `cineos.hardware`. The
hardware package treats command failures and missing optional libraries as
data, invokes subprocesses without a shell, and has no renderer or model
dependency. Recommendations are cautious guidance rather than capability
guarantees; this subsystem does not install drivers, CUDA, or models.

## Plugin framework

`cineos.plugins` is a small host boundary for optional, separately distributed
extensions. A plugin declares immutable name, version, description, and plugin
API version metadata, plus optional dependencies on other plugins. The manager
rejects duplicate names and incompatible API versions, exposes stable name
ordering, activates dependencies first, and invokes idempotent activation and
deactivation callbacks with a host-created `PluginContext`. Context mappings
are copied and read-only so plugins cannot rewrite the host's service registry.

Discovery uses only the `cineos.plugins` packaging entry-point group; importing
the framework never scans the filesystem or imports optional plugins. Hosts may
also register instances directly, which supports embedded applications and
deterministic tests. Bulk activation rolls back plugins activated by that call
if a later callback fails, while failures retain their original exception as
the cause. Hosts can enable or disable registered plugins independently of
discovery; disabling an active plugin deactivates it and its active dependants
first.

The framework has no renderer, GPU, AI, Atlas Runtime, Film Compiler, or CLI
dependency. A host may deliberately expose one of those APIs as a context
service, but plugins consume its public contract and the framework does not
select a backend. This preserves replaceable subsystem boundaries and prevents
plugin discovery from becoming a hidden rendering pipeline.

## Data and interface evolution

Persistent and exchanged data formats will be versioned. Readers should reject
unsupported versions with actionable errors; migrations must be explicit and
tested. Public interfaces follow semantic versioning once they are declared
stable.

## Current status

Packaging, quality tooling, the core project model, deterministic Film Package
compilation, Atlas Runtime orchestration, the execution CLI, and the optional
plugin framework are established. Rendering, GPU integrations, and AI models
remain deliberately outside the current foundation.
