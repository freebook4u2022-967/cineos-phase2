# CINEOS Phase 2

## Persistent character identity

CineDNA v1 turns approved character references and explicit identity metadata
into deterministic, versioned profiles without embeddings, recognition, AI
services, or renderer-specific data. See [the CineDNA guide](docs/CINEDNA.md) for
the schema, authoring rules, and CLI commands.

CINEOS Phase 2 is the foundation for an open, modular cinematic production
platform. This repository establishes the project structure, engineering
standards, and architectural boundaries that future implementations will use.

Phase 2 is intentionally foundation-first. It does **not** include a production
renderer or placeholder AI models. The renderer-independent core project model
provides typed assets, scenes, shots, timeline ordering, and validation. Sprint 3 adds a
Film Compiler that turns that model into a deterministic, portable Film Package.
Atlas Runtime consumes that package as ordered, renderer-independent work; it
tracks lifecycle and progress while leaving execution to application code.
The `cineos` command composes these APIs into a deterministic local preview
workflow; preview artifacts are inspection files and do not require a GPU or AI
model. A renderer-independent plugin framework lets optional distributions join
the host through versioned contracts and explicit lifecycle callbacks.

## Requirements

- Python 3.12
- [pytest](https://docs.pytest.org/) for tests
- [Ruff](https://docs.astral.sh/ruff/) for linting
- [Black](https://black.readthedocs.io/) for formatting

## Hardware diagnostics

Use `cineos hardware-report` to inspect a workstation before selecting a future
local AI-video renderer. Add `--json` for deterministic machine output,
`--output hardware-report.json` to save it, or `--verbose` for raw detail.

The command reports OS, CPU, memory, disk, GPU/NVIDIA/CUDA, optional PyTorch
CUDA support, and FFmpeg status on Windows, Linux, and macOS. Missing tools and
CPU-only systems are supported. Render profiles are conservative, non-binding
guidance; the diagnostic neither downloads models nor installs CUDA or a
renderer.

## Getting started

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
black --check .
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/cineos/` | Shared Python package and public interfaces |
| `src/cineos/core/` | Movie project, asset registry, timeline, and validation models |
| `src/cineos/assets/` | UUID asset catalog, references, versions, relationships, and JSON storage |
| `src/cineos/compiler/` | Deterministic Film Package compilation, serialization, hashing, and validation |
| `src/cineos/atlas/` | Renderer contracts and renderer-independent package runtime orchestration |
| `src/cineos/cli/` | Command-line validation, compilation, preview rendering, and assembly adapters |
| `src/cineos/plugins/` | Renderer-independent plugin contracts, discovery, and lifecycle management |
| `src/cineos/hardware/` | Portable hardware probes, reports, and renderer guidance |
| `docs/` | Architecture and project planning |
| `tests/` | Automated tests |
| `scripts/` | Development and automation entry points |
| `atlas/` | Asset and metadata subsystem boundary |
| `nova/` | Creative workflow subsystem boundary |
| `compiler/` | Scene and project compilation boundary |
| `studio/` | Production tooling boundary |
| `renderer/` | Reserved rendering subsystem boundary |
| `hardware/` | Hardware integration boundary |
| `benchmarks/` | Reproducible performance evaluation |

See [the architecture](docs/ARCHITECTURE.md) for dependency principles and
[the roadmap](docs/ROADMAP.md) for planned work.

## Core project model

```python
from cineos.core import MovieProject, ProjectValidator, Scene, Shot, Timeline

shot = Shot("shot-1", action="The door opens.", duration=2.5)
scene = Scene("scene-1", "Arrival", shots=[shot], duration=2.5)
timeline = Timeline()
timeline.add_scene(scene.scene_id)
timeline.add_shot(scene.scene_id, shot.shot_id)
project = MovieProject("Example", "Filmmaker", scenes=[scene], timeline=timeline)

ProjectValidator().raise_for_errors(project)
```

Asset references use stable project-local IDs. Timeline order is explicit and
must mirror the project scene and shot collections. Declared scene durations
must equal the sum of their shot durations.

## Asset and reference management

The `cineos.assets` subsystem provides UUID-identified character, environment,
prop, vehicle, wardrobe, and storyboard assets. Every type supports arbitrary
JSON metadata, searchable tags, multiple reference images, and numbered
snapshots. `AssetRegistry` owns assets and typed directed relationships such as
`character --wears--> wardrobe` or `character --uses--> prop`; it is also
available as `MovieProject.asset_registry`, so project validation checks the
catalog along with scenes and the timeline.

```python
from cineos.assets import AssetRegistry, Character, Wardrobe
from cineos.assets.storage import save

registry = AssetRegistry()
hero = registry.register(Character(name="Hero", tags={"principal"}))
coat = registry.register(Wardrobe(name="Blue coat"))
hero.add_reference("references/hero-front.png", label="front")
hero.create_version("approved concept")
registry.relate(hero, coat, "wears")
save(registry, "assets.json")
```

Registry JSON uses the versioned `cineos-assets-v1` format. The asset CLI can
inspect, validate, and make a normalized deterministic export:

```bash
cineos assets list assets.json
cineos assets validate assets.json
cineos assets export assets.json --output assets-export.json
```

A project JSON selects canonical assets without embedding them by declaring an
external registry path (resolved relative to the project file) and stable UUIDs:

```json
{
  "title": "Example",
  "author": "Filmmaker",
  "asset_registry": "assets.json",
  "asset_ids": ["6fa459ea-ee8a-3ca4-894e-db77e160355e"]
}
```

Compilation validates every selected UUID and writes only identity metadata to
the Film Package. Reference paths, checksums, and media remain in the external
registry.

## Film Compiler

The compiler validates a `MovieProject` before producing a versioned
`FilmPackage`. Its JSON-safe manifests contain project metadata, scenes, shots,
characters, locations, all assets, timeline ordering, and SHA-256 content
hashes. Canonical JSON (sorted keys, UTF-8, and fixed separators) ensures the
same project always produces the same package bytes and hashes.

```python
from cineos.compiler import compile, load, save, verify

package = compile(project)
verify(package)
save(package, "film-package.json")
restored = load("film-package.json")
```

`save(package)` also returns canonical JSON without writing a file, and `load`
accepts canonical JSON or a decoded mapping. Compilation only creates portable
metadata: it does not render media or invoke Atlas or NOVA.

## Atlas Runtime

Atlas Runtime validates a `FilmPackage`, converts its timeline into ordered
shot tasks, and tracks job state, progress, results, cancellation, and errors.
Applications supply a task handler at the integration boundary. The runtime
does not include a renderer, GPU integration, or AI model.

```python
from cineos.atlas import AtlasRuntime

runtime = AtlasRuntime()
job = runtime.execute(package, lambda task: dispatch_to_application(task))
assert job.progress == 1.0
```

Use `prepare()` and `run()` separately to inspect or cancel a pending job before
dispatch. Tasks follow the Film Package timeline rather than manifest insertion
order.

## Command-line interface

After installation, run `cineos --help` or a command-specific `--help` for
examples. Every command supports structured output by placing `--json` before or
after the command (for example, `cineos validate project.json --json`). Usage
errors also use the selected output format. Exit statuses
are stable: 0 succeeds, 2 indicates invalid command usage, 3 an input problem,
4 failed validation, and 5 an execution failure.

```bash
cineos validate project.json
cineos compile project.json --output film-package.json
cineos render film-package.json --output-dir renders
cineos assemble renders --output movie.mp4
cineos demo --output-dir demo-output
cineos version
```

Project JSON uses the core model's field names. A timeline may be omitted, in
which case collection order becomes its explicit order:

```json
{
  "title": "Example",
  "author": "Filmmaker",
  "scenes": [
    {
      "scene_id": "scene-1",
      "title": "Arrival",
      "duration": 2.5,
      "shots": [{"shot_id": "shot-1", "action": "The door opens.", "duration": 2.5}]
    }
  ]
}
```

The preview renderer emits canonical JSON shot artifacts and `assemble` creates
a deterministic preview container with an `.mp4` filename. It is intended for
pipeline verification, not media playback or production-quality encoding.
During rendering, installed CINEOS plugins are activated in stable name order
with explicit access to the renderer registry and Atlas runtime, then deactivated.
The built-in preview renderer follows that same plugin contract rather than being
called directly by the command layer.

The complete integration smoke test is one command:

```bash
cineos demo --output-dir output/demo
```

It writes the source `project.json`, compiled `film-package.json`, Atlas
`runtime-log.json`, per-shot JSON files and `render-manifest.json` under
`renders/`, and the final `demo.mp4` preview container. The path through those
artifacts is `MovieProject` → Film Compiler → `FilmPackage` → Atlas Runtime →
preview renderer plugin → preview assembly.

## Plugin framework

Plugins subclass `Plugin`, declare immutable identity, API compatibility, and
dependency metadata, and can use `activate` and `deactivate` to acquire and
release host resources. `PluginManager` also discovers separately installed
plugins from the `cineos.plugins` Python entry-point group. Registration,
discovery, and lifecycle order are deterministic; dependencies activate first,
while incompatible API versions, missing dependencies, and duplicate names are
rejected. Plugins are enabled by default and may be disabled without unloading
their distributions.

```python
from cineos.plugins import Plugin, PluginContext, PluginManager, PluginMetadata

class EditorialPlugin(Plugin):
    metadata = PluginMetadata("editorial", "1.0.0")

    def activate(self, context: PluginContext) -> None:
        self.catalog = context.services["catalog"]

manager = PluginManager()
manager.register(EditorialPlugin())
manager.activate_all(PluginContext(services={"catalog": catalog}))
```

Plugin context values are copied into read-only mappings. They are generic host
services and settings, not renderer globals: renderers, compilers, runtimes, and
other integrations remain optional services behind their own contracts.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), follow the
[Code of Conduct](CODE_OF_CONDUCT.md), and report vulnerabilities according to
[SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Canonical assets

Canonical production assets, approved visual references, deterministic storage, and asset CLI workflows are documented in [docs/ASSETS.md](docs/ASSETS.md).

## Reference conditioning

A compiled shot can be converted into a deterministic, renderer-independent
`ConditioningPackage`. The contract resolves only approved canonical asset
references and versioned CineDNA; it does not generate embeddings or renderer
prompts. Use `cineos condition build SHOT`, `validate FILE`, `show FILE`, or
`export SHOT --output FILE` (the default source files are `film-package.json`,
`assets.json`, and `cinedna.json`).
