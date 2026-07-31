# CINEOS Phase 2

CINEOS Phase 2 is the foundation for an open, modular cinematic production
platform. This repository establishes the project structure, engineering
standards, and architectural boundaries that future implementations will use.

Phase 2 is intentionally foundation-first. It does **not** include a renderer
or placeholder AI models. The renderer-independent core project model provides
typed assets, scenes, shots, timeline ordering, and validation. Sprint 3 adds a
Film Compiler that turns that model into a deterministic, portable Film Package.
Atlas Runtime consumes that package as ordered, renderer-independent work; it
tracks lifecycle and progress while leaving execution to application code.

## Requirements

- Python 3.12
- [pytest](https://docs.pytest.org/) for tests
- [Ruff](https://docs.astral.sh/ruff/) for linting
- [Black](https://black.readthedocs.io/) for formatting

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
| `src/cineos/compiler/` | Deterministic Film Package compilation, serialization, hashing, and validation |
| `src/cineos/atlas/` | Renderer contracts and renderer-independent package runtime orchestration |
| `src/cineos/plugins/` | Generic plugin contracts, discovery, dependencies, and lifecycle management |
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

## Plugin Framework

The plugin framework adds optional capabilities without coupling the core,
compiler, or runtime to a renderer. Each plugin supplies immutable identity,
semantic version, host API compatibility, and dependency metadata, plus
optional load, enable, disable, and unload hooks. `PluginManager` keeps those
states distinct, loads and enables dependencies first, prevents removal of
dependencies that are still in use, and safely treats repeated lifecycle
operations as no-ops.

```python
from cineos.plugins import Plugin, PluginManager, PluginMetadata


class EditorialPlugin(Plugin):
    metadata = PluginMetadata(
        name="editorial-notes", version="1.0.0", api_version="1.0.0"
    )


manager = PluginManager(context={"project": project})
manager.register(EditorialPlugin())
manager.enable("editorial-notes")
manager.disable("editorial-notes")
manager.unload("editorial-notes")
```

Installed distributions can advertise a plugin class, factory, or instance in
the `cineos.plugins` Python entry-point group. Calling `discover()` loads and
registers those entry points in deterministic name order. Discovery is a Python
packaging boundary only; plugins remain generic and renderer-independent.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), follow the
[Code of Conduct](CODE_OF_CONDUCT.md), and report vulnerabilities according to
[SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).
