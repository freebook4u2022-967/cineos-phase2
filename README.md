# CINEOS Phase 2

CINEOS Phase 2 is the foundation for an open, modular cinematic production
platform. This repository establishes the project structure, engineering
standards, and architectural boundaries that future implementations will use.

Phase 2 is intentionally foundation-first. It does **not** include a renderer
or placeholder AI models. Sprint 2 introduces the renderer-independent core
project model: typed assets, scenes, shots, timeline ordering, and validation.

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

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), follow the
[Code of Conduct](CODE_OF_CONDUCT.md), and report vulnerabilities according to
[SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).
