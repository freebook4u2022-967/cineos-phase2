# CINEOS Phase 2

CINEOS Phase 2 is the foundation for an open, modular cinematic production
platform. This repository establishes the project structure, engineering
standards, and architectural boundaries that future implementations will use.

Phase 2 is intentionally foundation-first. Sprint 3 adds a deterministic Film
Compiler, but does **not** include a renderer or placeholder AI models.

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
| `src/cineos/compiler/` | Deterministic Film Package compiler and persistence |
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

## Film Compiler

The compiler accepts a `MovieProject`-like dataclass or object, or a mapping,
with `metadata`, `scenes`, `shots`, `characters`, `locations`, `assets`, and
`timeline` fields. It produces a versioned `FilmPackage` containing normalized
manifests and SHA-256 content hashes. Compilation and compact JSON output are
deterministic, including when entity manifest input order differs.

```python
from cineos.compiler import compile, load, save, verify

project = {
    "metadata": {"title": "Example", "fps": 24},
    "scenes": [{"id": "scene-1"}],
    "shots": [{"id": "shot-1", "scene_id": "scene-1"}],
    "characters": [],
    "locations": [],
    "assets": [],
    "timeline": [{"shot_id": "shot-1", "start_frame": 0}],
}

package = compile(project)
assert verify(package)
save(package, "example.film.json")
restored = load("example.film.json")
```

`load` validates the format version, required manifest shapes, and every
content hash. The compiler only constructs portable package data; it performs
no rendering and has no Atlas or NOVA integration.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), follow the
[Code of Conduct](CODE_OF_CONDUCT.md), and report vulnerabilities according to
[SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).
