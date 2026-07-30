# CINEOS Phase 2

CINEOS Phase 2 is the foundation for an open, modular cinematic production
platform. This repository establishes the project structure, engineering
standards, and architectural boundaries that future implementations will use.

Phase 2 is intentionally foundation-first. It does **not** include a renderer
or placeholder AI models.

## Atlas Runtime

The Atlas Runtime queues and executes infrastructure tasks while tracking job
IDs, lifecycle state, progress, retries, cancellation, logs, and event
callbacks. Tasks are application-supplied Python callables; the runtime itself
performs no rendering and uses no GPU or AI model.

```python
from cineos.runtime import AtlasRuntime, RenderJob

def prepare(context):
    context.report_progress(0.5)
    return {"status": "prepared"}

with AtlasRuntime() as runtime:
    job = RenderJob(prepare, max_retries=1)
    runtime.submit(job)
    runtime.run_next().result()
```

Configuration can be loaded from a JSON file with `load_config`. The environment
variables `CINEOS_RUNTIME_WORKERS`, `CINEOS_RUNTIME_LOG_LEVEL`, and
`CINEOS_RUNTIME_SHUTDOWN_TIMEOUT` override file values.

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

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), follow the
[Code of Conduct](CODE_OF_CONDUCT.md), and report vulnerabilities according to
[SECURITY.md](SECURITY.md).

## License

This project is licensed under the [MIT License](LICENSE).
