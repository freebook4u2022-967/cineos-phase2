# Contributing to CINEOS

Thank you for helping build CINEOS Phase 2. The project is currently focused on
clear contracts and dependable foundations rather than speculative features.

## Development setup

1. Install Python 3.12.
2. Create and activate a virtual environment.
3. Install the project with `python -m pip install -e '.[dev]'`.
4. Run the checks before proposing a change:

   ```bash
   pytest
   ruff check .
   black --check .
   ```

## Making changes

- Keep changes small, focused, and covered by tests where behavior changes.
- Preserve subsystem boundaries documented in `docs/ARCHITECTURE.md`.
- Discuss substantial architectural or public API changes before implementation.
- Do not add generated assets, credentials, local environments, renderer
  implementations, or speculative AI model placeholders.
- Update documentation when behavior, interfaces, or project direction changes.

Commit messages should be imperative and explain one logical change. Pull
requests should state the motivation, summarize the approach, and list the
checks performed.

By participating, you agree to follow the project Code of Conduct.
