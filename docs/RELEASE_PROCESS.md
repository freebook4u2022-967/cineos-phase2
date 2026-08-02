# Controlled Alpha release process

The first candidate is `0.1.0-alpha.1`; it is not production ready. Run pytest,
Ruff, Black, package build, deterministic smoke benchmarks, and release
verification. Release approval requires the mandatory suite, no blocking
regressions, deterministic fixtures, CLI and Studio smoke checks, preview
pipeline, documentation, valid manifest, and package integrity. Real rendering
is optional and must be recorded as hardware-gated rather than silently skipped.

Build wheel and sdist with `python -m build`. Metadata is reproducible; model
weights, copyrighted media, and fonts are excluded. Install the wheel in a new
virtual environment, import public modules, run `cineos version`, `cineos demo`,
`cineos hardware-report`, benchmark smoke, verify outputs/checksums, then
uninstall. Windows, Linux, and macOS can use Python wheels and virtual
environments. No native OS installer is implemented or claimed.
