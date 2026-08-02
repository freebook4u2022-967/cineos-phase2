"""Release smoke checks."""

import importlib

PUBLIC_MODULES = ("cineos", "cineos.benchmarks", "cineos.release")


def import_smoke_test() -> tuple[str, ...]:
    failures = []
    for name in PUBLIC_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as error:
            failures.append(f"{name}: {error}")
    return tuple(failures)
