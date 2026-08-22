from pathlib import Path

from .package import ColabRenderPackage
from .serializer import load_json


def production_shots(package: ColabRenderPackage, smoke_mode: bool = False):
    """Return one shot for a cheap hardware smoke test, otherwise the full package."""
    return package.shots[:1] if smoke_mode else package.shots


def load_package(path: Path) -> ColabRenderPackage:
    return ColabRenderPackage.from_dict(load_json(path))
