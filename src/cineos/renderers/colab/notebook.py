from pathlib import Path

from .package import ColabRenderPackage
from .serializer import load_json


def load_package(path: Path) -> ColabRenderPackage:
    return ColabRenderPackage.from_dict(load_json(path))
