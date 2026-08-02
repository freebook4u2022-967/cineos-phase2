"""Semantic version parsing."""

import re

SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)
ALPHA_VERSION = "0.1.0-alpha.1"


def is_semantic_version(value: str) -> bool:
    return SEMVER.fullmatch(value) is not None
