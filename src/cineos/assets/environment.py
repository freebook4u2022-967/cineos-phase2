"""Environment assets."""

from dataclasses import dataclass

from .asset import Asset


@dataclass(slots=True, kw_only=True)
class Environment(Asset):
    """A location or set."""

    asset_type = "environment"
