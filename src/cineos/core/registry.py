"""Project-local asset registry."""

import re
from collections.abc import Iterator

from .asset import Asset, Character, Environment, Prop


class AssetRegistry:
    """Register typed assets and allocate deterministic unique identifiers."""

    def __init__(self) -> None:
        self.characters: dict[str, Character] = {}
        self.environments: dict[str, Environment] = {}
        self.props: dict[str, Prop] = {}

    def generate_unique_id(self, name: str, *, prefix: str = "asset") -> str:
        stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or prefix
        candidate = f"{prefix}-{stem}"
        used = {asset.asset_id for asset in self.assets()}
        suffix = 2
        while candidate in used:
            candidate = f"{prefix}-{stem}-{suffix}"
            suffix += 1
        return candidate

    def register_character(
        self, character: Character | str, *, description: str = ""
    ) -> Character:
        asset = self._coerce(character, Character, "character", description)
        return self._register(self.characters, asset)

    def register_environment(
        self, environment: Environment | str, *, description: str = ""
    ) -> Environment:
        asset = self._coerce(environment, Environment, "environment", description)
        return self._register(self.environments, asset)

    register_location = register_environment

    def register_prop(self, prop: Prop | str, *, description: str = "") -> Prop:
        asset = self._coerce(prop, Prop, "prop", description)
        return self._register(self.props, asset)

    def assets(self) -> Iterator[Asset]:
        yield from self.characters.values()
        yield from self.environments.values()
        yield from self.props.values()

    def _coerce(self, value, asset_type, prefix: str, description: str):
        if isinstance(value, asset_type):
            return value
        return asset_type(
            self.generate_unique_id(value, prefix=prefix), value, description
        )

    def _register(self, collection: dict[str, Asset], asset: Asset):
        if any(existing.asset_id == asset.asset_id for existing in self.assets()):
            raise ValueError(f"duplicate asset ID: {asset.asset_id}")
        collection[asset.asset_id] = asset
        return asset
