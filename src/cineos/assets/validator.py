"""Schema, relationship, and optional media-integrity validation."""

from __future__ import annotations

from .registry import AssetRegistry


class AssetValidator:
    def validate(
        self, registry: AssetRegistry, *, check_files: bool = False
    ) -> list[str]:
        errors = registry.validate()
        if check_files:
            for asset in registry.list():
                for reference in asset.references:
                    errors.extend(
                        f"asset {asset.asset_id}: {error}"
                        for error in reference.validate(check_file=True)
                    )
        return errors

    def raise_for_errors(
        self, registry: AssetRegistry, *, check_files: bool = False
    ) -> None:
        from .exceptions import AssetValidationError

        if errors := self.validate(registry, check_files=check_files):
            raise AssetValidationError(errors)
