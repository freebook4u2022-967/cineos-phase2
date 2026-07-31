"""Asset subsystem exceptions."""


class AssetError(ValueError):
    """Base error for asset operations."""


class DuplicateAssetError(AssetError):
    """Raised when a stable UUID or unique type/name is already registered."""


class AssetNotFoundError(AssetError, KeyError):
    """Raised when an asset UUID cannot be resolved."""


class InvalidRelationshipError(AssetError):
    """Raised for an unknown endpoint or unsupported typed relationship."""


class AssetValidationError(AssetError):
    """Raised when one or more asset validation rules fail."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))
