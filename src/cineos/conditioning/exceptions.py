"""Errors raised by the renderer-independent conditioning contract."""


class ConditioningError(ValueError):
    """Base conditioning error."""


class ConditioningBuildError(ConditioningError):
    """A package could not be built without inventing source data."""


class ConditioningValidationError(ConditioningError):
    """A conditioning package is invalid."""


class UnsupportedRendererCapabilities(ConditioningValidationError):
    """A renderer cannot satisfy a package's declared requirements."""
