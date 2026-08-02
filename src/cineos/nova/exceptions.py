"""NOVA domain errors."""


class NOVAError(ValueError):
    """Base error raised for an invalid directing plan."""


class MissingAssetError(NOVAError):
    """An approved asset required by the brief is unavailable."""


class PlanValidationError(NOVAError):
    """A generated plan failed production validation."""


class PlannerNotFoundError(NOVAError):
    """The requested planning provider is not registered."""
