class PerformanceError(ValueError):
    """Base error for invalid or unsupported performance planning."""


class CapabilityNegotiationError(PerformanceError):
    pass


class ExpressionConstraintError(PerformanceError):
    pass


class ContinuityError(PerformanceError):
    pass
