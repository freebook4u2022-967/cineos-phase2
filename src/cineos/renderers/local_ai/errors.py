"""Actionable local renderer failures."""


class RendererError(RuntimeError):
    """Base error for the isolated backend adapter."""


class EnvironmentValidationError(RendererError):
    """The host cannot safely execute the configured backend."""


class RequestValidationError(RendererError, ValueError):
    """A backend-independent request cannot be represented faithfully."""


class RenderCancelled(RendererError):
    """Rendering was cancelled cooperatively."""
