class ColabRendererError(Exception):
    pass


class HardwarePreflightError(ColabRendererError):
    """The Colab GPU does not meet the renderer's minimum requirements."""


class InvalidRenderError(ColabRendererError):
    """A renderer output exists but is not safe to assemble."""


class AssemblyBlockedError(InvalidRenderError):
    """At least one mandatory shot did not pass content validation."""
