"""Failures raised by the auditable film build pipeline."""


class FilmBuildError(RuntimeError):
    """Base film pipeline failure."""


class AssemblyError(FilmBuildError):
    """A final timeline could not be assembled."""


class ValidationError(FilmBuildError):
    """A project or rendered output failed validation."""


class BuildCancelled(FilmBuildError):
    """The build was cooperatively cancelled."""
