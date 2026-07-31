"""CineDNA domain errors."""


class CineDNAError(ValueError):
    """Base class for invalid identity-profile operations."""


class MissingIdentityDataError(CineDNAError):
    """Required identity information was not supplied by an approved asset."""


class ConflictingIdentityDataError(CineDNAError):
    """Two identity declarations cannot both be honored."""


class ProfileNotFoundError(CineDNAError, KeyError):
    """A requested profile is not registered."""


class DuplicateProfileError(CineDNAError):
    """A profile version is already registered."""
