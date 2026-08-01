"""Configurable score thresholds."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    """Scores are normalized to ``0..1``; larger is more consistent."""

    pass_threshold: float = 0.85
    warning_threshold: float = 0.65
    fail_threshold: float = 0.40
    identity_threshold: float = 0.80
    wardrobe_threshold: float = 0.80
    temporal_threshold: float = 0.75
    environment_threshold: float = 0.75

    def __post_init__(self) -> None:
        values = (
            vars(self)
            if hasattr(self, "__dict__")
            else {name: getattr(self, name) for name in self.__slots__}
        )
        if any(not 0 <= value <= 1 for value in values.values()):
            raise ValueError("validation thresholds must be between zero and one")
        if not self.pass_threshold >= self.warning_threshold >= self.fail_threshold:
            raise ValueError("thresholds must be ordered pass >= warning >= fail")
