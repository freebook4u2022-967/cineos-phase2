"""Environment appearance and spatial-continuity validation."""

from ._common import scored_result, value
from .base import BaseValidator, ValidationResult, ValidationStatus


class EnvironmentValidator(BaseValidator):
    category = "environment"

    def validate(self, context) -> ValidationResult:
        environment = value(context.conditioning, "environment_conditioning")
        if environment is None:
            return ValidationResult(
                self.category,
                ValidationStatus.MANUAL_REVIEW_REQUIRED,
                None,
                warnings=["no approved environment reference was supplied"],
            )
        checks = {
            name: context.backend.score(
                f"environment.{name}", environment, context.frames
            )
            for name in (
                "asset",
                "time_of_day",
                "weather",
                "lighting",
                "architecture",
                "spatial_continuity",
                "background",
            )
        }
        return scored_result(
            self.category, checks, context.thresholds.environment_threshold
        )
