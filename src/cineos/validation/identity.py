"""Character identity-invariant validation (not person identification)."""

from ._common import scored_result, value
from .base import BaseValidator, ValidationResult, ValidationStatus


class IdentityValidator(BaseValidator):
    category = "identity"

    def validate(self, context) -> ValidationResult:
        characters = value(context.conditioning, "character_conditioning", [])
        if not characters:
            return ValidationResult(
                self.category,
                ValidationStatus.MANUAL_REVIEW_REQUIRED,
                None,
                warnings=["no approved character references were supplied"],
            )
        checks = {}
        for name in ("face", "body", "hairstyle", "facial_marks", "accessories"):
            checks[name] = context.backend.score(
                f"identity.{name}", characters, context.frames
            )
        return scored_result(
            self.category, checks, context.thresholds.identity_threshold
        )
