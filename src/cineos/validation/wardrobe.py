"""Wardrobe and continuity-lock validation."""

from ._common import scored_result, value
from .base import BaseValidator, ValidationResult, ValidationStatus


class WardrobeValidator(BaseValidator):
    category = "wardrobe"

    def validate(self, context) -> ValidationResult:
        wardrobe = value(context.conditioning, "wardrobe_conditioning", [])
        if not wardrobe:
            return ValidationResult(self.category, ValidationStatus.PASS, 1.0)
        checks = {
            name: context.backend.score(f"wardrobe.{name}", wardrobe, context.frames)
            for name in (
                "asset",
                "continuity_locks",
                "colors",
                "components",
                "accessories",
                "allowed_variations",
            )
        }
        return scored_result(
            self.category, checks, context.thresholds.wardrobe_threshold
        )
