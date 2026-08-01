"""Prop and vehicle presence and continuity validation."""

from ._common import scored_result, value
from .base import BaseValidator, ValidationResult, ValidationStatus


class PropValidator(BaseValidator):
    category = "props"

    def validate(self, context) -> ValidationResult:
        props = value(context.conditioning, "prop_conditioning", [])
        if not props:
            return ValidationResult(self.category, ValidationStatus.PASS, 1.0)
        checks = {
            name: context.backend.score(f"props.{name}", props, context.frames)
            for name in (
                "presence",
                "ownership",
                "state",
                "spatial_role",
                "forbidden_substitutions",
            )
        }
        return scored_result(
            self.category, checks, context.thresholds.warning_threshold
        )


VehicleValidator = PropValidator
