"""Across-frame temporal continuity validation."""

from .base import BaseValidator, ValidationResult, ValidationStatus

TEMPORAL_CHECKS = (
    "frame_flicker",
    "face_drift",
    "body_drift",
    "wardrobe_drift",
    "lighting_jumps",
    "prop_disappearance",
    "duplicate_characters",
    "abrupt_geometry_changes",
    "frame_instability",
)


class TemporalValidator(BaseValidator):
    category = "temporal"

    def validate(self, context) -> ValidationResult:
        metrics = context.backend.temporal_metrics(context.frames)
        if metrics is None:
            return ValidationResult(self.category, ValidationStatus.UNSUPPORTED, None)
        checks = {name: float(metrics.get(name, 0.0)) for name in TEMPORAL_CHECKS}
        instability = max(checks.values(), default=0.0)
        score = 1.0 - min(1.0, max(0.0, instability))
        if score < context.thresholds.temporal_threshold:
            offenders = [
                name
                for name, amount in checks.items()
                if amount > 1 - context.thresholds.temporal_threshold
            ]
            return ValidationResult(
                self.category,
                ValidationStatus.FAIL,
                score,
                checks,
                failures=["temporal drift detected: " + ", ".join(offenders)],
            )
        return ValidationResult(self.category, ValidationStatus.PASS, score, checks)
