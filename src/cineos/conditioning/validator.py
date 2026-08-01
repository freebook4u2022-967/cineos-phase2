"""Structural and renderer capability validation."""

from .exceptions import ConditioningValidationError, UnsupportedRendererCapabilities
from .package import CONDITIONING_SCHEMA_VERSION, ConditioningPackage
from .serializer import calculate_content_hash


class ConditioningValidator:
    def validate(self, package: ConditioningPackage) -> list[str]:
        errors: list[str] = []
        if package.schema_version != CONDITIONING_SCHEMA_VERSION:
            errors.append(f"unsupported schema version: {package.schema_version}")
        if not package.shot_id or not package.scene_id:
            errors.append("shot_id and scene_id are required")
        if not package.approved_reference_ids:
            errors.append("at least one approved reference is required")
        if package.camera_conditioning.duration < 0:
            errors.append("camera duration cannot be negative")
        if package.content_hash and package.content_hash != calculate_content_hash(
            package
        ):
            errors.append("content hash does not match canonical content")
        return errors

    def raise_for_errors(self, package: ConditioningPackage) -> None:
        if errors := self.validate(package):
            raise ConditioningValidationError("; ".join(errors))

    def validate_renderer(self, package: ConditioningPackage, capabilities) -> None:
        req = package.renderer_capability_requirements
        features = capabilities.supported_features
        wanted = {
            name
            for needed, name in (
                (req.image_reference_support, "image-reference"),
                (req.multi_reference_support, "multi-reference"),
                (req.face_identity_support, "face-identity"),
                (req.control_image_support, "control-image"),
                (req.motion_reference_support, "motion-reference"),
            )
            if needed
        }
        problems = sorted(wanted - features)
        try:
            capabilities.negotiate(
                resolution=req.supported_resolution,
                duration=req.maximum_duration,
                fps=req.supported_fps,
                features=wanted,
            )
        except ValueError as error:
            problems.append(str(error))
        maximum = getattr(capabilities, "maximum_character_count", None)
        if maximum is not None and req.character_count > maximum:
            problems.append(f"character count {req.character_count} exceeds {maximum}")
        if problems:
            raise UnsupportedRendererCapabilities(
                "renderer does not meet conditioning requirements: "
                + "; ".join(problems)
            )


def validate_renderer_capabilities(package, capabilities) -> None:
    ConditioningValidator().validate_renderer(package, capabilities)
