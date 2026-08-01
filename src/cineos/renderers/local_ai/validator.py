"""Complete request validation before inference."""

from pathlib import Path

from cineos.conditioning import ConditioningValidator

from .errors import RequestValidationError
from .request import RenderRequest


def validate_request(
    request: RenderRequest, conditioning, capabilities, asset_ids: set[str]
) -> None:
    problems: list[str] = []
    if request.output_path.suffix.lower() != ".mp4":
        problems.append("output format must be MP4")
    if request.shot_id != conditioning.shot_id:
        problems.append("conditioning shot_id does not match requested shot")
    unknown = set(request.approved_reference_ids) - asset_ids
    if unknown:
        problems.append(
            "unapproved or missing asset references: " + ", ".join(sorted(unknown))
        )
    if (
        request.cinedna_ids
        or conditioning.renderer_capability_requirements.face_identity_support
    ):
        problems.append(
            "unsupported identity conditioning: this backend has no "
            "identity/reference adapter"
        )
    try:
        ConditioningValidator().raise_for_errors(conditioning)
        ConditioningValidator().validate_renderer(conditioning, capabilities)
        capabilities.negotiate(
            resolution=(request.width, request.height),
            duration=request.duration,
            fps=request.fps,
        )
    except ValueError as error:
        problems.append(str(error))
    parent = Path(request.output_path).expanduser().parent
    if parent.exists() and not parent.is_dir():
        problems.append("output parent is not a directory")
    if problems:
        raise RequestValidationError("; ".join(problems))
