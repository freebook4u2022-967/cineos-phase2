"""Production integrity gate for Diffusers-backed video execution.

The generic Diffusers renderer intentionally remains injectable and lightweight for
unit tests and research adapters. Production foundation profiles use the subclass in
this module so a renderer cannot report success merely because an exporter returned:
the resulting artifact must also be structurally plausible MP4 evidence.
"""

from __future__ import annotations

from typing import Any

from .diffusers_video import DiffusersVideoError, DiffusersVideoRenderer, DiffusersVideoResult
from .video_artifact import VideoArtifactError, inspect_mp4_container


class ValidatedDiffusersVideoRenderer(DiffusersVideoRenderer):
    """Diffusers renderer that fails closed on malformed or empty MP4 artifacts."""

    def render(self, request: Any) -> DiffusersVideoResult:
        result = super().render(request)
        try:
            inspect_mp4_container(result.output_path)
        except VideoArtifactError as exc:
            raise DiffusersVideoError(
                "rendered video artifact failed MP4 integrity validation"
            ) from exc
        return result


__all__ = ["ValidatedDiffusersVideoRenderer"]
