"""Minimal fail-closed validation for rendered MP4 evidence.

CINEOS must not treat arbitrary non-empty bytes with a ``.mp4`` suffix as proof
that a renderer produced video. This module performs a dependency-free ISO BMFF
container sanity check before GPU render receipts are accepted. It is deliberately
conservative and is not a replacement for downstream ffprobe/decoder QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class VideoArtifactError(RuntimeError):
    """Raised when a rendered artifact is not structurally plausible MP4 video."""


@dataclass(frozen=True, slots=True)
class MP4ContainerEvidence:
    """Structural evidence captured from one ISO BMFF/MP4 artifact."""

    box_types: tuple[str, ...]
    has_ftyp: bool
    has_movie_metadata: bool
    has_media_data: bool


def _read_box_header(handle, *, offset: int, file_size: int) -> tuple[int, str, int]:
    handle.seek(offset)
    header = handle.read(8)
    if len(header) != 8:
        raise VideoArtifactError("truncated MP4 box header")

    size32 = int.from_bytes(header[:4], "big")
    try:
        box_type = header[4:8].decode("ascii")
    except UnicodeDecodeError as exc:
        raise VideoArtifactError("MP4 box type is not ASCII") from exc

    header_size = 8
    if size32 == 1:
        extended = handle.read(8)
        if len(extended) != 8:
            raise VideoArtifactError("truncated extended MP4 box size")
        box_size = int.from_bytes(extended, "big")
        header_size = 16
    elif size32 == 0:
        box_size = file_size - offset
    else:
        box_size = size32

    if box_size < header_size:
        raise VideoArtifactError(f"invalid MP4 box size for {box_type!r}")
    if offset + box_size > file_size:
        raise VideoArtifactError(f"MP4 box {box_type!r} exceeds artifact bounds")
    return box_size, box_type, header_size


def inspect_mp4_container(path: str | Path, *, max_boxes: int = 256) -> MP4ContainerEvidence:
    """Validate top-level MP4 structure without loading the whole artifact.

    Accepted evidence requires ``ftyp`` plus media data (``mdat``), and either a
    normal movie metadata box (``moov``) or a fragmented-movie box (``moof``).
    The scan is bounded to avoid pathological files from consuming unbounded work.
    """
    artifact = Path(path)
    try:
        file_size = artifact.stat().st_size
    except OSError as exc:
        raise VideoArtifactError(f"cannot stat rendered video artifact: {artifact}") from exc
    if file_size < 24:
        raise VideoArtifactError("rendered artifact is too small to be a plausible MP4 container")
    if max_boxes <= 0:
        raise ValueError("max_boxes must be positive")

    box_types: list[str] = []
    offset = 0
    try:
        with artifact.open("rb") as handle:
            while offset < file_size and len(box_types) < max_boxes:
                box_size, box_type, _header_size = _read_box_header(
                    handle,
                    offset=offset,
                    file_size=file_size,
                )
                box_types.append(box_type)
                offset += box_size
    except OSError as exc:
        raise VideoArtifactError(f"cannot read rendered video artifact: {artifact}") from exc

    if offset != file_size:
        if len(box_types) >= max_boxes:
            raise VideoArtifactError("MP4 top-level box scan exceeded safety limit")
        raise VideoArtifactError("MP4 artifact ended on a non-box boundary")

    has_ftyp = "ftyp" in box_types
    has_movie_metadata = "moov" in box_types or "moof" in box_types
    has_media_data = "mdat" in box_types
    if not has_ftyp:
        raise VideoArtifactError("rendered artifact is missing MP4 ftyp box")
    if not has_movie_metadata:
        raise VideoArtifactError("rendered artifact is missing MP4 movie metadata")
    if not has_media_data:
        raise VideoArtifactError("rendered artifact is missing MP4 media data")

    return MP4ContainerEvidence(
        box_types=tuple(box_types),
        has_ftyp=has_ftyp,
        has_movie_metadata=has_movie_metadata,
        has_media_data=has_media_data,
    )


__all__ = ["MP4ContainerEvidence", "VideoArtifactError", "inspect_mp4_container"]
