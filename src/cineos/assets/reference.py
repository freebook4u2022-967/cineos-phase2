"""Approved media reference records; media itself is never stored."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class ViewType(StrEnum):
    FRONT = "front"
    THREE_QUARTER = "three-quarter"
    LEFT_PROFILE = "left-profile"
    RIGHT_PROFILE = "right-profile"
    REAR = "rear"
    FULL_BODY = "full-body"
    CLOSE_UP = "close-up"
    EXPRESSION = "expression"
    WARDROBE = "wardrobe"
    PROP = "prop"
    ENVIRONMENT = "environment"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(slots=True, init=False)
class ReferenceImage:
    reference_id: UUID
    file_path: str
    media_type: str
    view_type: str
    checksum: str
    dimensions: tuple[int, int] | None
    approval_status: str
    notes: str
    priority: int
    source: str
    metadata: dict[str, Any]

    def __init__(
        self,
        file_path: str | None = None,
        label: str = "",
        metadata: dict[str, Any] | None = None,
        *,
        uri: str | None = None,
        reference_id: UUID | str | None = None,
        media_type: str = "",
        view_type: str = "front",
        checksum: str = "",
        dimensions: tuple[int, int] | list[int] | None = None,
        approval_status: str = "pending",
        notes: str = "",
        priority: int = 0,
        source: str = "",
    ) -> None:
        self.reference_id = UUID(str(reference_id)) if reference_id else uuid4()
        self.file_path = file_path if file_path is not None else (uri or "")
        self.media_type = (
            media_type or mimetypes.guess_type(self.file_path)[0] or "image/*"
        )
        self.view_type = str(view_type)
        self.checksum = checksum.lower()
        self.dimensions = tuple(dimensions) if dimensions else None  # type: ignore[assignment]
        self.approval_status = str(approval_status)
        self.notes = notes or label
        self.priority = priority
        self.source = source
        self.metadata = dict(metadata or {})

    @property
    def uri(self) -> str:
        return self.file_path

    @property
    def label(self) -> str:
        return self.notes

    def copy(self) -> ReferenceImage:
        return replace(self, metadata=dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": str(self.reference_id),
            "file_path": self.file_path,
            "media_type": self.media_type,
            "view_type": self.view_type,
            "checksum": self.checksum,
            "dimensions": list(self.dimensions) if self.dimensions else None,
            "approval_status": self.approval_status,
            "notes": self.notes,
            "priority": self.priority,
            "source": self.source,
            "metadata": self.metadata,
        }

    def validate(self, *, check_file: bool = False) -> list[str]:
        errors: list[str] = []
        if not self.file_path.strip():
            errors.append("reference file path cannot be empty")
        if self.view_type not in {item.value for item in ViewType}:
            errors.append(f"invalid reference view type: {self.view_type}")
        if self.approval_status not in {item.value for item in ApprovalStatus}:
            errors.append(f"invalid approval status: {self.approval_status}")
        if self.priority < 0:
            errors.append("reference priority cannot be negative")
        if self.dimensions and (len(self.dimensions) != 2 or min(self.dimensions) <= 0):
            errors.append("reference dimensions must be two positive integers")
        if self.checksum and (
            len(self.checksum) != 64
            or any(c not in "0123456789abcdef" for c in self.checksum)
        ):
            errors.append("reference checksum must be a SHA-256 hex digest")
        if check_file and "://" not in self.file_path:
            path = Path(self.file_path)
            if not path.is_file():
                errors.append(f"missing reference file: {self.file_path}")
            elif (
                self.checksum
                and hashlib.sha256(path.read_bytes()).hexdigest() != self.checksum
            ):
                errors.append(f"checksum mismatch: {self.file_path}")
        return errors


Reference = ReferenceImage
