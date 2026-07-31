"""Named, approved expression descriptions."""

from dataclasses import dataclass, field

STANDARD_EXPRESSIONS = ("neutral", "happy", "sad", "angry", "afraid", "surprised")


@dataclass(slots=True)
class ExpressionProfile:
    name: str
    description: str = ""
    approved_reference_ids: list[str] = field(default_factory=list)
