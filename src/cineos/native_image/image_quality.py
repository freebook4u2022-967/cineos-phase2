"""Image quality intelligence for CINEOS training datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImageQualityPolicy:
    minimum_width: int = 256
    minimum_height: int = 256
    minimum_aspect_ratio: float = 0.5
    maximum_aspect_ratio: float = 2.4
    minimum_blur_score: float = 2.0
    minimum_mean_luma: float = 0.05
    maximum_mean_luma: float = 0.95
    near_duplicate_hamming_distance: int = 4


@dataclass(frozen=True, slots=True)
class ImageQualityMetrics:
    width: int
    height: int
    aspect_ratio: float
    mean_luma: float
    blur_score: float
    perceptual_hash: str


@dataclass(frozen=True, slots=True)
class ImageQualityAssessment:
    path: str
    approved: bool
    metrics: ImageQualityMetrics
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetQualityReport:
    assessed: tuple[ImageQualityAssessment, ...]
    approved_count: int
    rejected_count: int
    near_duplicate_pairs: tuple[tuple[str, str], ...]

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "assessed": [asdict(item) for item in self.assessed],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination


def _load_ppm(path: Path):
    payload = path.read_bytes()
    if not payload.startswith(b"P6"):
        raise ValueError("image quality intelligence currently supports P6 PPM")
    header, rgb = payload.split(b"\n255\n", 1)
    parts = header.split()
    width, height = int(parts[-2]), int(parts[-1])
    if len(rgb) != width * height * 3:
        raise ValueError("invalid P6 PPM payload size")
    return width, height, rgb


def _mean_luma(rgb: bytes) -> float:
    total = 0.0
    pixels = len(rgb) // 3
    for index in range(0, len(rgb), 3):
        r, g, b = rgb[index], rgb[index + 1], rgb[index + 2]
        total += (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return total / pixels


def _blur_score(rgb: bytes, width: int, height: int) -> float:
    if width < 3 or height < 3:
        return 0.0
    values = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            offset = (y * width + x) * 3
            center = sum(rgb[offset : offset + 3]) / 3.0
            left = sum(rgb[offset - 3 : offset]) / 3.0
            right = sum(rgb[offset + 3 : offset + 6]) / 3.0
            values.append(abs((2.0 * center) - left - right))
    return sum(values) / len(values) if values else 0.0


def _perceptual_hash(rgb: bytes, width: int, height: int) -> str:
    buckets = []
    for by in range(8):
        for bx in range(8):
            x0, x1 = bx * width // 8, (bx + 1) * width // 8
            y0, y1 = by * height // 8, (by + 1) * height // 8
            values = []
            for y in range(y0, max(y0 + 1, y1)):
                for x in range(x0, max(x0 + 1, x1)):
                    offset = (y * width + x) * 3
                    values.append(sum(rgb[offset : offset + 3]) / 3.0)
            buckets.append(sum(values) / len(values))
    threshold = sum(buckets) / len(buckets)
    bits = "".join("1" if value >= threshold else "0" for value in buckets)
    return f"{int(bits, 2):016x}"


def _hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


class ImageQualityInspector:
    def __init__(self, policy: ImageQualityPolicy | None = None) -> None:
        self.policy = policy or ImageQualityPolicy()

    def assess(self, path: str | Path) -> ImageQualityAssessment:
        image_path = Path(path)
        width, height, rgb = _load_ppm(image_path)
        metrics = ImageQualityMetrics(
            width=width,
            height=height,
            aspect_ratio=width / height,
            mean_luma=_mean_luma(rgb),
            blur_score=_blur_score(rgb, width, height),
            perceptual_hash=_perceptual_hash(rgb, width, height),
        )
        reasons = []
        if width < self.policy.minimum_width or height < self.policy.minimum_height:
            reasons.append("resolution below minimum")
        if (
            not self.policy.minimum_aspect_ratio
            <= metrics.aspect_ratio
            <= self.policy.maximum_aspect_ratio
        ):
            reasons.append("aspect ratio outside policy")
        if metrics.blur_score < self.policy.minimum_blur_score:
            reasons.append("image appears too blurry")
        if metrics.mean_luma < self.policy.minimum_mean_luma:
            reasons.append("image appears underexposed")
        if metrics.mean_luma > self.policy.maximum_mean_luma:
            reasons.append("image appears overexposed")
        return ImageQualityAssessment(
            str(image_path), not reasons, metrics, tuple(reasons)
        )

    def report(self, paths: tuple[str | Path, ...]) -> DatasetQualityReport:
        assessed = tuple(self.assess(path) for path in paths)
        near_duplicates = []
        for index, first in enumerate(assessed):
            for second in assessed[index + 1 :]:
                if (
                    _hamming(
                        first.metrics.perceptual_hash, second.metrics.perceptual_hash
                    )
                    <= self.policy.near_duplicate_hamming_distance
                ):
                    near_duplicates.append((first.path, second.path))
        return DatasetQualityReport(
            assessed=assessed,
            approved_count=sum(item.approved for item in assessed),
            rejected_count=sum(not item.approved for item in assessed),
            near_duplicate_pairs=tuple(near_duplicates),
        )
