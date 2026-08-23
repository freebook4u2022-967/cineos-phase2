"""Character reference coverage intelligence for CINEOS datasets."""

from __future__ import annotations

from dataclasses import dataclass

from .training import NativeDatasetManifest


REQUIRED_IDENTITY_VIEWS = frozenset({"front", "side", "three_quarter", "full_body"})
REQUIRED_VARIATIONS = frozenset({"expression", "lighting", "costume"})


@dataclass(frozen=True, slots=True)
class CharacterIdentityCoverage:
    character_id: str
    sample_count: int
    views: tuple[str, ...]
    variations: tuple[str, ...]
    missing_views: tuple[str, ...]
    missing_variations: tuple[str, ...]
    coverage_score: float


@dataclass(frozen=True, slots=True)
class IdentityCoverageReport:
    characters: tuple[CharacterIdentityCoverage, ...]

    @property
    def mean_coverage_score(self) -> float:
        if not self.characters:
            return 0.0
        return sum(item.coverage_score for item in self.characters) / len(self.characters)


class IdentityCoverageAnalyzer:
    """Measure whether each identity has enough reference diversity for continuity training.

    Metadata convention:
      identity_tags: character identifiers, e.g. ("arif",)
      metadata["identity_views"]: list/tuple of front, side, three_quarter, full_body
      metadata["identity_variations"]: list/tuple of expression, lighting, costume
    """

    def analyze(self, manifest: NativeDatasetManifest) -> IdentityCoverageReport:
        buckets: dict[str, dict[str, object]] = {}
        for sample in manifest.samples:
            for character_id in sample.identity_tags:
                bucket = buckets.setdefault(
                    character_id,
                    {"samples": 0, "views": set(), "variations": set()},
                )
                bucket["samples"] = int(bucket["samples"]) + 1
                bucket["views"].update(sample.metadata.get("identity_views", ()))
                bucket["variations"].update(sample.metadata.get("identity_variations", ()))

        results = []
        for character_id, bucket in sorted(buckets.items()):
            views = set(bucket["views"])
            variations = set(bucket["variations"])
            missing_views = REQUIRED_IDENTITY_VIEWS - views
            missing_variations = REQUIRED_VARIATIONS - variations
            view_score = len(REQUIRED_IDENTITY_VIEWS & views) / len(REQUIRED_IDENTITY_VIEWS)
            variation_score = len(REQUIRED_VARIATIONS & variations) / len(REQUIRED_VARIATIONS)
            reference_score = min(1.0, int(bucket["samples"]) / 4.0)
            score = 0.50 * view_score + 0.30 * variation_score + 0.20 * reference_score
            results.append(
                CharacterIdentityCoverage(
                    character_id=character_id,
                    sample_count=int(bucket["samples"]),
                    views=tuple(sorted(views)),
                    variations=tuple(sorted(variations)),
                    missing_views=tuple(sorted(missing_views)),
                    missing_variations=tuple(sorted(missing_variations)),
                    coverage_score=score,
                )
            )
        return IdentityCoverageReport(tuple(results))
