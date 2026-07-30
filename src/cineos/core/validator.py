"""Cross-model validation for CINEOS projects."""

from collections import Counter

from .project import MovieProject


class ProjectValidationError(ValueError):
    """Raised when a movie project fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class ProjectValidator:
    """Validate identities, references, duration, and ordering consistency."""

    def validate(self, project: MovieProject) -> list[str]:
        errors: list[str] = []
        scene_ids = [scene.scene_id for scene in project.scenes]
        shot_ids = [shot.shot_id for scene in project.scenes for shot in scene.shots]
        assets = [*project.characters, *project.locations, *project.props]
        asset_ids = [asset.asset_id for asset in assets]

        errors.extend(self._invalid_and_duplicate_ids("scene", scene_ids))
        errors.extend(self._invalid_and_duplicate_ids("shot", shot_ids))
        errors.extend(self._invalid_and_duplicate_ids("asset", asset_ids))

        character_ids = {asset.asset_id for asset in project.characters}
        location_ids = {asset.asset_id for asset in project.locations}
        all_asset_ids = set(asset_ids)
        for scene in project.scenes:
            if scene.location is not None and scene.location not in location_ids:
                errors.append(
                    f"scene {scene.scene_id!r} references unknown location "
                    f"{scene.location!r}"
                )
            for character_id in scene.characters:
                if character_id not in character_ids:
                    errors.append(
                        f"scene {scene.scene_id!r} references unknown character "
                        f"{character_id!r}"
                    )
            for shot in scene.shots:
                for reference in shot.references:
                    if reference not in all_asset_ids:
                        errors.append(
                            f"shot {shot.shot_id!r} references unknown asset "
                            f"{reference!r}"
                        )

        errors.extend(project.timeline.validate_durations(project.scenes))
        errors.extend(self._validate_timeline(project))
        return errors

    def is_valid(self, project: MovieProject) -> bool:
        return not self.validate(project)

    def raise_for_errors(self, project: MovieProject) -> None:
        if errors := self.validate(project):
            raise ProjectValidationError(errors)

    @staticmethod
    def _invalid_and_duplicate_ids(kind: str, identifiers: list[str]) -> list[str]:
        errors = [f"{kind} ID cannot be empty" for value in identifiers if not value]
        duplicates = sorted(
            value
            for value, count in Counter(identifiers).items()
            if value and count > 1
        )
        errors.extend(f"duplicate {kind} ID: {value}" for value in duplicates)
        return errors

    @staticmethod
    def _validate_timeline(project: MovieProject) -> list[str]:
        errors: list[str] = []
        expected_scenes = [scene.scene_id for scene in project.scenes]
        if project.timeline.scene_order != expected_scenes:
            errors.append("timeline scene order does not match project scenes")
        for scene in project.scenes:
            expected_shots = [shot.shot_id for shot in scene.shots]
            if project.timeline.shot_order.get(scene.scene_id, []) != expected_shots:
                errors.append(
                    f"timeline shot order does not match scene {scene.scene_id!r}"
                )
        unknown = set(project.timeline.shot_order) - set(expected_scenes)
        for scene_id in sorted(unknown):
            errors.append(f"timeline contains unknown scene {scene_id!r}")
        return errors
