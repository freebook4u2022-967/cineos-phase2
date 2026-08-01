"""Build deterministic conditioning from compiled shots and canonical registries."""

from __future__ import annotations

import hashlib

from cineos.cinedna.serializer import profile_to_dict

from .camera import CameraConditioning
from .character import CharacterConditioning
from .continuity import ContinuityConditioning
from .environment import EnvironmentConditioning
from .exceptions import ConditioningBuildError
from .package import ConditioningPackage, RendererCapabilityRequirements
from .props import PropConditioning
from .serializer import calculate_content_hash
from .validator import ConditioningValidator
from .wardrobe import WardrobeConditioning


def _approved(asset) -> list[str]:
    return sorted(
        str(ref.reference_id)
        for ref in asset.references
        if ref.approval_status == "approved"
    )


def _metadata(asset, key: str, default=None):
    return asset.metadata[key] if key in asset.metadata else default


class ConditioningBuilder:
    """Strict compiler: source registries remain the sole authority for values."""

    def __init__(self, film_package, asset_registry, cinedna_registry) -> None:
        self.film_package = film_package
        self.assets = asset_registry
        self.cinedna = cinedna_registry

    def build(self, shot_id: str) -> ConditioningPackage:
        try:
            shot = next(
                x for x in self.film_package.shot_manifest if x["shot_id"] == shot_id
            )
        except StopIteration as error:
            raise ConditioningBuildError(f"unknown shot: {shot_id}") from error
        scene_id = str(shot["scene_id"])
        scene = next(
            (x for x in self.film_package.scene_manifest if x["scene_id"] == scene_id),
            None,
        )
        if scene is None:
            raise ConditioningBuildError(f"shot {shot_id} has no scene")

        characters: list[CharacterConditioning] = []
        wardrobes: list[WardrobeConditioning] = []
        props: list[PropConditioning] = []
        all_refs: set[str] = set()
        for character_id in sorted(str(x) for x in scene.get("characters", [])):
            asset = self._asset(character_id, "character")
            refs = _approved(asset)
            profile = self._profile(character_id)
            profile_refs = sorted(str(x) for x in profile.approved_reference_ids)
            refs = sorted(set(refs) & set(profile_refs)) if profile_refs else refs
            if not refs:
                raise ConditioningBuildError(
                    f"character {character_id} has no approved references"
                )
            dna = profile_to_dict(profile)
            constraints = dna.get("continuity_constraints", {})
            characters.append(
                CharacterConditioning(
                    character_uuid=character_id,
                    cinedna_profile_id=str(profile.character_uuid),
                    cinedna_profile_version=profile.profile_version,
                    approved_reference_ids=refs,
                    identity_invariants=list(dna["face_profile"].get("invariants", [])),
                    face_constraints=dna["face_profile"],
                    body_constraints=dna["body_profile"],
                    expression_target=shot.get("expression_target"),
                    motion_target=shot.get("motion_target"),
                    scene_specific_overrides=constraints.get(
                        "scene_specific_overrides", {}
                    ).get(scene_id, {}),
                )
            )
            all_refs.update(refs)
            for related in self.assets.resolve_relationships(asset):
                if related.kind == "wardrobe":
                    r = self._require_refs(related)
                    wardrobes.append(
                        WardrobeConditioning(
                            str(related.asset_id),
                            _metadata(related, "garment_components", []),
                            _metadata(related, "continuity_locks", {}),
                            _metadata(related, "allowed_variations", []),
                            _metadata(related, "scene_applicability", []),
                            r,
                        )
                    )
                    all_refs.update(r)
                elif related.kind in {"prop", "vehicle"}:
                    r = self._require_refs(related)
                    props.append(
                        PropConditioning(
                            str(related.asset_id),
                            character_id,
                            _metadata(related, "spatial_role", ""),
                            _metadata(related, "state", {}),
                            _metadata(related, "continuity_locks", {}),
                            r,
                        )
                    )
                    all_refs.update(r)

        environment = None
        location_id = scene.get("location")
        if location_id:
            asset = self._asset(
                str(location_id), "environment", alternate_kind="location"
            )
            refs = self._require_refs(asset)
            all_refs.update(refs)
            environment = EnvironmentConditioning(
                str(asset.asset_id),
                refs,
                asset.description,
                _metadata(asset, "time_of_day"),
                _metadata(asset, "weather"),
                shot.get("lighting"),
                _metadata(asset, "atmosphere"),
                _metadata(asset, "spatial_continuity_constraints", {}),
            )
        for reference in shot.get("references", []):
            if not self._is_approved_reference(str(reference)):
                raise ConditioningBuildError(
                    f"shot reference is not approved: {reference}"
                )
            all_refs.add(str(reference))
        if not all_refs:
            raise ConditioningBuildError(f"shot {shot_id} has no approved references")

        order = [
            x
            for ids in self.film_package.timeline_manifest.get(
                "shot_order", {}
            ).values()
            for x in ids
        ]
        index = order.index(shot_id) if shot_id in order else -1
        continuity_data = dict(shot.get("continuity", {}))
        continuity = ContinuityConditioning(
            previous_shot_id=order[index - 1] if index > 0 else None,
            next_shot_id=(
                order[index + 1] if index >= 0 and index + 1 < len(order) else None
            ),
            **continuity_data,
        )
        self._reject_contradictions(continuity, wardrobes, props)
        metadata = self.film_package.project_metadata
        camera = CameraConditioning(
            shot_type=shot.get("shot_type", shot.get("camera", "")),
            framing=shot.get("framing", ""),
            lens=shot.get("lens", ""),
            aperture=shot.get("aperture"),
            camera_position=shot.get("camera_position"),
            camera_movement=shot.get("movement", ""),
            focus_target=shot.get("focus_target"),
            depth_of_field_intent=shot.get("depth_of_field_intent"),
            aspect_ratio=metadata["aspect_ratio"],
            resolution=tuple(metadata["resolution"]),
            fps=float(metadata["fps"]),
            duration=float(shot["duration"]),
        )
        requirements = RendererCapabilityRequirements(
            bool(all_refs),
            len(all_refs) > 1,
            bool(characters),
            len(characters),
            camera.duration,
            camera.resolution,
            camera.fps,
            bool(shot.get("control_images")),
            bool(shot.get("motion_reference")),
        )
        seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"{self.film_package.content_hashes.get('package', '')}:"
                    f"{shot_id}"
                ).encode()
            ).digest()[:8],
            "big",
        )
        package = ConditioningPackage(
            shot_id,
            scene_id,
            characters,
            environment,
            sorted(wardrobes, key=lambda x: x.wardrobe_asset_id),
            sorted(props, key=lambda x: x.asset_id),
            camera,
            continuity,
            sorted(all_refs),
            requirements,
            seed,
            metadata={
                "film_package_hash": self.film_package.content_hashes.get("package", "")
            },
        )
        package.content_hash = calculate_content_hash(package)
        ConditioningValidator().raise_for_errors(package)
        return package

    build_shot = build

    def _asset(self, identity: str, kind: str, alternate_kind: str | None = None):
        try:
            asset = self.assets.retrieve(identity)
        except Exception:
            asset = next((x for x in self.assets.list() if x.name == identity), None)
        if asset is None or asset.kind not in {kind, alternate_kind}:
            raise ConditioningBuildError(f"missing {kind} asset: {identity}")
        return asset

    def _profile(self, identity: str):
        try:
            return self.cinedna.retrieve(identity)
        except Exception as error:
            raise ConditioningBuildError(
                f"missing CineDNA profile: {identity}"
            ) from error

    def _require_refs(self, asset):
        refs = _approved(asset)
        if not refs:
            raise ConditioningBuildError(
                f"{asset.kind} {asset.asset_id} has no approved references"
            )
        return refs

    def _is_approved_reference(self, identity: str) -> bool:
        return any(identity in _approved(asset) for asset in self.assets.list())

    @staticmethod
    def _reject_contradictions(continuity, wardrobes, props) -> None:
        forbidden = set(continuity.forbidden_changes)
        for item in [*wardrobes, *props]:
            locks = set(item.continuity_locks)
            variations = set(getattr(item, "allowed_variations", []))
            if locks & variations or forbidden & variations:
                raise ConditioningBuildError("contradictory continuity constraints")
