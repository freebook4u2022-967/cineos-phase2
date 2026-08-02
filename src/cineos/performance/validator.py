from dataclasses import dataclass, field

from .timing import validate_timeline


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self):
        return not self.errors


class PerformanceValidator:
    def validate(self, plan, *, renderer_features=None, fallback_policy=None):
        r = ValidationReport()
        r.errors += validate_timeline(plan.performance_beats)
        for track in plan.facial_performance_tracks:
            r.errors += [
                f"unapproved expression: {x}" for x in track.validate_expressions()
            ]
        r.errors += [
            f"contradictory emotional states: {', '.join(x)}"
            for x in plan.emotional_arc.contradictions()
        ]
        if renderer_features is not None:
            missing = plan.renderer_capability_requirements.required_features() - set(
                renderer_features
            )
            approved = set((fallback_policy or {}).get("approved_capabilities", []))
            fatal = missing - approved
            plan.lost_capabilities = sorted(missing & approved)
            r.warnings += [
                f"approved fallback loses capability: {x}"
                for x in plan.lost_capabilities
            ]
            r.errors += [f"unsupported renderer capability: {x}" for x in sorted(fatal)]
        if plan.lip_sync_tracks and not plan.dialogue_cue_ids:
            r.errors.append("lip-sync cannot remove dialogue identity")
        return r
