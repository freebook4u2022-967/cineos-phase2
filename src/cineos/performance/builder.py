from uuid import NAMESPACE_URL, uuid5

from .beat import PerformanceBeat
from .body import BodyPerformanceTrack
from .emotion import EmotionalArc, EmotionalState
from .facial import FacialKeyframe, FacialPerformanceTrack
from .lipsync import LipSyncTrack
from .plan import PerformanceCapabilityRequirements, PerformancePlan
from .serializer import calculate_content_hash
from .validator import PerformanceValidator


class PerformanceBuilder:
    """Build renderer-independent, deterministic direction from approved inputs."""

    def build(
        self,
        shot_plan,
        dialogue_cues=(),
        lip_sync_metadata=(),
        character_dna=(),
        conditioning_package=None,
        *,
        existing=None,
        renderer_features=None,
        fallback_policy=None,
    ):
        cues = [c for c in dialogue_cues if c.shot_id == shot_plan.shot_id]
        dna = {str(d.character_id): d for d in character_dna}
        chars = sorted(
            set(shot_plan.character_blocking) | {str(c.character_id) for c in cues}
        )
        cue_seed = ",".join(sorted(str(c.cue_id) for c in cues))
        seed = f"{shot_plan.scene_id}:{shot_plan.shot_id}:{','.join(chars)}:{cue_seed}"
        pid = str(uuid5(NAMESPACE_URL, seed))
        old_beats = {
            b.beat_id: b
            for b in (existing.performance_beats if existing else [])
            if b.locked or b.manual
        }
        beats = []
        for cue in cues:
            bid = str(uuid5(NAMESPACE_URL, f"{pid}:beat:{cue.cue_id}"))
            beats.append(
                old_beats.get(
                    bid,
                    PerformanceBeat(
                        str(cue.character_id),
                        cue.start_time,
                        cue.end_time,
                        shot_plan.shot_purpose,
                        cue.delivery_intent or cue.emotional_state,
                        shot_plan.action,
                        reaction=shot_plan.performance_direction.reaction_beat,
                        subtext=shot_plan.performance_direction.subtext,
                        restraint=shot_plan.performance_direction.restraint_level,
                        pacing=shot_plan.performance_direction.tempo,
                        beat_id=bid,
                    ),
                )
            )
        facial = []
        arc = []
        for cue in cues:
            cid = str(cue.character_id)
            desired = (cue.emotional_state or "neutral").lower()
            profile = dna.get(cid)
            approved = set(profile.expression_profiles) if profile else set()
            standard_map = {
                "happy": "smile",
                "sad": "sadness",
                "angry": "anger",
                "afraid": "fear",
            }
            desired = standard_map.get(desired, desired)
            custom = list(approved)
            facial.append(
                FacialPerformanceTrack(
                    cid,
                    [FacialKeyframe(cue.start_time, desired)],
                    approved_custom_expressions=custom,
                )
            )
            arc.append(EmotionalState(cue.start_time, desired, source="nova"))
        metas = {
            (str(x.character_id), str(x.dialogue_cue_id)): x for x in lip_sync_metadata
        }
        lips = []
        for cue in cues:
            m = metas.get((str(cue.character_id), str(cue.cue_id)))
            lips.append(
                LipSyncTrack(
                    str(cue.character_id),
                    str(cue.cue_id),
                    list(m.phoneme_timeline) if m else [],
                    list(m.viseme_timeline) if m else [],
                    list(m.word_timeline) if m else [],
                    confidence=m.timing_confidence if m else 0.0,
                    fallback_mode=m.fallback_mode if m else "none",
                    source_audio_hash=getattr(m, "source_audio_hash", "") if m else "",
                )
            )
        plan = PerformancePlan(
            shot_plan.shot_id,
            shot_plan.scene_id,
            chars,
            [str(c.cue_id) for c in cues],
            beats,
            facial,
            lips,
            body_performance_tracks=[
                BodyPerformanceTrack(
                    c,
                    spatial_blocking=[
                        {"description": shot_plan.character_blocking.get(c, "")}
                    ],
                )
                for c in chars
            ],
            emotional_arc=EmotionalArc(arc),
            continuity_inputs=dict(shot_plan.continuity_constraints),
            renderer_capability_requirements=PerformanceCapabilityRequirements(
                facial_control=bool(facial),
                viseme_control=bool(lips),
                pose_control=bool(chars),
                multi_character_performance=len(chars) > 1,
            ),
            performance_id=pid,
            metadata={
                "conditioning_package_id": getattr(
                    conditioning_package, "content_hash", ""
                )
            },
        )
        if existing:
            for field in (
                "facial_performance_tracks",
                "lip_sync_tracks",
                "gesture_tracks",
                "body_performance_tracks",
                "eye_line_tracks",
            ):
                locked = [
                    x for x in getattr(existing, field) if getattr(x, "locked", False)
                ]
                if locked:
                    setattr(
                        plan,
                        field,
                        locked
                        + [
                            x
                            for x in getattr(plan, field)
                            if x.character_id not in {y.character_id for y in locked}
                        ],
                    )
        report = PerformanceValidator().validate(
            plan, renderer_features=renderer_features, fallback_policy=fallback_policy
        )
        if report.errors:
            raise ValueError("; ".join(report.errors))
        plan.content_hash = calculate_content_hash(plan)
        return plan
