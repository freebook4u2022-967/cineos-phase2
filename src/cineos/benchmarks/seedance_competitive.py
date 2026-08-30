"""Production GPU benchmark catalog for Seedance-class competitive evaluation.

The catalog deliberately tests film-level failure modes rather than isolated pretty
clips.  It is foundation-agnostic: an external pretrained checkpoint may execute a
shot, while CINEOS-owned conditioning, continuity, QC/retry, audio, and assembly are
what this suite is intended to measure.
"""

from __future__ import annotations

import uuid

from .case import BenchmarkCase
from .suite import BenchmarkSuite

COMPETITIVE_CASES = (
    (
        "identity-closeup",
        "Persistent identity under close-up motion",
        "Measure face/wardrobe identity while expression, head pose, and camera distance change.",
        ("identity_lock", "reference_image"),
        {"identity_score": 0.90, "temporal_stability": 0.86},
    ),
    (
        "two-character-dialogue",
        "Two-character dialogue and turn taking",
        "Measure distinct identities, eyelines, lip timing, and shot/reverse-shot continuity.",
        ("multi_character", "dialogue", "identity_lock"),
        {
            "identity_score": 0.88,
            "lip_sync_timing_accuracy": 0.80,
            "audio_alignment": 0.85,
        },
    ),
    (
        "hands-object",
        "Hands and object interaction",
        "Stress anatomy, grasp persistence, prop state, and contact during a handoff.",
        ("multi_character", "prop_continuity"),
        {"prop_continuity_score": 0.86, "temporal_stability": 0.84},
    ),
    (
        "walk-run",
        "Walking and running body dynamics",
        "Stress full-body anatomy, gait, foot contact, identity, and motion coherence.",
        ("full_body_motion", "identity_lock"),
        {"identity_score": 0.86, "temporal_stability": 0.84},
    ),
    (
        "fast-camera",
        "Fast camera movement",
        "Stress subject persistence through whip-pan, orbit, and rapid dolly movement.",
        ("camera_motion", "identity_lock"),
        {"identity_score": 0.84, "temporal_stability": 0.82},
    ),
    (
        "lighting-transition",
        "Lighting and exposure transition",
        "Measure identity/environment continuity while lighting changes substantially.",
        ("lighting_control", "identity_lock"),
        {"identity_score": 0.88, "environment_continuity_score": 0.84},
    ),
    (
        "physics-weather",
        "Weather and lightweight physics",
        "Stress rain, cloth/hair response, splashes, and environment persistence.",
        ("weather", "physics"),
        {"environment_continuity_score": 0.84, "temporal_stability": 0.82},
    ),
    (
        "scene-boundary",
        "Scene-boundary continuity",
        "Measure wardrobe, props, identity, and story state across a location transition.",
        ("scene_memory", "identity_lock", "prop_continuity"),
        {
            "identity_score": 0.88,
            "wardrobe_continuity_score": 0.90,
            "prop_continuity_score": 0.88,
        },
    ),
    (
        "qc-rerender",
        "Automated QC reject and rerender",
        "Force one below-threshold take and verify automatic rejection, retry, and recovery evidence.",
        ("automatic_qc", "rerender"),
        {"validation_pass_rate": 1.0, "render_completion_rate": 1.0},
    ),
    (
        "connected-film",
        "Connected 5-10 shot film",
        "Validate a complete connected sequence with identity, continuity, dialogue, QC, audio, and assembly.",
        ("identity_lock", "scene_memory", "automatic_qc", "audio", "film_assembly"),
        {
            "execution_success": 1.0,
            "render_completion_rate": 1.0,
            "identity_score": 0.88,
            "temporal_stability": 0.84,
            "audio_alignment": 0.85,
            "final_assembly_success": 1.0,
        },
    ),
)


def seedance_competitive_suite() -> BenchmarkSuite:
    """Return the production-GPU competitive suite used for the one-month gate."""
    cases = tuple(
        BenchmarkCase(
            case_id=f"competitive-{case_id}",
            title=title,
            purpose=purpose,
            project_fixture=f"benchmarks/projects/competitive-{case_id}.json",
            renderer_requirements=requirements,
            expected_outputs=("report.json", "render_receipt.json", "output.mp4"),
            maximum_runtime=1800.0 if case_id == "connected-film" else 600.0,
            hardware_requirements={"gpu": True, "real_inference": True},
            deterministic_seed=20260929 + index,
            validation_thresholds=thresholds,
            mandatory=True,
            slow=True,
        )
        for index, (case_id, title, purpose, requirements, thresholds) in enumerate(
            COMPETITIVE_CASES, 1
        )
    )
    return BenchmarkSuite(
        suite_id=str(uuid.UUID("04a5ad04-dc31-5b2b-8ebf-78fa5d26829f")),
        suite_version="1.0.0",
        cases=cases,
        target_platform="gpu",
        renderer_profile="external-pretrained-foundation+cineos-control",
        thresholds={"failure_rate": 0.0, "manual_review_count": 0.0},
        metadata={
            "release_gate": "seedance-class-competitive",
            "real_inference": True,
            "minimum_connected_shots": 5,
            "maximum_connected_shots": 10,
            "foundation_origin_required": "external_pretrained_foundation",
            "cineos_owned_layers": (
                "conditioning",
                "identity",
                "continuity",
                "automatic_qc",
                "rerender",
                "audio",
                "assembly",
            ),
        },
    )
