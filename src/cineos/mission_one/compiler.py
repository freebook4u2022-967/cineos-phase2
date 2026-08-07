from __future__ import annotations

from cineos.renderers.colab.config import ColabRenderConfig
from cineos.renderers.colab.package import ColabRenderPackage

from .brief import DirectedSceneBrief
from .continuity import propagate_continuity
from .performance import compile_performance
from .prompt_builder import build_prompt
from .shot_package import DirectedShotPackage
from .validator import validate_brief


def compile_scene(
    brief: DirectedSceneBrief, config: ColabRenderConfig | None = None
) -> ColabRenderPackage:
    errors = validate_brief(brief)
    if errors:
        raise ValueError("; ".join(errors))
    config = config or ColabRenderConfig()
    propagate_continuity(brief)
    shots = []
    for shot in brief.shots:
        perf = compile_performance(brief, shot)
        prompt, negative = build_prompt(perf)
        seed = shot.deterministic_seed or config.seed
        shots.append(
            DirectedShotPackage(
                shot.shot_id,
                shot.duration,
                prompt,
                negative,
                seed,
                round(shot.duration * config.fps),
                perf.to_dict(),
                perf.sections["DIALOGUE"],
                f"{shot.shot_id}.mp4",
            )
        )
    return ColabRenderPackage(
        project_id=str(brief.metadata.get("project_id", "mission-one")),
        scene_id=brief.scene_id,
        shots=shots,
        config=config,
    )
