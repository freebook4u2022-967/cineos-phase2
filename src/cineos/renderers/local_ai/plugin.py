"""Plugin boundary and Atlas Runtime integration."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from cineos.atlas import AtlasRuntime

from .adapter import LocalAIRenderer
from .config import LocalAIConfig
from .request import RenderRequest, build_prompt
from .validator import validate_request


class LocalAIRendererPlugin:
    plugin_id = "local-ai"
    version = "1.0.0"

    def __init__(self, config: LocalAIConfig | None = None, **renderer_options) -> None:
        self.renderer = LocalAIRenderer(config, **renderer_options)

    def create_request(
        self,
        package,
        conditioning,
        output: str | Path,
        *,
        shot_id: str,
        job_id: str | None = None,
    ) -> RenderRequest:
        try:
            shot = next(
                item for item in package.shot_manifest if item["shot_id"] == shot_id
            )
        except StopIteration as error:
            raise ValueError(f"unknown shot ID: {shot_id}") from error
        config = self.renderer.config
        request = RenderRequest(
            job_id or uuid4().hex,
            shot_id,
            build_prompt(shot),
            (
                conditioning.deterministic_seed
                if conditioning.deterministic_seed is not None
                else config.seed
            ),
            Path(output),
            config.width,
            config.height,
            config.fps,
            config.duration,
            config.inference_steps,
            config.guidance,
            tuple(conditioning.approved_reference_ids),
            tuple(package.cinedna_ids),
            {"scene_id": conditioning.scene_id},
        )
        asset_ids = {str(asset.get("asset_id")) for asset in package.asset_manifest}
        validate_request(request, conditioning, self.renderer.capabilities, asset_ids)
        return request

    def render(
        self,
        package,
        conditioning,
        output: str | Path,
        *,
        shot_id: str,
        dry_run: bool = False,
    ):
        runtime = AtlasRuntime()
        job = runtime.prepare(package)
        task = next((task for task in job.tasks if task.shot_id == shot_id), None)
        if task is None:
            raise ValueError(f"unknown shot ID: {shot_id}")
        request = self.create_request(
            package, conditioning, output, shot_id=shot_id, job_id=job.job_id
        )
        if dry_run:
            return request
        self.renderer.initialize()
        try:
            self.renderer.load_model()
            self.renderer.warmup()
            return runtime.execute(
                package,
                lambda current: (
                    self.renderer.render_shot(request)
                    if current.shot_id == shot_id
                    else None
                ),
                job_id=job.job_id,
            ).results[shot_id]
        finally:
            self.renderer.shutdown()
