"""Atlas adapter for one real, local Diffusers video pipeline."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from cineos.atlas import BaseRenderer, Range, RendererCapabilities, Resolution

from .config import LocalAIConfig
from .environment import EnvironmentReport, validate_environment
from .errors import EnvironmentValidationError, RenderCancelled
from .model import DiffusersBackend
from .progress import EventSink, RendererEvent, null_sink
from .request import RenderRequest
from .result import RenderResult


class LocalAIRenderer(BaseRenderer):
    renderer_id = "local-ai"
    renderer_version = "1.0.0"
    model_identifier = "damo-vilab/text-to-video-ms-1.7b"

    def __init__(
        self,
        config: LocalAIConfig | None = None,
        *,
        backend: Any = None,
        event_sink: EventSink = null_sink,
    ) -> None:
        self.config = config or LocalAIConfig()
        self.backend = backend or DiffusersBackend()
        self.event_sink = event_sink
        self._cancelled = False
        self._loaded = False
        self._loaded_model_identifier = self.config.model_path

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            (Resolution(576, 320),), Range(0.125, 5.0), (8.0,), frozenset(), 0
        )

    def _emit(self, name: str, **payload: Any) -> None:
        self.event_sink(RendererEvent(name, payload))

    def initialize(self) -> None:
        self._emit("renderer.initializing", renderer_id=self.renderer_id)
        self._cancelled = False

    def validate_environment(self) -> EnvironmentReport:
        return validate_environment(self.config)

    def load_model(self, model: str | None = None, **options: Any) -> None:
        requested_model = model or self.config.model_path
        self._emit(
            "renderer.model_loading",
            model=requested_model,
            source="remote" if self.config.allow_remote_model else "local",
            revision=self.config.model_revision,
            provenance=self.config.model_provenance,
            license=self.config.model_license,
        )
        report = self.validate_environment()
        if not report.valid:
            raise EnvironmentValidationError("; ".join(report.errors))
        self.backend.load(
            requested_model,
            device=self.config.device,
            precision=self.config.precision,
            attention_slicing=self.config.enable_attention_slicing,
            vae_slicing=self.config.enable_vae_slicing,
            cpu_offload=self.config.cpu_offload,
            allow_remote_model=self.config.allow_remote_model,
            model_revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._loaded_model_identifier = requested_model
        self._loaded = True

    def warmup(self) -> None:
        self._emit("renderer.warming")
        self.backend.warmup()

    def render(self, request: RenderRequest) -> RenderResult:
        return self.render_shot(request)

    def render_shot(self, request: RenderRequest) -> RenderResult:
        if not self._loaded:
            raise RuntimeError("model is not loaded")
        started = time.monotonic()
        self._emit(
            "renderer.render_started", job_id=request.job_id, shot_id=request.shot_id
        )
        try:

            def progress(current: int, total: int) -> None:
                if self._cancelled:
                    raise RenderCancelled("render cancelled")
                self._emit(
                    "renderer.progress",
                    current=current,
                    total=total,
                    fraction=current / total,
                )

            frames = self.backend.generate(request, progress)
            if self._cancelled:
                raise RenderCancelled("render cancelled")
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._emit("renderer.encoding", output=str(request.output_path))
            self.backend.encode(frames, str(request.output_path), request.fps)
            digest = hashlib.sha256(request.output_path.read_bytes()).hexdigest()
            result = RenderResult(
                request.job_id,
                request.shot_id,
                self.renderer_id,
                self.renderer_version,
                self._loaded_model_identifier,
                request.seed,
                str(request.output_path),
                request.duration,
                (request.width, request.height),
                request.fps,
                time.monotonic() - started,
                self.backend.peak_vram(),
                (),
                digest,
                {
                    "device": self.config.device,
                    "precision": self.config.precision,
                    "inference_steps": request.inference_steps,
                    "guidance": request.guidance,
                    "model_source": (
                        "remote" if self.config.allow_remote_model else "local"
                    ),
                    "model_revision": self.config.model_revision,
                    "model_license": self.config.model_license,
                    "model_provenance": self.config.model_provenance,
                },
            )
            self._emit("renderer.completed", result=result.to_dict())
            return result
        except RenderCancelled:
            self._emit("renderer.cancelled", shot_id=request.shot_id)
            raise
        except Exception as error:
            self._emit("renderer.failed", shot_id=request.shot_id, error=str(error))
            raise

    def cancel(self) -> None:
        self._cancelled = True

    def unload_model(self) -> None:
        self.backend.unload()
        self._loaded = False

    def shutdown(self) -> None:
        self.unload_model()
