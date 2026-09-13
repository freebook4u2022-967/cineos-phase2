"""Private Diffusers backend wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class DiffusersBackend:
    def __init__(self) -> None:
        self.pipeline: Any = None
        self.torch: Any = None

    def load(
        self,
        model_path: str,
        *,
        device: str,
        precision: str,
        attention_slicing: bool,
        vae_slicing: bool,
        cpu_offload: bool,
        allow_remote_model: bool = False,
        model_revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from diffusers import DiffusionPipeline

        dtype = getattr(torch, precision)
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=not allow_remote_model,
            revision=model_revision,
            trust_remote_code=trust_remote_code,
        )
        self.torch = torch
        if attention_slicing and hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        if vae_slicing and hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
        if cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(device)

    def warmup(self) -> None:
        return None

    def generate(self, request: Any, progress: Callable[[int, int], None]) -> list[Any]:
        generator = self.torch.Generator(device="cpu").manual_seed(request.seed)

        def callback(_pipe: Any, step: int, _timestep: Any, kwargs: dict[str, Any]):
            progress(step + 1, request.inference_steps)
            return kwargs

        result = self.pipeline(
            prompt=request.prompt,
            num_frames=request.frame_count,
            width=request.width,
            height=request.height,
            num_inference_steps=request.inference_steps,
            guidance_scale=request.guidance,
            generator=generator,
            callback_on_step_end=callback,
        )
        return result.frames[0]

    def encode(self, frames: list[Any], output: str, fps: int) -> None:
        from diffusers.utils import export_to_video

        export_to_video(frames, output, fps=fps)

    def peak_vram(self) -> int | None:
        if self.torch is not None and self.torch.cuda.is_available():
            return int(self.torch.cuda.max_memory_allocated())
        return None

    def unload(self) -> None:
        self.pipeline = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
