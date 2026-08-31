"""Optional real GPU video execution through Hugging Face Diffusers.

This module deliberately keeps pretrained-foundation provenance explicit. A
Diffusers checkpoint can accelerate CINEOS research, but using one never turns
that checkpoint into a CINEOS-native model. CINEOS-owned conditioning,
continuity, QC and orchestration live above this execution boundary.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .base_renderer import BaseRenderer
from .capabilities import Range, RendererCapabilities, Resolution
from .native_request import NativeShotRequest


class DiffusersVideoError(RuntimeError):
    """Raised when the optional Diffusers execution boundary cannot render."""


@dataclass(frozen=True, slots=True)
class FoundationProvenance:
    """Auditable identity for a pretrained foundation used by CINEOS."""

    model_id: str
    revision: str | None = None
    license_id: str | None = None
    source_url: str | None = None
    foundation_name: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "license_id": self.license_id,
            "source_url": self.source_url,
            "foundation_name": self.foundation_name,
        }


@dataclass(frozen=True, slots=True)
class DiffusersVideoResult:
    """One rendered shot plus enough provenance to reproduce the execution."""

    shot_id: str
    scene_id: str
    output_path: str
    frame_count: int
    seed: int
    foundation: FoundationProvenance
    request_hash: str
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None


ReferenceLoader = Callable[[str], Any]
PipelineFactory = Callable[..., Any]
VideoExporter = Callable[..., Any]


class DiffusersVideoRenderer(BaseRenderer):
    """Lazy optional GPU renderer for Diffusers-compatible video checkpoints.

    The class has no hard dependency on torch or diffusers. Production installs
    opt in through the ``video`` extra; normal CINEOS core/tests stay lightweight.
    Tests can inject a pipeline factory/exporter without importing either package.

    ``memory_strategy`` is consumed by CINEOS rather than forwarded to
    ``from_pretrained``. This makes large foundations usable on constrained GPUs
    without hiding the fact that CPU offload is active.

    ``require_artifact_evidence`` is intentionally opt-in for backwards
    compatibility with orchestration layers that own artifact validation and
    error translation. Strict direct execution removes stale output before export
    and refuses to return without a fresh, non-empty SHA-256-bound artifact.
    """

    _MEMORY_STRATEGIES = frozenset(
        {"resident", "model_cpu_offload", "sequential_cpu_offload"}
    )
    _INFERENCE_CONTROL_KEYS = frozenset(
        {
            "negative_prompt",
            "guidance_scale",
            "num_inference_steps",
            "strength",
            "max_sequence_length",
        }
    )

    def __init__(
        self,
        foundation: FoundationProvenance,
        *,
        output_dir: str | Path,
        resolutions: tuple[tuple[int, int], ...] = ((832, 480), (1280, 720)),
        duration_range: tuple[float, float] = (1.0, 10.0),
        fps: tuple[float, ...] = (16.0, 24.0),
        supported_features: frozenset[str] = frozenset(),
        maximum_character_count: int | None = None,
        reference_loader: ReferenceLoader | None = None,
        pipeline_factory: PipelineFactory | None = None,
        video_exporter: VideoExporter | None = None,
        require_artifact_evidence: bool = False,
    ) -> None:
        self.foundation = foundation
        self.output_dir = Path(output_dir)
        self.reference_loader = reference_loader
        self._pipeline_factory = pipeline_factory
        self._video_exporter = video_exporter
        self._require_artifact_evidence = require_artifact_evidence
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._device = "cuda"
        self._dtype_name = "bfloat16"
        self._model_options: dict[str, Any] = {}
        self._memory_strategy = "resident"
        self._enable_vae_tiling = False
        self._enable_vae_slicing = False
        self._enable_attention_slicing = False
        self._capabilities = RendererCapabilities(
            supported_resolution=tuple(Resolution(*item) for item in resolutions),
            supported_duration=Range(*duration_range),
            supported_fps=fps,
            supported_features=supported_features,
            maximum_character_count=maximum_character_count,
        )

    @property
    def capabilities(self) -> RendererCapabilities:
        return self._capabilities

    def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model: str | None = None, **options: Any) -> None:
        model_id = model or self.foundation.model_id
        if model_id != self.foundation.model_id:
            raise DiffusersVideoError(
                "model override must match declared foundation provenance"
            )

        self._device = str(options.pop("device", "cuda"))
        self._dtype_name = str(options.pop("dtype", "bfloat16"))
        self._memory_strategy = str(options.pop("memory_strategy", "resident"))
        self._enable_vae_tiling = bool(options.pop("enable_vae_tiling", False))
        self._enable_vae_slicing = bool(options.pop("enable_vae_slicing", False))
        self._enable_attention_slicing = bool(
            options.pop("enable_attention_slicing", False)
        )
        if self._memory_strategy not in self._MEMORY_STRATEGIES:
            choices = ", ".join(sorted(self._MEMORY_STRATEGIES))
            raise DiffusersVideoError(
                f"unsupported memory_strategy {self._memory_strategy!r}; "
                f"expected one of: {choices}"
            )
        if self._memory_strategy != "resident" and not self._device.startswith("cuda"):
            raise DiffusersVideoError("CPU offload strategies require a CUDA device")
        self._model_options = dict(options)

        if self._pipeline_factory is None:
            try:
                diffusers = import_module("diffusers")
                self._torch = import_module("torch")
            except ImportError as exc:
                raise DiffusersVideoError(
                    "real video execution requires the optional 'video' dependencies"
                ) from exc
            self._pipeline_factory = diffusers.DiffusionPipeline.from_pretrained

        torch_dtype = None
        if self._torch is not None:
            torch_dtype = getattr(self._torch, self._dtype_name, None)
            if torch_dtype is None:
                raise DiffusersVideoError(
                    f"torch does not provide dtype {self._dtype_name!r}"
                )
            if self._device.startswith("cuda") and not self._torch.cuda.is_available():
                raise DiffusersVideoError(
                    "CUDA device requested but torch reports no GPU"
                )

        load_options = dict(self._model_options)
        if torch_dtype is not None:
            load_options["torch_dtype"] = torch_dtype
        if self.foundation.revision is not None:
            load_options.setdefault("revision", self.foundation.revision)

        self._pipeline = self._pipeline_factory(model_id, **load_options)
        self._configure_memory_runtime()

    def warmup(self) -> None:
        if self._pipeline is None:
            raise DiffusersVideoError("model must be loaded before warmup")
        if hasattr(self._pipeline, "set_progress_bar_config"):
            self._pipeline.set_progress_bar_config(disable=True)

    def render(self, request: Any) -> DiffusersVideoResult:
        if not isinstance(request, NativeShotRequest):
            raise TypeError("DiffusersVideoRenderer requires NativeShotRequest")
        if self._pipeline is None:
            raise DiffusersVideoError("renderer model is not loaded")

        call = self._pipeline.__call__
        parameters = inspect.signature(call).parameters
        camera = request.camera
        width, height = tuple(camera.get("resolution", (832, 480)))
        fps = float(camera.get("fps", 24.0))
        duration = float(camera.get("duration", 5.0))
        num_frames = max(1, round(duration * fps))

        kwargs: dict[str, Any] = {
            "prompt": self._compile_prompt(request),
            "width": width,
            "height": height,
            "num_frames": num_frames,
        }
        if "generator" in parameters:
            kwargs["generator"] = self._generator(request.deterministic_seed)

        if "image" in parameters:
            image = self._load_primary_reference(request)
            if image is not None:
                kwargs["image"] = image

        kwargs.update(self._compile_inference_controls(request, parameters))
        filtered = {key: value for key, value in kwargs.items() if key in parameters}
        output = call(**filtered)
        frames = self._extract_frames(output)
        output_path = self.output_dir / f"{request.scene_id}-{request.shot_id}.mp4"
        if self._require_artifact_evidence and output_path.exists():
            output_path.unlink()
        exporter = self._resolve_exporter()
        exporter(frames, str(output_path), fps=fps)

        artifact_size: int | None = None
        artifact_sha256: str | None = None
        if self._require_artifact_evidence:
            artifact_size, artifact_sha256 = self._validate_exported_artifact(
                output_path
            )

        return DiffusersVideoResult(
            shot_id=request.shot_id,
            scene_id=request.scene_id,
            output_path=str(output_path),
            frame_count=len(frames),
            seed=request.deterministic_seed,
            foundation=self.foundation,
            request_hash=request.content_hash,
            artifact_sha256=artifact_sha256,
            artifact_size_bytes=artifact_size,
        )

    def shutdown(self) -> None:
        self._pipeline = None
        if self._torch is not None and self._device.startswith("cuda"):
            empty_cache = getattr(self._torch.cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()

    @staticmethod
    def _validate_exported_artifact(output_path: Path) -> tuple[int, str]:
        """Bind a successful render result to a fresh, non-empty file on disk."""
        if not output_path.is_file():
            raise DiffusersVideoError(
                "video exporter did not create the requested output artifact"
            )
        artifact_size = output_path.stat().st_size
        if artifact_size <= 0:
            raise DiffusersVideoError("video exporter created an empty output artifact")

        digest = hashlib.sha256()
        with output_path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        return artifact_size, digest.hexdigest()

    def _configure_memory_runtime(self) -> None:
        if self._pipeline is None:
            raise DiffusersVideoError("renderer model is not loaded")

        if self._memory_strategy == "resident":
            if hasattr(self._pipeline, "to"):
                self._pipeline.to(self._device)
        elif self._memory_strategy == "model_cpu_offload":
            self._invoke_offload_feature("enable_model_cpu_offload")
        elif self._memory_strategy == "sequential_cpu_offload":
            self._invoke_offload_feature("enable_sequential_cpu_offload")

        if self._enable_vae_tiling:
            self._invoke_pipeline_feature("enable_vae_tiling", required=True)
        if self._enable_vae_slicing:
            self._invoke_pipeline_feature("enable_vae_slicing", required=True)
        if self._enable_attention_slicing:
            self._invoke_pipeline_feature("enable_attention_slicing", required=True)

    def _invoke_offload_feature(self, name: str) -> None:
        """Route Diffusers CPU offload to the GPU selected by CINEOS.

        Diffusers defaults ``gpu_id`` to zero. That default is unsafe after the GPU
        preflight has deliberately selected another CUDA device, because a busy GPU
        zero could receive the model despite a plan targeting ``cuda:1`` or later.
        Older/custom pipelines that cannot accept ``gpu_id`` remain usable on GPU
        zero, but fail closed for non-zero devices instead of silently misrouting.
        """
        gpu_id = self._cuda_device_index()
        self._invoke_pipeline_feature(
            name,
            required=True,
            required_keyword="gpu_id" if gpu_id != 0 else None,
            gpu_id=gpu_id,
        )

    def _cuda_device_index(self) -> int:
        if self._device == "cuda":
            return 0
        prefix, separator, suffix = self._device.partition(":")
        if prefix != "cuda" or not separator:
            raise DiffusersVideoError(
                f"CPU offload requires an explicit CUDA device, got {self._device!r}"
            )
        try:
            index = int(suffix)
        except ValueError as exc:
            raise DiffusersVideoError(
                f"invalid CUDA device {self._device!r} for CPU offload"
            ) from exc
        if index < 0:
            raise DiffusersVideoError(
                f"invalid CUDA device {self._device!r} for CPU offload"
            )
        return index

    def _invoke_pipeline_feature(
        self,
        name: str,
        *,
        required: bool,
        required_keyword: str | None = None,
        **kwargs: Any,
    ) -> None:
        if self._pipeline is None:
            raise DiffusersVideoError("renderer model is not loaded")
        feature = getattr(self._pipeline, name, None)
        if not callable(feature):
            if required:
                raise DiffusersVideoError(
                    f"loaded pipeline does not support requested feature {name!r}"
                )
            return

        parameters = inspect.signature(feature).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        accepted_kwargs = {
            key: value
            for key, value in kwargs.items()
            if accepts_kwargs or key in parameters
        }
        if required_keyword is not None and required_keyword not in accepted_kwargs:
            raise DiffusersVideoError(
                f"loaded pipeline feature {name!r} cannot target {self._device}; "
                f"it does not accept {required_keyword!r}"
            )
        feature(**accepted_kwargs)

    def _generator(self, seed: int) -> Any:
        if self._torch is None:
            return seed
        generator_device = (
            self._device if self._memory_strategy == "resident" else "cpu"
        )
        generator = self._torch.Generator(device=generator_device)
        return generator.manual_seed(seed)

    def _load_primary_reference(self, request: NativeShotRequest) -> Any | None:
        if not request.approved_reference_ids:
            return None
        if self.reference_loader is None:
            return None
        return self.reference_loader(request.approved_reference_ids[0])

    @classmethod
    def _compile_inference_controls(
        cls,
        request: NativeShotRequest,
        parameters: dict[str, inspect.Parameter] | Any,
    ) -> dict[str, Any]:
        """Forward an audited allow-list of quality controls to compatible pipelines.

        Controls may be supplied under ``renderer_requirements['inference']`` or
        ``metadata['inference']``. Metadata wins so benchmark jobs can tune a shot
        without mutating the renderer-independent capability contract. Unsupported
        controls are ignored rather than leaked into arbitrary pipeline kwargs.
        """
        controls: dict[str, Any] = {}
        requirements = request.renderer_requirements.get("inference", {})
        metadata = request.metadata.get("inference", {})
        if isinstance(requirements, dict):
            controls.update(requirements)
        if isinstance(metadata, dict):
            controls.update(metadata)

        direct_negative = request.metadata.get("negative_prompt")
        if direct_negative is not None:
            controls["negative_prompt"] = direct_negative

        result: dict[str, Any] = {}
        for key in cls._INFERENCE_CONTROL_KEYS:
            if key not in controls or key not in parameters:
                continue
            value = controls[key]
            cls._validate_inference_control(key, value)
            result[key] = value
        return result

    @staticmethod
    def _validate_inference_control(key: str, value: Any) -> None:
        if key == "negative_prompt":
            if not isinstance(value, str):
                raise DiffusersVideoError("negative_prompt must be a string")
            return
        if key in {"num_inference_steps", "max_sequence_length"}:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DiffusersVideoError(f"{key} must be a positive integer")
            return
        if key in {"guidance_scale", "strength"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DiffusersVideoError(f"{key} must be numeric")
            numeric = float(value)
            if key == "guidance_scale" and numeric < 0:
                raise DiffusersVideoError("guidance_scale must be non-negative")
            if key == "strength" and not 0 <= numeric <= 1:
                raise DiffusersVideoError("strength must be between 0 and 1")

    def _resolve_exporter(self) -> VideoExporter:
        if self._video_exporter is not None:
            return self._video_exporter
        try:
            utils = import_module("diffusers.utils")
        except ImportError as exc:
            raise DiffusersVideoError(
                "video export requires the optional 'video' dependencies"
            ) from exc
        self._video_exporter = utils.export_to_video
        return self._video_exporter

    @staticmethod
    def _extract_frames(output: Any) -> list[Any]:
        frames = getattr(output, "frames", None)
        if frames is None and isinstance(output, dict):
            frames = output.get("frames")
        if frames is None:
            raise DiffusersVideoError("pipeline output does not expose video frames")
        if isinstance(frames, tuple):
            frames = list(frames)
        if frames and isinstance(frames[0], list):
            frames = frames[0]
        result = list(frames)
        if not result:
            raise DiffusersVideoError("pipeline returned zero video frames")
        return result

    @staticmethod
    def _compile_prompt(request: NativeShotRequest) -> str:
        """Convert structured CINEOS conditioning into deterministic model text."""
        fragments: list[str] = []
        explicit = request.metadata.get("prompt") or request.metadata.get("action")
        if explicit:
            fragments.append(str(explicit))

        for character in request.characters:
            identity = character.get("identity_invariants", [])
            if identity:
                fragments.append("character identity: " + ", ".join(map(str, identity)))

        if request.environment:
            description = request.environment.get(
                "description"
            ) or request.environment.get("name")
            if description:
                fragments.append(f"environment: {description}")

        camera = request.camera
        for key in ("shot_size", "angle", "movement", "lens"):
            value = camera.get(key)
            if value:
                fragments.append(f"camera {key.replace('_', ' ')}: {value}")

        facial = request.performance.get("facial_targets")
        if facial:
            fragments.append(f"facial performance: {facial}")
        gestures = request.performance.get("gesture_tracks")
        if gestures:
            fragments.append(f"gestures: {gestures}")

        if not fragments:
            fragments.append(f"cinematic shot {request.shot_id}")
        return ". ".join(fragments)
