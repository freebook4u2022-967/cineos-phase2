from dataclasses import asdict, dataclass

PINNED_COLAB_DEPENDENCIES = {
    "diffusers": "0.30.3",
    "transformers": "4.44.2",
    "accelerate": "0.34.2",
    "sentencepiece": "0.2.0",
    "imageio[ffmpeg]": "2.35.1",
    "safetensors": "0.4.5",
}


@dataclass(slots=True)
class ColabRenderConfig:
    model_id: str = "THUDM/CogVideoX-2b"
    fps: int = 8
    resolution: str = "720x480"
    inference_steps: int = 50
    guidance_scale: float = 6.0
    seed: int = 42
    minimum_file_size: int = 2_048
    black_luminance_threshold: float = 3.0
    black_variance_threshold: float = 2.0
    frozen_frame_delta_threshold: float = 0.35
    minimum_free_vram_gb: float = 2.0
    retry_resolution: str = "480x320"
    retry_frame_count: int = 33
    retry_inference_steps: int | None = None

    def to_dict(self):
        return asdict(self)
