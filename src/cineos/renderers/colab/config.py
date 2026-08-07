from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ColabRenderConfig:
    model_id: str = "THUDM/CogVideoX-2b"
    fps: int = 8
    resolution: str = "720x480"
    inference_steps: int = 50
    guidance_scale: float = 6.0
    seed: int = 42

    def to_dict(self):
        return asdict(self)
