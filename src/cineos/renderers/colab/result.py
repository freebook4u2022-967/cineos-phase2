from dataclasses import asdict, dataclass, field


@dataclass
class ColabRenderResult:
    project_id: str
    shots: list[dict] = field(default_factory=list)
    final_film: str = ""
    model_id: str = ""
    hardware: str = ""
    render_time_seconds: float = 0

    def to_dict(self):
        return asdict(self)
