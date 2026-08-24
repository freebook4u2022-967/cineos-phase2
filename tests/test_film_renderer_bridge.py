from types import SimpleNamespace

from cineos.film.planner import plan_shots
from cineos.film.renderer_bridge import LocalAIFilmRenderer


class _Package:
    shot_manifest = (
        {
            "shot_id": "shot-1",
            "scene_id": "scene-1",
            "duration": 2.5,
            "prompt": "A figure crosses a rain-soaked platform.",
            "character_ids": ("lead",),
            "continuity_key": "scene-1:lead",
        },
    )
    timeline_manifest = {
        "scene_order": ["scene-1"],
        "shot_order": {"scene-1": ["shot-1"]},
    }


class _LocalRenderer:
    def __init__(self):
        self.request = None
        self.cancelled = False

    def render(self, request):
        self.request = request
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(b"mp4")
        return SimpleNamespace(output_path=str(request.output_path))

    def cancel(self):
        self.cancelled = True


def test_planner_preserves_renderer_payload():
    shot = plan_shots(_Package())[0]
    assert shot.payload["prompt"].startswith("A figure")
    assert shot.payload["continuity_key"] == "scene-1:lead"


def test_local_ai_film_bridge_builds_deterministic_render_request(tmp_path):
    backend = _LocalRenderer()
    bridge = LocalAIFilmRenderer(backend)
    planned = plan_shots(_Package())[0]
    target = tmp_path / "shot.mp4"
    result = bridge.render(planned, target)
    assert result == target
    assert backend.request.shot_id == "shot-1"
    assert backend.request.prompt == "A figure crosses a rain-soaked platform."
    assert backend.request.duration == 2.5
    assert backend.request.metadata["character_ids"] == ("lead",)
    seed = backend.request.seed
    bridge.render(planned, tmp_path / "shot-2.mp4")
    assert backend.request.seed == seed


def test_local_ai_bridge_delegates_cancel():
    backend = _LocalRenderer()
    bridge = LocalAIFilmRenderer(backend)
    bridge.cancel_pending()
    assert backend.cancelled
