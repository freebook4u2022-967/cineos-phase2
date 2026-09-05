"""Critical-path renderer bridges for complete film execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from cineos.renderers.local_ai.request import RenderRequest, build_prompt


class LocalAIFilmRenderer:
    """Adapt a ready LocalAIRenderer to FilmOrchestrator's shot contract."""

    def __init__(
        self,
        renderer: Any,
        *,
        width: int = 576,
        height: int = 320,
        fps: int = 8,
        inference_steps: int = 25,
        guidance: float = 9.0,
    ) -> None:
        self.renderer = renderer
        self.width = width
        self.height = height
        self.fps = fps
        self.inference_steps = inference_steps
        self.guidance = guidance

    @staticmethod
    def _seed(shot_id: str, prompt: str) -> int:
        digest = hashlib.sha256(f"{shot_id}\n{prompt}".encode()).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

    def render(self, planned: Any, target: str | Path) -> Path:
        payload = dict(getattr(planned, "payload", {}) or {})
        payload.setdefault("shot_id", planned.shot_id)
        prompt = build_prompt(payload)
        approved_refs = tuple(payload.get("approved_reference_ids", ()))
        cinedna_ids = tuple(payload.get("cinedna_ids", ()))
        request = RenderRequest(
            job_id=str(uuid4()),
            shot_id=planned.shot_id,
            prompt=prompt,
            seed=self._seed(planned.shot_id, prompt),
            output_path=Path(target),
            width=self.width,
            height=self.height,
            fps=self.fps,
            duration=float(planned.duration),
            inference_steps=self.inference_steps,
            guidance=self.guidance,
            approved_reference_ids=approved_refs,
            cinedna_ids=cinedna_ids,
            metadata={
                "scene_id": planned.scene_id,
                "continuity_key": payload.get("continuity_key"),
                "character_ids": tuple(payload.get("character_ids", ())),
            },
        )
        result = self.renderer.render(request)
        output_path = getattr(result, "output_path", target)
        return Path(output_path)

    def cancel_pending(self) -> None:
        cancel = getattr(self.renderer, "cancel", None)
        if callable(cancel):
            cancel()
