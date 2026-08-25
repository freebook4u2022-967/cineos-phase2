"""Concrete CINEOS-owned temporal shot rendering path.

This renderer turns the existing native temporal latent runtime into real RGB
frames and then uses FFmpeg only as a container/video encoder. FFmpeg is not a
visual generation backend: every generated pixel originates from CINEOS temporal
model state. The implementation is deliberately dependency-light and provides a
production integration boundary that future learned decoders can replace without
changing complete-film orchestration semantics.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cineos.native_image.tensor_model import Tensor

from .runtime import NativeTemporalRuntime
from .temporal_model import TemporalFrameInput, TemporalSequenceState


class NativeShotRenderError(RuntimeError):
    """Raised when native temporal shot rendering cannot complete safely."""


class NativeLatentRGBDecoder(Protocol):
    """Stable native boundary for converting CINEOS latents into RGB frames.

    Implementations may be analytic or learned, but must be CINEOS-owned and must
    return exactly ``width * height * 3`` RGB bytes. Keeping this contract separate
    from shot orchestration allows a trained decoder to replace the bootstrap pixel
    decoder without changing continuity, retry, checkpoint, or film assembly logic.
    """

    decoder_id: str

    def decode(self, latent: Tensor, *, width: int, height: int) -> bytes: ...


def _vector(text: str, size: int, *, device: str) -> Tensor:
    """Create a deterministic conditioning vector from provider-neutral metadata."""
    if size <= 0:
        raise ValueError("conditioning vector size must be positive")
    values: list[float] = []
    counter = 0
    while len(values) < size:
        digest = hashlib.sha256(f"{text}\n{counter}".encode()).digest()
        for byte in digest:
            values.append((float(byte) / 127.5) - 1.0)
            if len(values) == size:
                break
        counter += 1
    return Tensor(tuple(values), (size,), device)


def _motion_vector(
    shot_id: str,
    frame_index: int,
    size: int,
    *,
    device: str,
) -> Tensor:
    base = _vector(f"{shot_id}:motion", size, device=device)
    phase = frame_index * 0.17320508075688773
    values = tuple(
        value * (0.55 + 0.45 * math.sin(phase + index * 0.71))
        for index, value in enumerate(base.values)
    )
    return Tensor(values, (size,), device)


def _latent_to_rgb(latent: Tensor, width: int, height: int) -> bytes:
    """Decode a native latent into deterministic RGB pixels without a provider.

    This bootstrap decoder is intentionally simple but real. It exists so the
    native video runtime has an executable owned pixel path today; a learned
    VAE/RGB decoder can implement :class:`NativeLatentRGBDecoder` later. Spatial
    frequencies are derived from the model latent rather than canned frames or
    fixtures.
    """
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if not latent.values:
        raise ValueError("latent must not be empty")

    values = latent.values
    count = len(values)
    rgb = bytearray(width * height * 3)
    offset = 0
    for y in range(height):
        ny = (y + 0.5) / height
        for x in range(width):
            nx = (x + 0.5) / width
            for channel in range(3):
                a = values[(channel * 5 + x + y) % count]
                b = values[(channel * 7 + 3) % count]
                c = values[(channel * 11 + x * 3 + y * 5) % count]
                signal = (
                    math.sin((nx * (2.0 + abs(a) * 4.0) + b) * math.tau)
                    + math.cos((ny * (2.0 + abs(c) * 4.0) - a) * math.tau)
                    + math.sin(((nx + ny) * (1.0 + abs(b) * 3.0) + c) * math.pi)
                ) / 3.0
                normalized = max(0.0, min(1.0, 0.5 + 0.42 * signal))
                rgb[offset] = int(round(normalized * 255.0))
                offset += 1
    return bytes(rgb)


@dataclass(frozen=True, slots=True)
class AnalyticLatentRGBDecoder:
    """Bootstrap CINEOS decoder used until a trained RGB decoder is supplied."""

    decoder_id: str = "cineos-analytic-latent-rgb/1"

    def decode(self, latent: Tensor, *, width: int, height: int) -> bytes:
        return _latent_to_rgb(latent, width, height)


def _write_ppm(path: Path, width: int, height: int, rgb: bytes) -> None:
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError(f"RGB payload has {len(rgb)} bytes; expected {expected}")
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb)


def _promote_state(
    source: TemporalSequenceState,
    destination: TemporalSequenceState,
) -> None:
    """Promote a fully rendered working state into the caller-owned state.

    Rendering is intentionally performed against a restored snapshot. The durable
    state is changed only after all frames have passed temporal QC and FFmpeg has
    produced a non-empty output. This keeps a failed whole-shot attempt from
    poisoning continuity even when failure happens after many accepted frames.
    """
    if source.shot_id != destination.shot_id:
        raise ValueError("cannot promote temporal state across different shots")
    destination.hidden = source.hidden
    destination.last_frame_index = source.last_frame_index
    destination.last_latent = source.last_latent
    destination.metadata = dict(source.metadata)


@dataclass(slots=True)
class CINEOSNativeTemporalShotRenderer:
    """Render complete shots using CINEOS temporal generation and transactional QC.

    ``NativeFilmRendererBinding`` supplies the active attempt state. Every frame
    advances an isolated working copy only through :class:`NativeTemporalRuntime`,
    whose QC/retry transaction rejects excessive latent drift before commit. The
    caller-visible state is promoted only after successful video encoding, and
    whole-shot film QC still decides whether the resulting attempt is promoted into
    durable scene continuity memory.
    """

    runtime: NativeTemporalRuntime = field(default_factory=NativeTemporalRuntime.default)
    decoder: NativeLatentRGBDecoder = field(default_factory=AnalyticLatentRGBDecoder)
    width: int = 320
    height: int = 180
    fps: int = 8
    ffmpeg_binary: str = "ffmpeg"
    max_frames: int = 2400

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("native shot dimensions must be positive")
        if self.fps <= 0:
            raise ValueError("native shot fps must be positive")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        decoder_id = getattr(self.decoder, "decoder_id", "")
        if not isinstance(decoder_id, str) or not decoder_id.strip():
            raise ValueError("native RGB decoder requires a non-empty decoder_id")

    def _conditioning(
        self, planned: Any, state: TemporalSequenceState
    ) -> tuple[Tensor, Tensor]:
        payload = dict(getattr(planned, "payload", {}) or {})
        identity_source = "|".join(
            str(value)
            for key in (
                "approved_reference_ids",
                "cinedna_ids",
                "character_ids",
            )
            for value in payload.get(key, ())
        )
        if not identity_source:
            identity_source = str(getattr(planned, "shot_id", ""))
        scene_source = "|".join(
            (
                str(getattr(planned, "scene_id", "")),
                str(payload.get("prompt", "")),
                str(payload.get("location_id", "")),
                str(payload.get("continuity_key", "")),
            )
        )
        return (
            _vector(
                identity_source,
                self.runtime.model.identity_dim,
                device=state.hidden.device,
            ),
            _vector(
                scene_source, self.runtime.model.scene_dim, device=state.hidden.device
            ),
        )

    def render(
        self,
        planned: Any,
        target: str | Path,
        *,
        temporal_state: TemporalSequenceState,
    ) -> Path:
        shot_id = str(getattr(planned, "shot_id", ""))
        if not shot_id:
            raise ValueError("planned shot requires a shot_id")
        if temporal_state.shot_id != shot_id:
            raise ValueError("temporal state must belong to the planned shot")
        duration = float(getattr(planned, "duration", 0.0))
        if duration <= 0:
            raise ValueError("planned shot duration must be positive")

        frame_count = max(1, int(round(duration * self.fps)))
        if frame_count > self.max_frames:
            raise NativeShotRenderError(
                "shot requests "
                f"{frame_count} frames; native safety limit is {self.max_frames}"
            )

        ffmpeg = shutil.which(self.ffmpeg_binary)
        if ffmpeg is None:
            raise NativeShotRenderError(
                f"FFmpeg encoder {self.ffmpeg_binary!r} is not available"
            )

        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Whole-shot transaction boundary: frame-level QC may commit to this working
        # state, but no caller-visible continuity is advanced until the final encoded
        # artifact exists and has been atomically promoted into place.
        working_state = TemporalSequenceState.restore(temporal_state.snapshot())
        identity, scene = self._conditioning(planned, working_state)

        with tempfile.TemporaryDirectory(
            prefix=f".cineos-{shot_id}-", dir=destination.parent
        ) as temp_dir:
            frames = Path(temp_dir)
            encoded = frames / "encoded.mp4"
            for frame_index in range(frame_count):
                request = TemporalFrameInput(
                    shot_id=shot_id,
                    frame_index=frame_index,
                    identity=identity,
                    scene=scene,
                    motion=_motion_vector(
                        shot_id,
                        frame_index,
                        self.runtime.model.motion_dim,
                        device=working_state.hidden.device,
                    ),
                )
                result = self.runtime.generate_frame(request, working_state)
                try:
                    rgb = self.decoder.decode(
                        result.candidate.latent,
                        width=self.width,
                        height=self.height,
                    )
                    _write_ppm(
                        frames / f"frame-{frame_index:06d}.ppm",
                        self.width,
                        self.height,
                        rgb,
                    )
                except Exception as exc:
                    raise NativeShotRenderError(
                        "native RGB decoder "
                        f"{self.decoder.decoder_id!r} failed at frame {frame_index}"
                    ) from exc

            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(self.fps),
                "-i",
                str(frames / "frame-%06d.ppm"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(encoded),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise NativeShotRenderError(
                    "FFmpeg failed to encode native frames: " + completed.stderr.strip()
                )
            if not encoded.is_file() or encoded.stat().st_size <= 0:
                raise NativeShotRenderError("native shot encoder produced no video output")

            # The staging directory lives under destination.parent so replace() is an
            # atomic same-filesystem promotion on supported platforms. An old durable
            # artifact is therefore never replaced by a partial/failed render.
            encoded.replace(destination)

        working_state.metadata["native_renderer"] = "cineos-temporal-pixel/0.3"
        working_state.metadata["native_decoder"] = self.decoder.decoder_id
        working_state.metadata["native_width"] = self.width
        working_state.metadata["native_height"] = self.height
        working_state.metadata["native_fps"] = self.fps
        working_state.metadata["native_frame_count"] = frame_count
        _promote_state(working_state, temporal_state)
        return destination
