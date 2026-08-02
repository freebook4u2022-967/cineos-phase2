"""Portable PCM mixing with an optional safe FFmpeg conversion boundary."""

from __future__ import annotations

import shutil
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from .exceptions import AudioMixError
from .project import MixSettings


@dataclass(frozen=True, slots=True)
class MixInput:
    path: Path
    start_time: float = 0.0
    gain: float = 1.0
    kind: str = "dialogue"
    fade_in: float = 0.0
    fade_out: float = 0.0
    pan: float = 0.0
    muted: bool = False


class Mixer:
    def __init__(
        self,
        sample_rate: int = 48_000,
        channel_layout: str = "stereo",
        settings: MixSettings | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channel_layout = channel_layout
        self.settings = settings or MixSettings()

    @property
    def channels(self) -> int:
        return {"mono": 1, "stereo": 2}.get(self.channel_layout, 2)

    def mix(
        self,
        tracks: list[MixInput],
        output: str | Path,
        *,
        duration: float | None = None,
    ) -> Path:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        available = [item for item in tracks if not item.muted and item.path.is_file()]
        decoded: list[tuple[MixInput, array[int], int]] = []
        max_frames = round((duration or 0) * self.sample_rate)
        for item in available:
            try:
                with wave.open(str(item.path), "rb") as source:
                    if source.getsampwidth() != 2:
                        raise AudioMixError(
                            "only 16-bit PCM WAV inputs are supported by portable mixer"
                        )
                    samples = array("h", source.readframes(source.getnframes()))
                    input_channels = source.getnchannels()
                    rate = source.getframerate()
                    if rate != self.sample_rate:
                        frame_count = len(samples) // input_channels
                        output_count = round(frame_count * self.sample_rate / rate)
                        samples = array(
                            "h",
                            (
                                samples[
                                    min(
                                        frame_count - 1,
                                        int(index * rate / self.sample_rate),
                                    )
                                    * input_channels
                                    + channel
                                ]
                                for index in range(output_count)
                                for channel in range(input_channels)
                            ),
                        )
                    if input_channels == 1 and self.channels == 2:
                        samples = array(
                            "h", (value for item in samples for value in (item, item))
                        )
                    elif input_channels != self.channels:
                        raise AudioMixError(
                            "unsupported input/output channel conversion"
                        )
                gain = item.gain
                if item.kind == "ambience":
                    gain *= self.settings.ambience_gain
                elif item.kind == "effects":
                    gain *= self.settings.effects_gain
                elif item.kind == "music":
                    gain *= self.settings.music_gain
                samples = array(
                    "h",
                    (max(-32768, min(32767, round(item * gain))) for item in samples),
                )
                start = round(item.start_time * self.sample_rate)
                decoded.append((item, samples, start))
                max_frames = max(max_frames, start + len(samples) // self.channels)
            except (wave.Error, OSError) as error:
                raise AudioMixError(f"cannot read {item.path}: {error}") from error
        mixed = array("h", [0]) * (max_frames * self.channels)
        for _, samples, start in decoded:
            offset = start * self.channels
            for index, sample in enumerate(samples):
                mixed[offset + index] = max(
                    -32768, min(32767, mixed[offset + index] + sample)
                )
        with wave.open(str(target), "wb") as result:
            result.setparams(
                (
                    self.channels,
                    2,
                    self.sample_rate,
                    max_frames,
                    "NONE",
                    "not compressed",
                )
            )
            result.writeframes(mixed.tobytes())
        return target

    def convert(self, source: str | Path, output: str | Path) -> Path:
        if not shutil.which("ffmpeg"):
            raise AudioMixError("FFmpeg is required for compressed audio export")
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-i", str(source), "-vn", str(target)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            raise AudioMixError(completed.stderr.strip() or "FFmpeg conversion failed")
        return target
