"""External LatentSync SyncNet adapter for measured dialogue lip-sync QC.

CINEOS does not claim the SyncNet weights or LatentSync code as native capability.
This adapter runs an explicitly pinned external checkout against rendered artifacts,
verifies the configured checkpoint bytes, and exposes conservative audiovisual QC.

The pinned upstream evaluator averages measurements across detected face tracks. CINEOS
accepts that result directly only for a single detected track. Multi-face dialogue is
speaker-bound: single-speaker shots are rerun on the declared face track; alternating
speakers are evaluated cue-by-cue against their declared face tracks and timing windows.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_video_observer import RGBVideoSample
from .semantic_video_ensemble import SemanticScorerComponent

LATENTSYNC_SYNCNET_SCHEMA = "cineos-latentsync-syncnet-av-qc/0.5"
LATENTSYNC_REPOSITORY = "bytedance/LatentSync"
LATENTSYNC_PINNED_REVISION = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
LATENTSYNC_CODE_LICENSE = "Apache-2.0"
LATENTSYNC_CHECKPOINT_LICENSE = "OpenRAIL++"
DIALOGUE_LIP_SYNC_METRIC = "dialogue_lip_sync"
DIALOGUE_CHALLENGES = ("dialogue", "dialogue_lip_sync")
SPEAKER_FACE_TRACK_INDEX_KEY = "speaker_face_track_index"

_CONFIDENCE_RE = re.compile(r"SyncNet confidence:\s*([-+]?\d+(?:\.\d+)?)")
_OFFSET_RE = re.compile(r"AV offset:\s*([-+]?\d+)")


class LatentSyncSyncNetError(RuntimeError):
    """Raised when real audiovisual lip-sync evidence cannot be established."""


@dataclass(frozen=True, slots=True)
class _DialogueCueBinding:
    speaker_id: str
    face_track_index: int
    start_seconds: float
    end_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LatentSyncSyncNetError(f"cannot read SyncNet checkpoint: {path}") from exc
    return digest.hexdigest()


def _detected_face_tracks(temp_root: Path) -> tuple[Path, ...]:
    crop_dir = temp_root / "detect_results" / "crop"
    if not crop_dir.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in crop_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        )
    )


def _dialogue_timing(shot: Any) -> list[dict[str, Any]]:
    performance = getattr(shot, "performance", None)
    if not isinstance(performance, dict):
        raise LatentSyncSyncNetError(
            "multi-face dialogue requires performance.dialogue_timing speaker binding"
        )
    dialogue_timing = performance.get("dialogue_timing")
    if not isinstance(dialogue_timing, list) or not dialogue_timing:
        raise LatentSyncSyncNetError(
            "multi-face dialogue requires non-empty performance.dialogue_timing"
        )
    return dialogue_timing


def _speaker_track_binding(shot: Any) -> tuple[str, int]:
    """Resolve a whole-shot binding when every dialogue cue has one speaker/track."""

    speakers: set[str] = set()
    track_indices: set[int] = set()
    for index, cue in enumerate(_dialogue_timing(shot)):
        if not isinstance(cue, dict):
            raise LatentSyncSyncNetError(
                f"performance.dialogue_timing[{index}] must be a mapping"
            )
        speaker_id = cue.get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise LatentSyncSyncNetError(
                f"performance.dialogue_timing[{index}] requires speaker_id for multi-face QC"
            )
        speakers.add(speaker_id.strip())
        track_index = cue.get(SPEAKER_FACE_TRACK_INDEX_KEY)
        if (
            isinstance(track_index, bool)
            or not isinstance(track_index, int)
            or track_index < 0
        ):
            raise LatentSyncSyncNetError(
                f"performance.dialogue_timing[{index}].{SPEAKER_FACE_TRACK_INDEX_KEY} "
                "must be a non-negative integer for multi-face QC"
            )
        track_indices.add(track_index)

    if len(speakers) != 1:
        raise LatentSyncSyncNetError(
            "multi-face dialogue with multiple speakers requires cue-window speaker-bound evaluation"
        )
    if len(track_indices) != 1:
        raise LatentSyncSyncNetError(
            "all dialogue cues for one speaker must bind the same speaker_face_track_index"
        )
    return next(iter(speakers)), next(iter(track_indices))


def _cue_window_bindings(shot: Any) -> tuple[_DialogueCueBinding, ...]:
    """Validate exact speaker/track/time evidence for alternating-speaker dialogue."""

    bindings: list[_DialogueCueBinding] = []
    for index, cue in enumerate(_dialogue_timing(shot)):
        if not isinstance(cue, dict):
            raise LatentSyncSyncNetError(
                f"performance.dialogue_timing[{index}] must be a mapping"
            )
        speaker_id = cue.get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise LatentSyncSyncNetError(
                f"performance.dialogue_timing[{index}] requires speaker_id for multi-face QC"
            )
        track_index = cue.get(SPEAKER_FACE_TRACK_INDEX_KEY)
        if (
            isinstance(track_index, bool)
            or not isinstance(track_index, int)
            or track_index < 0
        ):
            raise LatentSyncSyncNetError(
                f"performance.dialogue_timing[{index}].{SPEAKER_FACE_TRACK_INDEX_KEY} "
                "must be a non-negative integer for multi-face QC"
            )
        start = cue.get("start_seconds")
        end = cue.get("end_seconds")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise LatentSyncSyncNetError(
                "multi-face dialogue with multiple speakers requires finite "
                "start_seconds/end_seconds for cue-window evaluation"
            )
        start_f = float(start)
        end_f = float(end)
        if (
            not math.isfinite(start_f)
            or not math.isfinite(end_f)
            or start_f < 0.0
            or end_f <= start_f
        ):
            raise LatentSyncSyncNetError(
                "multi-face dialogue with multiple speakers requires valid positive cue windows"
            )
        bindings.append(
            _DialogueCueBinding(
                speaker_id=speaker_id.strip(),
                face_track_index=track_index,
                start_seconds=start_f,
                end_seconds=end_f,
            )
        )
    if len({binding.speaker_id for binding in bindings}) < 2:
        raise LatentSyncSyncNetError(
            "cue-window dialogue evaluation requires multiple speakers"
        )
    return tuple(bindings)


def _parse_syncnet_output(completed: subprocess.CompletedProcess) -> tuple[float, int]:
    output = f"{completed.stdout}\n{completed.stderr}"
    confidence_match = _CONFIDENCE_RE.search(output)
    offset_match = _OFFSET_RE.search(output)
    if confidence_match is None or offset_match is None:
        raise LatentSyncSyncNetError(
            "LatentSync SyncNet output did not contain confidence and AV offset"
        )
    return float(confidence_match.group(1)), int(offset_match.group(1))


class LatentSyncSyncNetScorer:
    """Measure AV synchrony with a hash-bound, revision-pinned LatentSync SyncNet."""

    semantic_measurement_evidence = True

    def __init__(
        self,
        *,
        repository_root: str | Path,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        revision: str = LATENTSYNC_PINNED_REVISION,
        minimum_confidence: float = 3.0,
        maximum_abs_offset_frames: int = 2,
        python_executable: str = sys.executable,
        ffmpeg_executable: str = "ffmpeg",
        timeout_seconds: int = 180,
    ) -> None:
        self.repository_root = Path(repository_root).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable 40-character git SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if minimum_confidence < 0:
            raise ValueError("minimum_confidence must be non-negative")
        if maximum_abs_offset_frames < 0:
            raise ValueError("maximum_abs_offset_frames must be non-negative")
        if not isinstance(python_executable, str) or not python_executable.strip():
            raise ValueError("python_executable must be non-empty")
        if not isinstance(ffmpeg_executable, str) or not ffmpeg_executable.strip():
            raise ValueError("ffmpeg_executable must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.revision = revision
        self.checkpoint_sha256 = checkpoint_sha256
        self.minimum_confidence = float(minimum_confidence)
        self.maximum_abs_offset_frames = int(maximum_abs_offset_frames)
        self.python_executable = python_executable
        self.ffmpeg_executable = ffmpeg_executable
        self.timeout_seconds = int(timeout_seconds)
        self._verified = False
        self.last_measurement: dict[str, float | int | str] | None = None

    def _verify_external_runtime(self) -> None:
        if self._verified:
            return
        if not self.repository_root.is_dir():
            raise LatentSyncSyncNetError(
                f"LatentSync repository is unavailable: {self.repository_root}"
            )
        if not self.checkpoint_path.is_file():
            raise LatentSyncSyncNetError(
                f"SyncNet checkpoint is unavailable: {self.checkpoint_path}"
            )
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LatentSyncSyncNetError(
                "cannot verify pinned LatentSync repository revision"
            ) from exc
        if completed.stdout.strip() != self.revision:
            raise LatentSyncSyncNetError(
                "LatentSync checkout revision does not match the pinned production revision"
            )
        if _sha256(self.checkpoint_path) != self.checkpoint_sha256:
            raise LatentSyncSyncNetError(
                "SyncNet checkpoint SHA-256 does not match approved production bytes"
            )
        self._verified = True

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": LATENTSYNC_SYNCNET_SCHEMA,
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
            "repository": LATENTSYNC_REPOSITORY,
            "repository_revision": self.revision,
            "code_license": LATENTSYNC_CODE_LICENSE,
            "checkpoint_license": LATENTSYNC_CHECKPOINT_LICENSE,
            "checkpoint_sha256": self.checkpoint_sha256,
            "measurement_method": "latentsync_syncnet_confidence_and_av_offset",
            "measured_metrics": [DIALOGUE_LIP_SYNC_METRIC],
            "minimum_confidence": self.minimum_confidence,
            "maximum_abs_offset_frames": self.maximum_abs_offset_frames,
            "score_semantics": "binary_pass_fail_not_probability",
            "face_track_policy": "single_track_or_explicit_speaker_bound_track_or_cue_windows",
            "speaker_track_binding_key": SPEAKER_FACE_TRACK_INDEX_KEY,
            "multi_speaker_policy": "speaker_bound_cue_window_evaluation",
            "limitations": [
                "not a CINEOS-native model",
                "requires audible dialogue and a detectable speaking face",
                "upstream SyncNet confidence is thresholded rather than treated as probability",
                "speaker face-track binding must be supplied by an auditable tracking stage",
            ],
        }

    def _run_syncnet(
        self, artifact_path: Path, *, work_root: Path
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.repository_root) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        command = [
            self.python_executable,
            "-m",
            "eval.eval_sync_conf",
            "--initial_model",
            str(self.checkpoint_path),
            "--video_path",
            str(artifact_path),
            "--temp_dir",
            str(work_root / "temp"),
        ]
        try:
            return subprocess.run(
                command,
                cwd=work_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LatentSyncSyncNetError(
                "LatentSync SyncNet audiovisual evaluation failed"
            ) from exc

    def _speaker_bound_artifact(
        self,
        *,
        face_track: Path,
        original_artifact: Path,
        output_path: Path,
    ) -> None:
        command = [
            self.ffmpeg_executable,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(face_track),
            "-i",
            str(original_artifact),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        self._run_ffmpeg(command, output_path, "speaker-bound audiovisual artifact")

    def _speaker_bound_cue_artifact(
        self,
        *,
        face_track: Path,
        original_artifact: Path,
        output_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        # Filter-domain trimming gives frame/sample-accurate cue boundaries instead of
        # keyframe-limited input seeking. Re-encoding is intentional QC preprocessing;
        # SyncNet evidence is still computed from the original rendered pixels/audio.
        filter_graph = (
            f"[0:v]trim=start={start_seconds:.6f}:end={end_seconds:.6f},"
            "setpts=PTS-STARTPTS[v];"
            f"[1:a]atrim=start={start_seconds:.6f}:end={end_seconds:.6f},"
            "asetpts=PTS-STARTPTS[a]"
        )
        command = [
            self.ffmpeg_executable,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(face_track),
            "-i",
            str(original_artifact),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        self._run_ffmpeg(command, output_path, "speaker-bound dialogue cue artifact")

    def _run_ffmpeg(self, command: list[str], output_path: Path, label: str) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LatentSyncSyncNetError(f"could not produce {label}") from exc
        if not output_path.is_file():
            raise LatentSyncSyncNetError(f"{label} was not produced by ffmpeg")

    def _evaluate_cue_windows(
        self,
        *,
        shot: Any,
        face_tracks: tuple[Path, ...],
        original_artifact: Path,
        temp_root: Path,
    ) -> tuple[float, int, int, str]:
        bindings = _cue_window_bindings(shot)
        confidences: list[float] = []
        offsets: list[int] = []
        speakers: list[str] = []
        for cue_index, binding in enumerate(bindings):
            if binding.face_track_index >= len(face_tracks):
                raise LatentSyncSyncNetError(
                    f"speaker_face_track_index {binding.face_track_index} is out of range "
                    f"for {len(face_tracks)} detected face tracks"
                )
            cue_artifact = temp_root / f"speaker-cue-{cue_index:03d}.mp4"
            self._speaker_bound_cue_artifact(
                face_track=face_tracks[binding.face_track_index],
                original_artifact=original_artifact,
                output_path=cue_artifact,
                start_seconds=binding.start_seconds,
                end_seconds=binding.end_seconds,
            )
            cue_root = temp_root / f"speaker-cue-eval-{cue_index:03d}"
            cue_root.mkdir()
            completed = self._run_syncnet(cue_artifact, work_root=cue_root)
            detected = len(_detected_face_tracks(cue_root))
            if detected != 1:
                raise LatentSyncSyncNetError(
                    "speaker-bound dialogue cue rerun requires exactly one detected "
                    f"face track; pinned LatentSync produced {detected} for cue {cue_index}"
                )
            confidence, offset = _parse_syncnet_output(completed)
            confidences.append(confidence)
            offsets.append(offset)
            speakers.append(binding.speaker_id)
        worst_offset = max(offsets, key=lambda value: abs(value))
        return (
            min(confidences),
            worst_offset,
            len(bindings),
            ",".join(sorted(set(speakers))),
        )

    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        del sample, attempt_index
        self._verify_external_runtime()
        artifact_path = Path(artifact).resolve()
        if not artifact_path.is_file():
            raise LatentSyncSyncNetError(
                f"rendered audiovisual artifact is unavailable: {artifact_path}"
            )

        speaker_id: str | None = None
        speaker_face_track_index: int | None = None
        evaluation_mode = "single_detected_track"
        cue_count = 0
        cue_speakers = ""
        with tempfile.TemporaryDirectory(prefix="cineos-latentsync-qc-") as temp:
            temp_root = Path(temp)
            initial = self._run_syncnet(artifact_path, work_root=temp_root)
            face_tracks = _detected_face_tracks(temp_root)
            original_face_track_count = len(face_tracks)
            if original_face_track_count == 0:
                raise LatentSyncSyncNetError(
                    "production dialogue lip-sync requires a detected face track; "
                    "pinned LatentSync produced 0"
                )

            accepted = initial
            accepted_face_track_count = original_face_track_count
            if original_face_track_count > 1:
                try:
                    speaker_id, speaker_face_track_index = _speaker_track_binding(shot)
                except LatentSyncSyncNetError as exc:
                    if "multiple speakers" not in str(exc):
                        raise
                    confidence, offset_frames, cue_count, cue_speakers = (
                        self._evaluate_cue_windows(
                            shot=shot,
                            face_tracks=face_tracks,
                            original_artifact=artifact_path,
                            temp_root=temp_root,
                        )
                    )
                    accepted_face_track_count = 1
                    evaluation_mode = "speaker_bound_cue_windows"
                else:
                    if speaker_face_track_index >= original_face_track_count:
                        raise LatentSyncSyncNetError(
                            f"speaker_face_track_index {speaker_face_track_index} is out of range "
                            f"for {original_face_track_count} detected face tracks"
                        )
                    bound_artifact = temp_root / "speaker-bound.mp4"
                    self._speaker_bound_artifact(
                        face_track=face_tracks[speaker_face_track_index],
                        original_artifact=artifact_path,
                        output_path=bound_artifact,
                    )
                    bound_root = temp_root / "speaker-bound-eval"
                    bound_root.mkdir()
                    accepted = self._run_syncnet(bound_artifact, work_root=bound_root)
                    accepted_face_track_count = len(_detected_face_tracks(bound_root))
                    if accepted_face_track_count != 1:
                        raise LatentSyncSyncNetError(
                            "speaker-bound dialogue lip-sync rerun requires exactly one detected "
                            f"face track; pinned LatentSync produced {accepted_face_track_count}"
                        )
                    evaluation_mode = "speaker_bound_whole_shot"

            if evaluation_mode != "speaker_bound_cue_windows":
                confidence, offset_frames = _parse_syncnet_output(accepted)
                if accepted_face_track_count != 1:
                    raise LatentSyncSyncNetError(
                        "production dialogue lip-sync requires exactly one accepted face track"
                    )

        measurement: dict[str, float | int | str] = {
            "syncnet_confidence": confidence,
            "av_offset_frames": offset_frames,
            "detected_face_tracks": original_face_track_count,
            "accepted_face_tracks": accepted_face_track_count,
        }
        if speaker_id is not None and speaker_face_track_index is not None:
            measurement.update(
                {
                    "speaker_id": speaker_id,
                    "speaker_face_track_index": speaker_face_track_index,
                    "speaker_binding_source": (
                        "performance.dialogue_timing[*].speaker_face_track_index"
                    ),
                }
            )
        if evaluation_mode == "speaker_bound_cue_windows":
            measurement.update(
                {
                    "speaker_evaluation_mode": evaluation_mode,
                    "evaluated_dialogue_cues": cue_count,
                    "evaluated_speakers": cue_speakers,
                    "speaker_binding_source": (
                        "performance.dialogue_timing[*].speaker_face_track_index+cue_window"
                    ),
                }
            )
        self.last_measurement = measurement
        passed = (
            confidence >= self.minimum_confidence
            and abs(offset_frames) <= self.maximum_abs_offset_frames
        )
        return {DIALOGUE_LIP_SYNC_METRIC: 1.0 if passed else 0.0}


def latentsync_syncnet_component(
    scorer: LatentSyncSyncNetScorer,
) -> SemanticScorerComponent:
    """Expose AV scoring only on shots that explicitly declare dialogue."""

    if not isinstance(scorer, LatentSyncSyncNetScorer):
        raise TypeError("scorer must be LatentSyncSyncNetScorer")
    return SemanticScorerComponent(
        name="latentsync_syncnet_av",
        scorer=scorer,
        measured_metrics=(DIALOGUE_LIP_SYNC_METRIC,),
        required_challenges=DIALOGUE_CHALLENGES,
    )


__all__ = [
    "DIALOGUE_CHALLENGES",
    "DIALOGUE_LIP_SYNC_METRIC",
    "LATENTSYNC_CHECKPOINT_LICENSE",
    "LATENTSYNC_CODE_LICENSE",
    "LATENTSYNC_PINNED_REVISION",
    "LATENTSYNC_REPOSITORY",
    "LATENTSYNC_SYNCNET_SCHEMA",
    "SPEAKER_FACE_TRACK_INDEX_KEY",
    "LatentSyncSyncNetError",
    "LatentSyncSyncNetScorer",
    "latentsync_syncnet_component",
]
