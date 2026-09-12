"""External LatentSync SyncNet adapter for measured dialogue lip-sync QC.

CINEOS does not claim the SyncNet weights or LatentSync code as native capability.
This adapter runs an explicitly pinned external checkout against the rendered artifact,
verifies the configured checkpoint bytes, parses real audiovisual confidence/offset
measurements, and exposes a conservative pass/fail score to the production semantic
ensemble.  It exists specifically so visual-only judges are never mislabeled as
mouth-to-dialogue synchronization evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .artifact_video_observer import RGBVideoSample
from .semantic_video_ensemble import SemanticScorerComponent

LATENTSYNC_SYNCNET_SCHEMA = "cineos-latentsync-syncnet-av-qc/0.1"
LATENTSYNC_REPOSITORY = "bytedance/LatentSync"
LATENTSYNC_PINNED_REVISION = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
LATENTSYNC_CODE_LICENSE = "Apache-2.0"
LATENTSYNC_CHECKPOINT_LICENSE = "OpenRAIL++"
DIALOGUE_LIP_SYNC_METRIC = "dialogue_lip_sync"

_CONFIDENCE_RE = re.compile(r"SyncNet confidence:\s*([-+]?\d+(?:\.\d+)?)")
_OFFSET_RE = re.compile(r"AV offset:\s*([-+]?\d+)")


class LatentSyncSyncNetError(RuntimeError):
    """Raised when real audiovisual lip-sync evidence cannot be established."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LatentSyncSyncNetError(f"cannot read SyncNet checkpoint: {path}") from exc
    return digest.hexdigest()


class LatentSyncSyncNetScorer:
    """Measure AV synchrony with a hash-bound, revision-pinned LatentSync SyncNet.

    Upstream SyncNet confidence is not a calibrated probability, so CINEOS does not
    pretend that it is one.  The production quality metric is deliberately binary:
    1.0 only when both the measured confidence floor and absolute AV-offset bound pass,
    otherwise 0.0.  The raw measurements and thresholds remain in runtime provenance.
    """

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
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.revision = revision
        self.checkpoint_sha256 = checkpoint_sha256
        self.minimum_confidence = float(minimum_confidence)
        self.maximum_abs_offset_frames = int(maximum_abs_offset_frames)
        self.python_executable = python_executable
        self.timeout_seconds = int(timeout_seconds)
        self._verified = False
        self.last_measurement: dict[str, float | int] | None = None

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
        observed_revision = completed.stdout.strip()
        if observed_revision != self.revision:
            raise LatentSyncSyncNetError(
                "LatentSync checkout revision does not match the pinned production revision"
            )
        observed_hash = _sha256(self.checkpoint_path)
        if observed_hash != self.checkpoint_sha256:
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
            "limitations": [
                "not a CINEOS-native model",
                "requires audible dialogue and a detectable speaking face",
                "upstream SyncNet confidence is thresholded rather than treated as probability",
            ],
        }

    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        del sample, shot, attempt_index
        self._verify_external_runtime()
        artifact_path = Path(artifact).resolve()
        if not artifact_path.is_file():
            raise LatentSyncSyncNetError(
                f"rendered audiovisual artifact is unavailable: {artifact_path}"
            )

        with tempfile.TemporaryDirectory(prefix="cineos-latentsync-qc-") as temp:
            temp_root = Path(temp)
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
                str(temp_root / "temp"),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=temp_root,
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

        output = f"{completed.stdout}\n{completed.stderr}"
        confidence_match = _CONFIDENCE_RE.search(output)
        offset_match = _OFFSET_RE.search(output)
        if confidence_match is None or offset_match is None:
            raise LatentSyncSyncNetError(
                "LatentSync SyncNet output did not contain confidence and AV offset"
            )
        confidence = float(confidence_match.group(1))
        offset_frames = int(offset_match.group(1))
        self.last_measurement = {
            "syncnet_confidence": confidence,
            "av_offset_frames": offset_frames,
        }
        passed = (
            confidence >= self.minimum_confidence
            and abs(offset_frames) <= self.maximum_abs_offset_frames
        )
        return {DIALOGUE_LIP_SYNC_METRIC: 1.0 if passed else 0.0}


def latentsync_syncnet_component(
    scorer: LatentSyncSyncNetScorer,
) -> SemanticScorerComponent:
    """Expose the AV scorer with exact metric ownership to the semantic ensemble."""

    if not isinstance(scorer, LatentSyncSyncNetScorer):
        raise TypeError("scorer must be LatentSyncSyncNetScorer")
    return SemanticScorerComponent(
        name="latentsync_syncnet_av",
        scorer=scorer,
        measured_metrics=(DIALOGUE_LIP_SYNC_METRIC,),
    )


__all__ = [
    "DIALOGUE_LIP_SYNC_METRIC",
    "LATENTSYNC_CHECKPOINT_LICENSE",
    "LATENTSYNC_CODE_LICENSE",
    "LATENTSYNC_PINNED_REVISION",
    "LATENTSYNC_REPOSITORY",
    "LATENTSYNC_SYNCNET_SCHEMA",
    "LatentSyncSyncNetError",
    "LatentSyncSyncNetScorer",
    "latentsync_syncnet_component",
]
