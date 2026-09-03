from __future__ import annotations

import pytest

from cineos.atlas.siglip2_video_scorer import SigLIP2FeatureVideoScorer


def test_motion_qc_rejects_fully_frozen_feature_sequence() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    )

    assert score == 0.0


def test_motion_qc_rejects_frozen_two_frame_sequence() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence([(1.0, 0.0), (1.0, 0.0)])

    assert score == 0.0


def test_motion_qc_preserves_nonzero_two_frame_motion() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence([(1.0, 0.0), (0.0, 1.0)])

    assert score == 1.0


def test_motion_qc_preserves_stable_nonzero_feature_steps() -> None:
    root_half = 2**-0.5
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [(1.0, 0.0), (root_half, root_half), (0.0, 1.0)]
    )

    assert score == pytest.approx(1.0)


def test_motion_qc_single_frame_cannot_claim_motion_evidence() -> None:
    assert SigLIP2FeatureVideoScorer._motion_coherence([(1.0, 0.0)]) == 0.0
