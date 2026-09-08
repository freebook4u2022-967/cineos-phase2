import hashlib
import random
from types import SimpleNamespace

import pytest

from cineos.film import production_delivery_gate as delivery
from cineos.film import visual_binding
from cineos.film.visual_binding import VisualBindingError, measure_visual_binding


def _frames(seed: int, count: int = 4) -> bytes:
    rng = random.Random(seed)
    return bytes(
        rng.randrange(0, 256)
        for _ in range(visual_binding.VISUAL_BINDING_FRAME_BYTES * count)
    )


def test_measure_visual_binding_accepts_ordered_transcoded_sequence(
    tmp_path, monkeypatch
):
    first = tmp_path / "shot-1.mp4"
    second = tmp_path / "shot-2.mp4"
    final = tmp_path / "film.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    final.write_bytes(b"final")
    first_frames = _frames(11)
    second_frames = _frames(29)
    decoded = iter((first_frames, second_frames, first_frames + second_frames))
    monkeypatch.setattr(
        visual_binding,
        "_decode_binding_luma",
        lambda _path, duration_seconds=None: next(decoded),
    )

    evidence = measure_visual_binding([first, second], final)

    assert evidence.accepted is True
    assert evidence.correlation == pytest.approx(1.0)
    assert evidence.alignment_lag_frames == 0
    assert evidence.source_sha256 == (
        hashlib.sha256(b"first").hexdigest(),
        hashlib.sha256(b"second").hexdigest(),
    )
    assert evidence.final_artifact_sha256 == hashlib.sha256(b"final").hexdigest()


def test_measure_visual_binding_rejects_reordered_final_sequence(tmp_path, monkeypatch):
    first = tmp_path / "shot-1.mp4"
    second = tmp_path / "shot-2.mp4"
    final = tmp_path / "film.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    final.write_bytes(b"final")
    first_frames = _frames(101)
    second_frames = _frames(307)
    decoded = iter((first_frames, second_frames, second_frames + first_frames))
    monkeypatch.setattr(
        visual_binding,
        "_decode_binding_luma",
        lambda _path, duration_seconds=None: next(decoded),
    )

    with pytest.raises(
        VisualBindingError, match="does not match approved connected shots"
    ):
        measure_visual_binding([first, second], final)


def test_delivery_gate_requires_visual_binding_for_native_connected_evidence(
    tmp_path, monkeypatch
):
    first = tmp_path / "shot-1.mp4"
    final = tmp_path / "film.mp4"
    first.write_bytes(b"first")
    final.write_bytes(b"final")
    first_sha = hashlib.sha256(b"first").hexdigest()
    final_sha = hashlib.sha256(b"final").hexdigest()
    connected = SimpleNamespace(
        accepted=True,
        shot_count=5,
        dialogue_shot_ids=(),
        to_dict=lambda: {"accepted": True},
    )
    binding = SimpleNamespace(
        accepted=True,
        source_sha256=(first_sha,),
        final_artifact_sha256=final_sha,
        to_dict=lambda: {"accepted": True},
    )
    monkeypatch.setattr(
        delivery,
        "validate_connected_production_film_evidence",
        lambda *_a, **_k: connected,
    )
    monkeypatch.setattr(delivery, "measure_visual_binding", lambda *_a, **_k: binding)
    assembly = {
        "shots": [
            {
                "shot_id": "shot-1",
                "output_path": str(first),
                "output_sha256": first_sha,
            }
        ],
        "final_mp4": str(final),
        "final_mp4_sha256": final_sha,
    }

    evidence = delivery.validate_production_delivery_evidence(object(), assembly)

    assert evidence.accepted is True
    assert evidence.visual_binding is binding


def test_delivery_gate_rejects_visual_binding_to_substituted_source(
    tmp_path, monkeypatch
):
    first = tmp_path / "shot-1.mp4"
    final = tmp_path / "film.mp4"
    first.write_bytes(b"first")
    final.write_bytes(b"final")
    first_sha = hashlib.sha256(b"first").hexdigest()
    final_sha = hashlib.sha256(b"final").hexdigest()
    connected = SimpleNamespace(
        accepted=True,
        shot_count=5,
        dialogue_shot_ids=(),
        to_dict=lambda: {"accepted": True},
    )
    binding = SimpleNamespace(
        accepted=True,
        source_sha256=("f" * 64,),
        final_artifact_sha256=final_sha,
        to_dict=lambda: {"accepted": True},
    )
    monkeypatch.setattr(
        delivery,
        "validate_connected_production_film_evidence",
        lambda *_a, **_k: connected,
    )
    monkeypatch.setattr(delivery, "measure_visual_binding", lambda *_a, **_k: binding)
    assembly = {
        "shots": [
            {
                "shot_id": "shot-1",
                "output_path": str(first),
                "output_sha256": first_sha,
            }
        ],
        "final_mp4": str(final),
        "final_mp4_sha256": final_sha,
    }

    with pytest.raises(
        delivery.ProductionDeliveryEvidenceError,
        match="approved shot artifact hashes",
    ):
        delivery.validate_production_delivery_evidence(object(), assembly)
