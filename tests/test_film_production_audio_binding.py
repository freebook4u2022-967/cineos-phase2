import hashlib
from types import SimpleNamespace

import pytest

from cineos.film import audio_binding
from cineos.film import production_delivery_gate as delivery
from cineos.film.audio_binding import AudioBindingError, measure_audio_binding


def _wave(length: int = 1200) -> tuple[int, ...]:
    return tuple(((index * 97) % 20001) - 10000 for index in range(length))


def test_measure_audio_binding_accepts_small_encoder_alignment(tmp_path, monkeypatch):
    approved = tmp_path / "approved.wav"
    final = tmp_path / "film.mp4"
    approved.write_bytes(b"approved")
    final.write_bytes(b"final")
    wave = _wave()
    decoded = iter((wave, (0, 0, 0, 0) + wave))
    monkeypatch.setattr(
        audio_binding, "_decode_binding_pcm", lambda _path: next(decoded)
    )

    evidence = measure_audio_binding(approved, final)

    assert evidence.accepted is True
    assert evidence.correlation == pytest.approx(1.0)
    assert abs(evidence.alignment_lag_samples) <= 16
    assert evidence.approved_audio_sha256 == hashlib.sha256(b"approved").hexdigest()
    assert evidence.final_artifact_sha256 == hashlib.sha256(b"final").hexdigest()


def test_measure_audio_binding_rejects_substituted_soundtrack(tmp_path, monkeypatch):
    approved = tmp_path / "approved.wav"
    final = tmp_path / "film.mp4"
    approved.write_bytes(b"approved")
    final.write_bytes(b"final")
    first = _wave()
    second = tuple(((index * 31 + 7000) % 18013) - 9006 for index in range(1200))
    decoded = iter((first, second))
    monkeypatch.setattr(
        audio_binding, "_decode_binding_pcm", lambda _path: next(decoded)
    )

    with pytest.raises(AudioBindingError, match="does not match the approved"):
        measure_audio_binding(approved, final)


def test_delivery_gate_binds_final_mp4_to_approved_mix(tmp_path, monkeypatch):
    approved = tmp_path / "approved.wav"
    final = tmp_path / "film.mp4"
    approved.write_bytes(b"approved")
    final.write_bytes(b"final")
    approved_sha = hashlib.sha256(b"approved").hexdigest()
    final_sha = hashlib.sha256(b"final").hexdigest()
    connected = SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True})
    binding = SimpleNamespace(
        accepted=True,
        approved_audio_sha256=approved_sha,
        final_artifact_sha256=final_sha,
        to_dict=lambda: {"accepted": True},
    )
    monkeypatch.setattr(
        delivery,
        "validate_connected_production_film_evidence",
        lambda *_a, **_k: connected,
    )
    monkeypatch.setattr(delivery, "measure_audio_binding", lambda *_a, **_k: binding)
    assembly = {
        "audio": {"path": str(approved), "sha256": approved_sha},
        "final_mp4": str(final),
        "final_mp4_sha256": final_sha,
    }

    evidence = delivery.validate_production_delivery_evidence(object(), assembly)

    assert evidence.accepted is True
    assert evidence.audio_binding is binding


def test_delivery_gate_rejects_binding_to_different_final_artifact(
    tmp_path, monkeypatch
):
    approved = tmp_path / "approved.wav"
    final = tmp_path / "film.mp4"
    approved.write_bytes(b"approved")
    final.write_bytes(b"final")
    approved_sha = hashlib.sha256(b"approved").hexdigest()
    final_sha = hashlib.sha256(b"final").hexdigest()
    connected = SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True})
    binding = SimpleNamespace(
        accepted=True,
        approved_audio_sha256=approved_sha,
        final_artifact_sha256="f" * 64,
        to_dict=lambda: {"accepted": True},
    )
    monkeypatch.setattr(
        delivery,
        "validate_connected_production_film_evidence",
        lambda *_a, **_k: connected,
    )
    monkeypatch.setattr(delivery, "measure_audio_binding", lambda *_a, **_k: binding)
    assembly = {
        "audio": {"path": str(approved), "sha256": approved_sha},
        "final_mp4": str(final),
        "final_mp4_sha256": final_sha,
    }

    with pytest.raises(
        delivery.ProductionDeliveryEvidenceError, match="final MP4 hash"
    ):
        delivery.validate_production_delivery_evidence(object(), assembly)
