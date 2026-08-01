"""Minimal deterministic validator-backend example."""

from cineos.validation import FakeValidatorBackend, ValidationPipeline

backend = FakeValidatorBackend(
    scores={"identity.face": 0.92, "wardrobe.colors": 0.88},
    temporal={"frame_flicker": 0.03, "face_drift": 0.02},
)
pipeline = ValidationPipeline(backend)
