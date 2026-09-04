from pathlib import Path

import pytest

from cineos.film.boundary_continuity import measure_connected_boundaries
from cineos.film.exceptions import AssemblyError


def _movies(tmp_path: Path, count: int = 5) -> list[Path]:
    movies = []
    for index in range(count):
        path = tmp_path / f"shot-{index}.mp4"
        path.write_bytes(f"shot-{index}".encode())
        movies.append(path)
    return movies


def test_measures_declared_continuous_boundary_at_approved_edit_endpoint(
    tmp_path, monkeypatch
):
    movies = _movies(tmp_path)
    samples = []
    frame = bytes([100]) * (64 * 36)

    def fake_decode(movie, *, timestamp_seconds):
        samples.append((Path(movie).name, timestamp_seconds))
        return frame

    monkeypatch.setattr(
        "cineos.film.boundary_continuity._decode_luma_frame", fake_decode
    )
    evidence = measure_connected_boundaries(
        movies,
        transitions=["continuous", "cut", "cut", "cut"],
        durations=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    boundary = evidence["boundaries"][0]
    assert boundary["accepted"] is True
    assert boundary["measured"] is True
    assert boundary["similarity"] == 1.0
    assert boundary["timing_source"] == "approved-edit-endpoint"
    assert samples == [("shot-0.mp4", 0.95), ("shot-1.mp4", 0.0)]
    assert len(boundary["from_frame_sha256"]) == 64
    assert len(boundary["from_artifact_sha256"]) == 64
    assert boundary["from_artifact_sha256"] == evidence["shot_artifacts"][0]["sha256"]
    assert boundary["to_artifact_sha256"] == evidence["shot_artifacts"][1]["sha256"]
    assert evidence["schema"] == "cineos-boundary-continuity-evidence/0.2"
    assert evidence["limitations"].startswith("not semantic identity")


def test_rejects_declared_continuous_boundary_with_visual_jump(tmp_path, monkeypatch):
    movies = _movies(tmp_path)
    dark = bytes([0]) * (64 * 36)
    bright = bytes([255]) * (64 * 36)
    frames = iter([dark, bright])

    monkeypatch.setattr(
        "cineos.film.boundary_continuity._decode_luma_frame",
        lambda *_args, **_kwargs: next(frames),
    )

    with pytest.raises(AssemblyError, match="failed decoded visual continuity"):
        measure_connected_boundaries(
            movies,
            transitions=["continuous", "cut", "cut", "cut"],
            durations=[1.0] * 5,
        )


def test_intentional_cuts_are_explicitly_recorded_not_mislabeled_as_continuity(
    tmp_path, monkeypatch
):
    movies = _movies(tmp_path)
    monkeypatch.setattr(
        "cineos.film.boundary_continuity._decode_luma_frame",
        lambda *_args, **_kwargs: pytest.fail("intentional cuts must not be measured"),
    )

    evidence = measure_connected_boundaries(
        movies,
        transitions=["cut", "cut", "cut", "cut"],
    )

    assert all(item["accepted"] for item in evidence["boundaries"])
    assert all(not item["measured"] for item in evidence["boundaries"])
    assert all(
        item["reason"] == "intentional-cut-explicitly-declared"
        for item in evidence["boundaries"]
    )
    assert len(evidence["shot_artifacts"]) == len(movies)
    assert all(len(item["sha256"]) == 64 for item in evidence["shot_artifacts"])


def test_cut_only_evidence_still_requires_every_exact_shot_artifact(tmp_path, monkeypatch):
    movies = _movies(tmp_path)
    movies[2].unlink()
    monkeypatch.setattr(
        "cineos.film.boundary_continuity._decode_luma_frame",
        lambda *_args, **_kwargs: pytest.fail("cuts must not trigger frame decode"),
    )

    with pytest.raises(AssemblyError, match="missing or empty continuity artifact"):
        measure_connected_boundaries(
            movies,
            transitions=["cut", "cut", "cut", "cut"],
        )


def test_cut_only_evidence_rejects_empty_shot_artifact(tmp_path, monkeypatch):
    movies = _movies(tmp_path)
    movies[3].write_bytes(b"")
    monkeypatch.setattr(
        "cineos.film.boundary_continuity._decode_luma_frame",
        lambda *_args, **_kwargs: pytest.fail("cuts must not trigger frame decode"),
    )

    with pytest.raises(AssemblyError, match="missing or empty continuity artifact"):
        measure_connected_boundaries(
            movies,
            transitions=["cut", "cut", "cut", "cut"],
        )


def test_rejects_undeclared_or_unsupported_transition_modes(tmp_path):
    movies = _movies(tmp_path)
    with pytest.raises(AssemblyError, match="transition count"):
        measure_connected_boundaries(movies, transitions=["cut"])

    with pytest.raises(AssemblyError, match="unsupported production transition"):
        measure_connected_boundaries(
            movies,
            transitions=["continuous", "magic", "cut", "cut"],
            durations=[1.0] * 5,
        )


def test_requires_edit_durations_for_continuous_release_boundary(tmp_path):
    movies = _movies(tmp_path)
    with pytest.raises(AssemblyError, match="require approved edit durations"):
        measure_connected_boundaries(
            movies,
            transitions=["continuous", "cut", "cut", "cut"],
        )


def test_rejects_invalid_similarity_threshold_and_duration_count(tmp_path):
    movies = _movies(tmp_path)
    with pytest.raises(AssemblyError, match="threshold"):
        measure_connected_boundaries(
            movies,
            transitions=["cut"] * 4,
            minimum_similarity=1.01,
        )
    with pytest.raises(AssemblyError, match="duration count"):
        measure_connected_boundaries(
            movies,
            transitions=["cut"] * 4,
            durations=[1.0],
        )
