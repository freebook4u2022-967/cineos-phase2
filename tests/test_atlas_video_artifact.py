from pathlib import Path

import pytest

from cineos.atlas.video_artifact import VideoArtifactError, inspect_mp4_container


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + box_type + payload


def _write(path: Path, *boxes: bytes) -> Path:
    path.write_bytes(b"".join(boxes))
    return path


def test_inspect_mp4_accepts_standard_movie_container(tmp_path):
    artifact = _write(
        tmp_path / "standard.mp4",
        _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2"),
        _box(b"free", b"padding"),
        _box(b"mdat", b"frames"),
        _box(b"moov"),
    )

    evidence = inspect_mp4_container(artifact)

    assert evidence.has_ftyp is True
    assert evidence.has_movie_metadata is True
    assert evidence.has_media_data is True
    assert evidence.media_payload_bytes == len(b"frames")
    assert evidence.box_types == ("ftyp", "free", "mdat", "moov")


def test_inspect_mp4_accepts_fragmented_movie_container(tmp_path):
    artifact = _write(
        tmp_path / "fragmented.mp4",
        _box(b"ftyp", b"iso6\x00\x00\x00\x01iso6mp41"),
        _box(b"moof", b"fragment-metadata"),
        _box(b"mdat", b"fragment-media"),
    )

    evidence = inspect_mp4_container(artifact)

    assert evidence.has_movie_metadata is True
    assert evidence.media_payload_bytes == len(b"fragment-media")
    assert "moof" in evidence.box_types


def test_inspect_mp4_sums_multiple_media_payloads(tmp_path):
    artifact = _write(
        tmp_path / "multi-mdat.mp4",
        _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2"),
        _box(b"moov", b"metadata"),
        _box(b"mdat", b"first"),
        _box(b"mdat", b"second"),
    )

    evidence = inspect_mp4_container(artifact)

    assert evidence.media_payload_bytes == len(b"firstsecond")


def test_inspect_mp4_rejects_extension_only_fake_video(tmp_path):
    artifact = tmp_path / "fake.mp4"
    artifact.write_bytes(b"arbitrary bytes that are not ISO BMFF video")

    with pytest.raises(VideoArtifactError):
        inspect_mp4_container(artifact)


def test_inspect_mp4_rejects_box_exceeding_artifact_bounds(tmp_path):
    artifact = tmp_path / "truncated.mp4"
    artifact.write_bytes((128).to_bytes(4, "big") + b"ftyp" + b"short")

    with pytest.raises(VideoArtifactError, match="exceeds artifact bounds"):
        inspect_mp4_container(artifact)


def test_inspect_mp4_requires_media_data(tmp_path):
    artifact = _write(
        tmp_path / "metadata-only.mp4",
        _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2"),
        _box(b"moov", b"metadata"),
    )

    with pytest.raises(VideoArtifactError, match="missing MP4 media data"):
        inspect_mp4_container(artifact)


def test_inspect_mp4_rejects_empty_media_payload(tmp_path):
    artifact = _write(
        tmp_path / "empty-media.mp4",
        _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2"),
        _box(b"moov", b"metadata"),
        _box(b"mdat"),
    )

    with pytest.raises(VideoArtifactError, match="empty MP4 media payload"):
        inspect_mp4_container(artifact)
