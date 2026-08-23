from cineos.native_image.image_quality import ImageQualityInspector, ImageQualityPolicy


def _ppm(width, height, rgb):
    return f"P6\n{width} {height}\n255\n".encode() + bytes(rgb) * (width * height)


def test_quality_rejects_low_resolution(tmp_path):
    path = tmp_path / "small.ppm"
    path.write_bytes(_ppm(8, 8, (120, 120, 120)))
    assessment = ImageQualityInspector(ImageQualityPolicy(minimum_width=16, minimum_height=16)).assess(path)
    assert assessment.approved is False
    assert "resolution below minimum" in assessment.reasons


def test_quality_rejects_extreme_exposure(tmp_path):
    dark = tmp_path / "dark.ppm"
    dark.write_bytes(_ppm(16, 16, (0, 0, 0)))
    assessment = ImageQualityInspector(ImageQualityPolicy(minimum_width=8, minimum_height=8, minimum_blur_score=0.0)).assess(dark)
    assert "underexposed" in assessment.reasons[0]


def test_quality_report_finds_near_duplicates(tmp_path):
    first = tmp_path / "a.ppm"
    second = tmp_path / "b.ppm"
    first.write_bytes(_ppm(16, 16, (120, 100, 80)))
    second.write_bytes(_ppm(16, 16, (120, 100, 80)))
    report = ImageQualityInspector(ImageQualityPolicy(minimum_width=8, minimum_height=8, minimum_blur_score=0.0)).report((first, second))
    assert report.near_duplicate_pairs == ((str(first), str(second)),)


def test_quality_report_can_be_saved(tmp_path):
    path = tmp_path / "frame.ppm"
    path.write_bytes(_ppm(16, 16, (120, 100, 80)))
    inspector = ImageQualityInspector(ImageQualityPolicy(minimum_width=8, minimum_height=8, minimum_blur_score=0.0))
    report = inspector.report((path,))
    output = report.save(tmp_path / "quality.json")
    assert output.exists()
    assert "approved_count" in output.read_text()
