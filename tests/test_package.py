import cineos


def test_package_exposes_version() -> None:
    assert cineos.__version__ == "0.1.0"
