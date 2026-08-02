"""Portable release diagnostics."""

import platform


def diagnose() -> dict[str, object]:
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python": platform.python_version(),
        "native_installer_support": False,
    }
