"""Studio process entry point."""

import sys

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def create_application(argv: list[str] | None = None) -> QApplication:
    """Create or return the Qt application for embedding and tests."""
    QCoreApplication.setOrganizationName("CINEOS")
    QCoreApplication.setApplicationName("StudioAlpha")
    return QApplication.instance() or QApplication(
        argv if argv is not None else sys.argv
    )


def main() -> int:
    """Launch CINEOS Studio Alpha."""
    app = create_application()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
