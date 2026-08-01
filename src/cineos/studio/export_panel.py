"""Export Studio panel."""

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget


class ExportPanel(QWidget):
    """Present export state without implementing domain behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        heading = QLabel("Export")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        self.items = QListWidget()
        self.items.addItems(
            [
                "Final MP4",
                "Build report",
                "FilmPackage",
                "Validation reports",
                "Recovery history",
                "Checksums",
            ]
        )
        layout.addWidget(self.items)

    def refresh(self, controller: object) -> None:
        """Refresh hook invoked by the main workspace."""
