"""Scenes & shots Studio panel."""

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget


class ScenePanel(QWidget):
    """Present scenes & shots state without implementing domain behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        heading = QLabel("Scenes & shots")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        self.items = QListWidget()
        self.items.addItems(
            [
                "Scene list",
                "Shot list",
                "Shot title",
                "Action",
                "Dialogue",
                "Camera / lens",
                "Movement / lighting",
                "Duration",
                "Continuity links",
            ]
        )
        layout.addWidget(self.items)

    def refresh(self, controller: object) -> None:
        """Refresh hook invoked by the main workspace."""
