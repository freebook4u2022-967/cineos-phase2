"""Render queue Studio panel."""

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget


class RenderQueuePanel(QWidget):
    """Present render queue state without implementing domain behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        heading = QLabel("Render queue")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        self.items = QListWidget()
        self.items.addItems(
            [
                "Job ID",
                "Shot ID",
                "State",
                "Progress",
                "Attempts",
                "Renderer",
                "Start time",
                "Duration",
            ]
        )
        layout.addWidget(self.items)

    def refresh(self, controller: object) -> None:
        """Refresh hook invoked by the main workspace."""
