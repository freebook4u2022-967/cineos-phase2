"""Renderer Studio panel."""

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget


class RendererPanel(QWidget):
    """Present renderer state without implementing domain behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        heading = QLabel("Renderer")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        self.items = QListWidget()
        self.items.addItems(
            [
                "Selected renderer",
                "Available plugins",
                "Capabilities",
                "Environment validation",
                "Model status",
                "Hardware compatibility",
                "Dry-run",
            ]
        )
        layout.addWidget(self.items)

    def refresh(self, controller: object) -> None:
        """Refresh hook invoked by the main workspace."""
