"""Assets Studio panel."""

from PySide6.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget


class AssetPanel(QWidget):
    """Present assets state without implementing domain behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        heading = QLabel("Assets")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        self.items = QListWidget()
        self.items.addItems(
            [
                "Search / tags",
                "Characters",
                "Environments",
                "Wardrobes",
                "Props",
                "Vehicles",
                "Storyboards",
                "Reference images",
                "Approval status",
            ]
        )
        layout.addWidget(self.items)

    def refresh(self, controller: object) -> None:
        """Refresh hook invoked by the main workspace."""
