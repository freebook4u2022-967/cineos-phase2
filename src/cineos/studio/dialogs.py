"""Reusable Studio dialogs."""

from PySide6.QtWidgets import QMessageBox, QWidget


def confirm_discard(parent: QWidget, project_name: str) -> bool:
    """Ask before discarding unsaved project changes."""
    answer = QMessageBox.warning(
        parent,
        "Unsaved changes",
        f"Save or discard changes to {project_name} before continuing?",
        QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return answer == QMessageBox.StandardButton.Discard


def show_error(parent: QWidget, title: str, error: BaseException | str) -> None:
    QMessageBox.critical(parent, title, str(error))
