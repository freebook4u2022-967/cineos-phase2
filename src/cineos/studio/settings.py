"""Non-sensitive Studio preference persistence."""

from PySide6.QtCore import QByteArray, QSettings


class StudioSettings:
    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings("CINEOS", "StudioAlpha")

    def recent_projects(self) -> list[str]:
        value = self._settings.value("recentProjects", [])
        return [str(item) for item in value] if isinstance(value, list) else []

    def add_recent_project(self, path: str) -> None:
        paths = [path, *(p for p in self.recent_projects() if p != path)][:10]
        self._settings.setValue("recentProjects", paths)

    def renderer(self) -> str:
        return str(self._settings.value("renderer", ""))

    def set_renderer(self, renderer: str) -> None:
        self._settings.setValue("renderer", renderer)

    def save_layout(self, geometry: QByteArray, state: QByteArray) -> None:
        self._settings.setValue("windowGeometry", geometry)
        self._settings.setValue("windowState", state)

    def restore_layout(self) -> tuple[QByteArray, QByteArray]:
        geometry = self._settings.value("windowGeometry", QByteArray())
        state = self._settings.value("windowState", QByteArray())
        return geometry, state
