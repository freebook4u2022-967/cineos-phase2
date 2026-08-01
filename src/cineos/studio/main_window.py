"""CINEOS Studio Alpha main desktop workspace."""

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolBar,
)

from .asset_panel import AssetPanel
from .controller import StudioController
from .dialogs import confirm_discard, show_error
from .export_panel import ExportPanel
from .project_panel import ProjectPanel
from .render_queue_panel import RenderQueuePanel
from .renderer_panel import RendererPanel
from .review_panel import ReviewPanel
from .scene_panel import ScenePanel
from .settings import StudioSettings
from .timeline_panel import TimelinePanel
from .workers import BackgroundWorker


class MainWindow(QMainWindow):
    """Dockable editor shell for the existing end-to-end CINEOS workflow."""

    def __init__(self, controller: StudioController | None = None) -> None:
        super().__init__()
        self.settings = StudioSettings()
        self.controller = controller or StudioController(settings=self.settings)
        self.thread_pool = QThreadPool.globalInstance()
        self._workers: set[BackgroundWorker] = set()
        self.setWindowTitle("CINEOS Studio Alpha")
        self.resize(1440, 900)
        self._tabs = QTabWidget()
        self._panels = [
            ProjectPanel(),
            AssetPanel(),
            ScenePanel(),
            TimelinePanel(),
            RendererPanel(),
            RenderQueuePanel(),
            ReviewPanel(),
            ExportPanel(),
        ]
        for panel in self._panels:
            self._tabs.addTab(panel, panel.findChild(QLabel, "panelHeading").text())
        self.setCentralWidget(self._tabs)
        self._status = QLabel("No project open")
        self.statusBar().addPermanentWidget(self._status)
        self._create_actions()
        self._create_build_dock()
        self.controller.subscribe(self.refresh)
        geometry, state = self.settings.restore_layout()
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

    def _create_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        actions = [
            ("&New project", self.new_project),
            ("&Open project…", self.open_project),
            ("&Save", self.save_project),
            ("Save &As…", self.save_project_as),
        ]
        for title, callback in actions:
            action = QAction(title, self)
            action.triggered.connect(callback)
            file_menu.addAction(action)
        recent = file_menu.addMenu("Recent projects")
        for path in self.settings.recent_projects():
            action = recent.addAction(path)
            action.triggered.connect(
                lambda checked=False, value=path: self._open(value)
            )
        toolbar = QToolBar("Film build", self)
        self.addToolBar(toolbar)
        for title, callback in (
            ("Validate", self.validate_project),
            ("Compile FilmPackage", self.compile_project),
            ("Build conditioning", self.not_configured),
            ("Dry-run film", self.not_configured),
            ("Render selected", self.not_configured),
            ("Build film", self.not_configured),
            ("Resume", self.not_configured),
            ("Cancel", self.cancel_jobs),
            ("Export", self.not_configured),
        ):
            action = toolbar.addAction(title)
            action.triggered.connect(callback)

    def _create_build_dock(self) -> None:
        dock = QDockWidget("Runtime events", self)
        dock.setObjectName("runtimeEventsDock")
        dock.setWidget(
            QLabel("Runtime events and background job progress appear here.")
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def refresh(self) -> None:
        marker = " *" if self.controller.state.dirty else ""
        self.setWindowTitle(
            f"{self.controller.state.display_name}{marker} — CINEOS Studio Alpha"
        )
        errors = self.controller.state.validation_errors
        self._status.setText(
            "Valid"
            if self.controller.state.project and not errors
            else f"{len(errors)} validation issue(s)"
        )
        for panel in self._panels:
            panel.refresh(self.controller)

    def _can_replace(self) -> bool:
        state = self.controller.state
        return not state.dirty or confirm_discard(self, state.display_name)

    def new_project(self) -> None:
        if self._can_replace():
            self.controller.new_project()

    def open_project(self) -> None:
        if not self._can_replace():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CINEOS project", "", "CINEOS project (*.cineos.json *.json)"
        )
        if path:
            self._open(path)

    def _open(self, path: str) -> None:
        try:
            self.controller.open(path)
        except Exception as error:
            show_error(self, "Could not open project", error)

    def save_project(self) -> None:
        if self.controller.state.project_path:
            self._save(self.controller.state.project_path)
        else:
            self.save_project_as()

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CINEOS project",
            "project.cineos.json",
            "CINEOS project (*.cineos.json)",
        )
        if path:
            self._save(Path(path))

    def _save(self, path: Path) -> None:
        try:
            self.controller.save(path)
        except Exception as error:
            show_error(self, "Could not save project", error)

    def _background(self, operation: object) -> None:
        def invoke(*, cancel_event: object, progress: object) -> object:
            return operation()  # type: ignore[operator]

        worker = BackgroundWorker(invoke)
        self._workers.add(worker)
        worker.signals.error.connect(
            lambda message, error: show_error(self, "Operation failed", error)
        )
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self.thread_pool.start(worker)

    def validate_project(self) -> None:
        if self.controller.state.project:
            self._background(self.controller.validate)

    def compile_project(self) -> None:
        if self.controller.state.project:
            self._background(self.controller.compile)

    def cancel_jobs(self) -> None:
        for worker in tuple(self._workers):
            worker.cancel()

    def not_configured(self) -> None:
        QMessageBox.information(
            self,
            "Studio Alpha",
            "Configure a renderer and output directory for this operation.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._can_replace():
            event.ignore()
            return
        self.cancel_jobs()
        self.settings.save_layout(self.saveGeometry(), self.saveState())
        event.accept()
