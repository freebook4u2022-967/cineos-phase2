"""Thread-safe, cooperatively cancellable Qt worker primitives."""

from collections.abc import Callable
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(float, str)
    result = Signal(object)
    error = Signal(str, object)
    cancelled = Signal()
    finished = Signal()


class BackgroundWorker(QRunnable):
    """Run a callable in QThreadPool; callables may accept cancel/progress hooks."""

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.function, self.args, self.kwargs = function, args, kwargs
        self.signals = WorkerSignals()
        self._cancel = Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            if self.cancelled:
                self.signals.cancelled.emit()
                return
            result = self.function(
                *self.args,
                cancel_event=self._cancel,
                progress=lambda value, message="": self.signals.progress.emit(
                    float(value), str(message)
                ),
                **self.kwargs,
            )
            if self.cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.result.emit(result)
        except Exception as error:
            self.signals.error.emit(str(error), error)
        finally:
            self.signals.finished.emit()
