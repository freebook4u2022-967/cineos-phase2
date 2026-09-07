import pytest

from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_persistent_session import (
    PersistentGPUFoundationExecutor,
    PersistentGPUSessionError,
)


class _FailingRollbackRenderer:
    def __init__(self) -> None:
        self.shutdown_called = False

    def discard_quality_rejected_result(self, _receipt) -> None:
        raise RuntimeError("continuity cache could not be rolled back")

    def shutdown(self) -> None:
        self.shutdown_called = True


class _MalformedRollbackRenderer:
    discard_quality_rejected_result = "not-callable"

    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _StatelessLegacyRenderer:
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


def _opened_test_session(tmp_path, renderer):
    session = PersistentGPUFoundationExecutor(
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
    )
    # These unit tests isolate the rejection boundary without acquiring a real GPU.
    # A renderer assigned here represents an already-open persistent session.
    session._renderer = renderer
    return session


def test_failed_quality_rollback_poison_closes_persistent_session(tmp_path):
    renderer = _FailingRollbackRenderer()
    session = _opened_test_session(tmp_path, renderer)

    with pytest.raises(PersistentGPUSessionError, match="session was closed"):
        session.discard_quality_rejected_result(object())

    assert session.is_open is False
    assert renderer.shutdown_called is True


def test_noncallable_quality_rollback_poison_closes_persistent_session(tmp_path):
    renderer = _MalformedRollbackRenderer()
    session = _opened_test_session(tmp_path, renderer)

    with pytest.raises(PersistentGPUSessionError, match="session was closed"):
        session.discard_quality_rejected_result(object())

    assert session.is_open is False
    assert renderer.shutdown_called is True


def test_stateless_legacy_renderer_without_rollback_hook_remains_compatible(tmp_path):
    renderer = _StatelessLegacyRenderer()
    session = _opened_test_session(tmp_path, renderer)

    session.discard_quality_rejected_result(object())

    assert session.is_open is True
    session.close()
    assert renderer.shutdown_called is True
