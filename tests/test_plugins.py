from pathlib import Path

import pytest

from cineos.plugins import (
    DuplicatePluginError,
    Plugin,
    PluginCompatibilityError,
    PluginDependencyError,
    PluginLifecycleError,
    PluginLoader,
    PluginManager,
    PluginMetadata,
    PluginNotFoundError,
    PluginRegistry,
    version_matches,
)


class RecordingPlugin(Plugin):
    def __init__(
        self,
        name: str = "recording",
        version: str = "1.0.0",
        *,
        cineos_version: str = ">=0.1.0,<1.0.0",
        dependencies: dict[str, str] | None = None,
    ) -> None:
        self.metadata = PluginMetadata(
            name=name,
            version=version,
            cineos_version=cineos_version,
            dependencies=dependencies or {},
        )
        self.calls: list[str] = []

    def initialize(self) -> None:
        self.calls.append("initialize")

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def shutdown(self) -> None:
        self.calls.append("shutdown")


def test_metadata_is_validated_and_dependencies_are_immutable() -> None:
    dependencies = {"core": ">=1.2.0"}
    metadata = PluginMetadata("example", "2.0.1", dependencies=dependencies)
    dependencies["later"] = "1.0.0"

    assert dict(metadata.dependencies) == {"core": ">=1.2.0"}
    with pytest.raises(TypeError):
        metadata.dependencies["other"] = "1.0.0"  # type: ignore[index]
    with pytest.raises(ValueError, match="semantic version"):
        PluginMetadata("example", "not-a-version")
    with pytest.raises(ValueError, match="constraint"):
        PluginMetadata("example", "1.0.0", cineos_version="^1.0.0")


@pytest.mark.parametrize(
    ("version", "constraint", "expected"),
    [
        ("1.2.3", ">=1.0.0,<2.0.0", True),
        ("2.0.0", ">=1.0.0,<2.0.0", False),
        ("1.2.9", "~=1.2.3", True),
        ("1.3.0", "~=1.2.3", False),
        ("1.0.0", "!=1.0.0", False),
    ],
)
def test_version_matching(version: str, constraint: str, expected: bool) -> None:
    assert version_matches(version, constraint) is expected


def test_registry_registers_and_removes_plugins() -> None:
    registry = PluginRegistry()
    plugin = RecordingPlugin()
    registry.register(plugin)

    assert registry.get("recording") is plugin
    assert registry.names == ("recording",)
    with pytest.raises(DuplicatePluginError):
        registry.register(plugin)
    assert registry.unregister("recording") is plugin
    with pytest.raises(PluginNotFoundError):
        registry.get("recording")


def test_manager_runs_the_complete_lifecycle() -> None:
    plugin = RecordingPlugin()
    registry = PluginRegistry()
    manager = PluginManager(registry=registry)

    assert manager.load(plugin) is plugin
    assert manager.registry is registry
    assert not manager.is_enabled("recording")
    manager.enable("recording")
    manager.enable("recording")  # Idempotent.
    assert manager.is_enabled("recording")
    manager.disable("recording")
    manager.disable("recording")  # Idempotent.
    assert manager.unload("recording") is plugin

    assert plugin.calls == ["initialize", "start", "stop", "shutdown"]
    assert len(manager.registry) == 0


def test_manager_rejects_incompatible_plugin() -> None:
    manager = PluginManager(cineos_version="2.0.0")
    with pytest.raises(PluginCompatibilityError, match="running 2.0.0"):
        manager.load(RecordingPlugin(cineos_version="<2.0.0"))


def test_manager_enforces_dependency_version_and_lifecycle_order() -> None:
    manager = PluginManager()
    dependent = RecordingPlugin(
        "dependent", dependencies={"foundation": ">=2.0.0,<3.0.0"}
    )

    with pytest.raises(PluginDependencyError, match="missing"):
        manager.load(dependent)
    manager.load(RecordingPlugin("foundation", "1.0.0"))
    with pytest.raises(PluginDependencyError, match="loaded version"):
        manager.load(dependent)
    manager.unload("foundation")

    manager.load(RecordingPlugin("foundation", "2.1.0"))
    manager.load(dependent)
    with pytest.raises(PluginDependencyError, match="enabled dependencies"):
        manager.enable("dependent")
    manager.enable("foundation")
    manager.enable("dependent")
    with pytest.raises(PluginDependencyError, match="depend on it"):
        manager.disable("foundation")
    with pytest.raises(PluginDependencyError, match="depend on it"):
        manager.unload("foundation")
    manager.disable("dependent")
    manager.unload("dependent")
    manager.disable("foundation")
    manager.unload("foundation")


def test_initialize_failure_rolls_back_registration() -> None:
    class BrokenPlugin(RecordingPlugin):
        def initialize(self) -> None:
            raise RuntimeError("broken")

    manager = PluginManager()
    with pytest.raises(PluginLifecycleError, match="initialize"):
        manager.load(BrokenPlugin())
    assert len(manager.registry) == 0


def test_loader_discovers_and_loads_file_plugins(tmp_path: Path) -> None:
    module = tmp_path / "sample.py"
    module.write_text("""\
from cineos.plugins import Plugin, PluginMetadata

class SamplePlugin(Plugin):
    metadata = PluginMetadata(name="sample", version="1.0.0")
""")
    loader = PluginLoader(entry_point_group="cineos.tests.no_entry_points")

    discovered = loader.discover([tmp_path])

    assert discovered == (module,)
    assert loader.load(discovered[0]).metadata.name == "sample"


def test_loader_accepts_plugin_class() -> None:
    class ClassPlugin(Plugin):
        metadata = PluginMetadata("class-plugin", "1.0.0")

    assert PluginLoader().load(ClassPlugin).metadata.name == "class-plugin"
