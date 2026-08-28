"""Cross-platform, dependency-optional hardware diagnostics."""

from .models import GPUInfo, HardwareReport, Recommendation
from .probe import probe
from .report import to_json, to_text
from .torch_readiness import TorchDeviceReadiness, probe_torch_device

__all__ = [
    "GPUInfo",
    "HardwareReport",
    "Recommendation",
    "TorchDeviceReadiness",
    "probe",
    "probe_torch_device",
    "to_json",
    "to_text",
]
