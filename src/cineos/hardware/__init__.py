"""Cross-platform, dependency-optional hardware diagnostics."""

from .models import GPUInfo, HardwareReport, Recommendation
from .probe import probe
from .report import to_json, to_text

__all__ = ["GPUInfo", "HardwareReport", "Recommendation", "probe", "to_json", "to_text"]
