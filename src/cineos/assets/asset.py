"""Backward-compatible imports for the canonical asset model."""

from .base import Asset, AssetType, AssetVersion
from .reference import ReferenceImage

__all__ = ["Asset", "AssetType", "AssetVersion", "ReferenceImage"]
