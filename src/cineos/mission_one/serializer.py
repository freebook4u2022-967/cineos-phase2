"""Compatibility serializer (canonical implementation lives in renderer package)."""

from cineos.renderers.colab.serializer import dump_json, load_json

__all__ = ["dump_json", "load_json"]
