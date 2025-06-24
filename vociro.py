"""Convenience shim so `import vociro` works regardless of install layout."""
from importlib.metadata import version as _v, PackageNotFoundError
from search_assistant import search as _search

try:
    __version__ = _v("vociro")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"

search = _search

__all__ = ["search", "__version__"] 