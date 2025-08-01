from importlib.metadata import version as _v, PackageNotFoundError

try:
    __version__ = _v("vociro")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"

from .search_assistant import search  # noqa: F401 – re-export

__all__ = ["search", "__version__"] 