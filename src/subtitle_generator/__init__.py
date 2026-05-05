"""Book subtitle generator from LOC MARC records."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("subtitle-generator")
except PackageNotFoundError:
    __version__ = "0.7.0"
