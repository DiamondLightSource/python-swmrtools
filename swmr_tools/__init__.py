from .keyfollower import KeyFollower, RowKeyFollower
from .datasource import DataSource
from . import utils
import importlib.metadata

__all__ = ["KeyFollower", "RowKeyFollower", "DataSource", "utils"]

try:
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
