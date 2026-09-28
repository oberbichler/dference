"""dference - compare two DataFrames by key and explore the differences.

Quick start::

    import dference

    result = dference.compare(left, right, key="id")
    result.summary  # counts per status
    result.mismatches()  # one row per differing cell

    # interactive, in marimo or Jupyter
    widget = dference.DataFrameDiff(left, right, key="id")
"""

from importlib.metadata import PackageNotFoundError, version

from ._compare import ColumnInfo, DiffResult, Status, Summary, compare
from ._widget import DataFrameDiff

try:
    __version__ = version("dference")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "ColumnInfo",
    "DataFrameDiff",
    "DiffResult",
    "Status",
    "Summary",
    "__version__",
    "compare",
]
