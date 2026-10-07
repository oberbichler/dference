"""The :class:`DataFrameDiff` anywidget.

State that the frontend needs once (column metadata, counts, names) is synced
as traitlets. Rows are fetched on demand through anywidget custom messages:

* frontend → Python ``{"type": "query", "req": n, ...}``
  → Python → frontend ``{"type": "page", "req": n, "rows": [...], ...}``
* frontend → Python ``{"type": "export", "req": n, ...}``
  → Python → frontend ``{"type": "export", "req": n, "filename": ...}`` + CSV buffer

Only ``selected_ids`` changes as the user interacts, so in marimo just the
cells that read the selection re-run - paging, sorting and filtering do not
trigger any re-execution.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anywidget
import traitlets

from ._compare import STATUS_ORDER, DiffResult, compare
from ._query import Query, ViewEngine

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import polars as pl

__all__ = ["DataFrameDiff"]

_STATIC = Path(__file__).parent / "static"
_log = logging.getLogger(__name__)


class DataFrameDiff(anywidget.AnyWidget):
    """Interactive comparison of two DataFrames.

    Example:
        >>> import marimo as mo
        >>> from dference import DataFrameDiff
        >>> view = mo.ui.anywidget(DataFrameDiff(crm, erp, key="customer_id"))  # doctest: +SKIP
        >>> view.value["selected_ids"]  # reactive selection  # doctest: +SKIP

    Args:
        left: Left frame (polars, pandas, pyarrow, …).
        right: Right frame.
        key: Key column(s); the combination must be unique on each side.
        left_name: Display name of the left side.
        right_name: Display name of the right side.
        left_short: Marker for left values; defaults to the first letter of
            ``left_name`` (``"L"`` if both names start with the same letter).
        right_short: Marker for right values, analogous to ``left_short``.
        ignore_columns: Columns to leave out of the comparison.
        strict: Require identical dtypes on both sides; ``False`` aligns
            lossless differences (see :func:`~dference.compare`).
        page_size: Initial rows per page (the user can change it).
        text_diff: Compare differing texts like ``git diff`` (a unified diff
            with the differing words highlighted) in the detail view and the
            full-text flyover. This is the initial state of the *Text diff*
            toggle (the user can change it); ``False`` shows both sides next
            to each other instead.
        **kwargs: Passed on to :class:`anywidget.AnyWidget`.
    """

    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    columns = traitlets.List(traitlets.Dict()).tag(sync=True)
    summary = traitlets.Dict().tag(sync=True)
    left_name = traitlets.Unicode("left").tag(sync=True)
    right_name = traitlets.Unicode("right").tag(sync=True)
    left_short = traitlets.Unicode("L").tag(sync=True)
    right_short = traitlets.Unicode("R").tag(sync=True)
    page_size = traitlets.Int(10).tag(sync=True)
    text_diff = traitlets.Bool(True).tag(sync=True)
    #: Row ids checked in the widget (see :meth:`selected_frame`).
    selected_ids = traitlets.List(traitlets.Int()).tag(sync=True)

    def __init__(
        self,
        left: Any,
        right: Any,
        key: str | Sequence[str],
        *,
        left_name: str = "left",
        right_name: str = "right",
        left_short: str | None = None,
        right_short: str | None = None,
        ignore_columns: Iterable[str] = (),
        strict: bool = True,
        page_size: int = 10,
        text_diff: bool = True,
        **kwargs: Any,
    ) -> None:
        result = compare(
            left,
            right,
            key,
            left_name=left_name,
            right_name=right_name,
            ignore_columns=ignore_columns,
            strict=strict,
        )
        self._init_from_result(
            result,
            left_short=left_short,
            right_short=right_short,
            page_size=page_size,
            text_diff=text_diff,
            kwargs=kwargs,
        )

    @classmethod
    def from_result(
        cls,
        result: DiffResult,
        *,
        left_short: str | None = None,
        right_short: str | None = None,
        page_size: int = 10,
        text_diff: bool = True,
        **kwargs: Any,
    ) -> DataFrameDiff:
        """Create a widget for an existing :func:`~dference.compare` result."""
        self = cls.__new__(cls)
        self._init_from_result(
            result,
            left_short=left_short,
            right_short=right_short,
            page_size=page_size,
            text_diff=text_diff,
            kwargs=kwargs,
        )
        return self

    def _init_from_result(
        self,
        result: DiffResult,
        *,
        left_short: str | None,
        right_short: str | None,
        page_size: int,
        text_diff: bool,
        kwargs: dict[str, Any],
    ) -> None:
        self._result = result
        self._engine = ViewEngine(result)
        self._last_query = Query()
        ls, rs = short_names(result.left_name, result.right_name, left_short, right_short)
        super().__init__(
            columns=[_column_payload(c) for c in result.columns],
            summary=_summary_payload(result),
            left_name=result.left_name,
            right_name=result.right_name,
            left_short=ls,
            right_short=rs,
            page_size=page_size,
            text_diff=text_diff,
            **kwargs,
        )
        self.on_msg(self._handle_message)

    # ---- Python API -------------------------------------------------------

    @property
    def result(self) -> DiffResult:
        """The underlying :class:`~dference.DiffResult`."""
        return self._result

    def selected_frame(self) -> pl.DataFrame:
        """Rows checked in the widget, as a wide result frame."""
        return self._result.frame(rows=sorted(set(self.selected_ids)))

    def view_frame(self) -> pl.DataFrame:
        """Rows matching the filters currently set in the widget, in view order.

        Note:
            This reflects the last query the frontend sent; it is not reactive
            in marimo (filters do not re-run cells).
        """
        return self._engine.frame(self._last_query)

    # ---- messaging ----------------------------------------------------------

    def _handle_message(self, _widget: Any, content: Any, _buffers: Any) -> None:
        """Dispatch a custom message from the frontend; never raises."""
        if not isinstance(content, Mapping):
            return
        req = content.get("req")
        try:
            match content.get("type"):
                case "query":
                    query = Query.from_message(content, self._result.columns)
                    self._last_query = query
                    page = self._engine.page(
                        query, content.get("page", 0), content.get("page_size", self.page_size)
                    )
                    self.send({"type": "page", "req": req, **page})
                case "export":
                    query = Query.from_message(content, self._result.columns)
                    data = self._engine.export_csv(query)
                    self.send(
                        {"type": "export", "req": req, "filename": "dataframe-diff.csv"},
                        buffers=[data],
                    )
                case other:
                    _log.debug("Ignoring unknown message type %r", other)
        except Exception as exc:
            _log.exception("dference: failed to handle %r message", content.get("type"))
            self.send({"type": "error", "req": req, "message": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def short_names(
    left_name: str, right_name: str, left_short: str | None, right_short: str | None
) -> tuple[str, str]:
    """Side markers: first letter of each name unless overridden.

    Falls back to ``"L"``/``"R"`` if a name is empty or both would get the same
    letter (e.g. ``"Prod"`` vs. ``"Preview"``).

    Example:
        >>> short_names("CRM", "ERP", None, None)
        ('C', 'E')
        >>> short_names("Prod", "Preview", None, None)
        ('L', 'R')
    """

    def first(name: str) -> str:
        stripped = name.strip()
        return stripped[0].upper() if stripped else ""

    ls = left_short if left_short is not None else first(left_name)
    rs = right_short if right_short is not None else first(right_name)
    clash = ls == rs
    if left_short is None and (not ls or clash):
        ls = "L"
    if right_short is None and (not rs or clash):
        rs = "R"
    return ls, rs


def _column_payload(c: Any) -> dict[str, Any]:
    return {
        "name": c.name,
        "kind": c.kind,
        "dtype": str(c.dtype),
        "numeric": c.dtype.is_numeric(),
        "ftype": c.filter_type,
        "mismatches": c.mismatches,
    }


def _summary_payload(result: DiffResult) -> dict[str, Any]:
    s = result.summary
    return {
        **{st.value: getattr(s, st.name.lower()) for st in STATUS_ORDER},
        "total": s.total,
        "found": s.found,
        "left_rows": s.left_rows,
        "right_rows": s.right_rows,
        "keys": list(result.keys),
        "compared": list(result.compared),
        "left_only_cols": [c.name for c in result.columns if c.kind == "left_only"],
        "right_only_cols": [c.name for c in result.columns if c.kind == "right_only"],
        "ignored": list(result.ignored),
    }
