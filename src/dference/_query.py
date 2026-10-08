"""Server-side view engine: filtering, sorting and paging of a :class:`DiffResult`.

The widget never receives the full table. Instead it sends a *query* (status
selection, search text, column filters, sort, page) and gets back only the rows
of the requested page. All work happens in polars on the joined frame:

1. the query is turned into one boolean predicate and one sort key,
2. the resulting ordered row ids are cached (paging through the same view is
   a cheap slice), and
3. only the ``page_size`` rows of the page are gathered and serialised.

Filter semantics for compared columns: a row matches if the **left or the
right** value matches - but only sides that actually exist in that row are
considered, so a missing side never matches ``is null``.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

import polars as pl

from ._compare import (
    COUNT_LEFT,
    COUNT_RIGHT,
    ROW,
    STATUS,
    STATUS_ORDER,
    ColumnInfo,
    DiffResult,
    Status,
    as_text,
    dcol,
    duplicate_key,
    lcol,
    rcol,
)

if TYPE_CHECKING:
    from polars.datatypes import DataType

__all__ = ["ColumnFilter", "Query", "ViewEngine"]

MAX_PAGE_SIZE: Final = 1000
_CACHE_SIZE: Final = 8

#: Operators per filter type, as offered by the frontend.
OPERATORS: Final[Mapping[str, frozenset[str]]] = {
    "string": frozenset(
        {
            "contains",
            "not_contains",
            "equals",
            "starts",
            "ends",
            "is_empty",
            "not_empty",
            "is_null",
            "not_null",
        }
    ),
    "number": frozenset({"between", "eq", "ne", "is_null", "not_null"}),
    "datetime": frozenset({"between", "is_null", "not_null"}),
    "boolean": frozenset({"true", "false", "is_null"}),
}


# --------------------------------------------------------------------------- #
# Query model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ColumnFilter:
    """Filter on one column (by index into ``DiffResult.columns``)."""

    column: int
    op: str
    a: str = ""
    b: str = ""


@dataclass(frozen=True, slots=True)
class Query:
    """Everything that determines which rows are shown and in which order.

    Hashable, so identical queries share one cached view.
    """

    statuses: frozenset[Status] = field(default_factory=lambda: frozenset(STATUS_ORDER))
    #: only rows whose key is not unique on a side
    duplicates_only: bool = False
    search: str = ""
    diff_column: int | None = None
    #: with ``diff_column``: rows *equal* in that column instead of differing
    diff_equal: bool = False
    filters: tuple[ColumnFilter, ...] = ()
    sort_column: int | None = None
    sort_by_status: bool = False
    descending: bool = False

    @classmethod
    def from_message(cls, msg: Mapping[str, Any], columns: tuple[ColumnInfo, ...]) -> Query:
        """Parse and validate a query sent by the frontend.

        Invalid parts are dropped rather than raising, so a stale or malformed
        message can never break the widget.
        """
        n = len(columns)

        def col_index(value: Any) -> int | None:
            return value if isinstance(value, int) and 0 <= value < n else None

        valid = {s.value for s in Status}
        statuses = frozenset(Status(s) for s in msg.get("statuses", STATUS_ORDER) if s in valid)
        diff_column = col_index(msg.get("diff_column"))
        if diff_column is not None and columns[diff_column].kind != "compared":
            diff_column = None

        filters: list[ColumnFilter] = []
        for raw in msg.get("filters", ()):
            if not isinstance(raw, Mapping):
                continue
            i = col_index(raw.get("column"))
            if i is None or raw.get("op") not in OPERATORS[columns[i].filter_type]:
                continue
            filters.append(
                ColumnFilter(i, str(raw["op"]), str(raw.get("a", "")), str(raw.get("b", "")))
            )

        sort = msg.get("sort") or {}
        sort_col = sort.get("column") if isinstance(sort, Mapping) else None
        return cls(
            statuses=statuses,
            duplicates_only=msg.get("duplicates_only") is True,
            search=str(msg.get("search", ""))[:500],
            diff_column=diff_column,
            diff_equal=diff_column is not None and msg.get("diff_equal") is True,
            filters=tuple(sorted(filters, key=lambda f: f.column)),
            sort_column=col_index(sort_col),
            sort_by_status=sort_col == "status",
            descending=bool(isinstance(sort, Mapping) and sort.get("descending")),
        )


@dataclass(frozen=True, slots=True)
class View:
    """Ordered row ids of a query plus the columns that differ within it."""

    ids: pl.Series
    diff_columns: tuple[int, ...]

    @property
    def size(self) -> int:
        """Number of rows in the view."""
        return self.ids.len()


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class ViewEngine:
    """Answers queries against one :class:`DiffResult`."""

    def __init__(self, result: DiffResult) -> None:
        self.result = result
        self._cache: OrderedDict[Query, View] = OrderedDict()
        # Row presence per side: a side exists unless the row is missing there.
        self._present_l = pl.col(STATUS) != Status.MISSING_LEFT.value
        self._present_r = pl.col(STATUS) != Status.MISSING_RIGHT.value
        self._compared_idx = tuple(i for i, c in enumerate(result.columns) if c.kind == "compared")

    # ---- public API ---------------------------------------------------------

    def view(self, query: Query) -> View:
        """Return (and cache) the ordered row ids matching ``query``."""
        if (hit := self._cache.get(query)) is not None:
            self._cache.move_to_end(query)
            return hit

        lazy = self.result.data.lazy().filter(self._predicate(query))
        ordered = self._sorted(lazy, query).select(ROW)
        flags = lazy.select(
            pl.col(dcol(self.result.columns[i].name)).any() for i in self._compared_idx
        )
        ids_df, flags_df = pl.collect_all([ordered, flags])

        flag_row = flags_df.row(0) if flags_df.width else ()
        diff_columns = tuple(i for i, hit in zip(self._compared_idx, flag_row, strict=True) if hit)
        view = View(ids=ids_df.get_column(ROW), diff_columns=diff_columns)

        self._cache[query] = view
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return view

    def page(self, query: Query, page: int, page_size: int) -> dict[str, Any]:
        """Serialise one page of ``query`` for the frontend."""
        page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
        view = self.view(query)
        pages = max(1, math.ceil(view.size / page_size))
        page = max(0, min(int(page), pages - 1))
        ids = view.ids.slice(page * page_size, page_size)
        return {
            "filtered": view.size,
            "page": page,
            "page_size": page_size,
            "diff_columns": list(view.diff_columns),
            "rows": self.serialise(ids),
        }

    def frame(self, query: Query) -> pl.DataFrame:
        """Wide result frame of the rows in ``query``, in view order."""
        return self.result.frame(rows=self.view(query).ids)

    def export_csv(self, query: Query) -> bytes:
        """CSV (UTF-8) of the rows in ``query``; nested values are stringified."""
        df = self.frame(query)
        nested = [name for name, dtype in df.schema.items() if dtype.is_nested()]
        if nested:
            df = df.with_columns(as_text(pl.col(n), df.schema[n]) for n in nested)
        return df.write_csv().encode()

    # ---- serialisation ----------------------------------------------------

    def serialise(self, ids: pl.Series) -> list[dict[str, Any]]:
        """Turn the given row ids into the compact row format of the frontend.

        Each row is ``{"id", "s", "d", "l", "r"}``: status, indices of differing
        columns and the left/right values aligned with ``columns`` (``None``
        for a side that does not exist in that row). A row whose key is not
        unique also has ``"k": [left count, right count]``.
        """
        if ids.is_empty():
            return []
        cols = self.result.columns
        # Column name in `data` holding the left/right value of column i.
        lsrc = [
            c.name if c.kind == "key" else lcol(c.name) if c.kind != "right_only" else None
            for c in cols
        ]
        rsrc = [
            c.name if c.kind == "key" else rcol(c.name) if c.kind != "left_only" else None
            for c in cols
        ]
        needed = {ROW, STATUS} | {n for n in (*lsrc, *rsrc) if n}
        needed |= {dcol(cols[i].name) for i in self._compared_idx}
        dups = self.result.has_duplicates
        if dups:
            needed |= {COUNT_LEFT, COUNT_RIGHT}
        page = self.result.data.select(pl.col(sorted(needed)).gather(ids)).to_dicts()

        rows: list[dict[str, Any]] = []
        for rec in page:
            status = rec[STATUS]
            has_l = status != Status.MISSING_LEFT.value
            has_r = status != Status.MISSING_RIGHT.value
            row = {
                "id": rec[ROW],
                "s": status,
                "d": [i for i in self._compared_idx if rec[dcol(cols[i].name)]],
                "l": [_json(rec[n]) if n else None for n in lsrc] if has_l else None,
                "r": [_json(rec[n]) if n else None for n in rsrc] if has_r else None,
            }
            if dups and (rec[COUNT_LEFT] > 1 or rec[COUNT_RIGHT] > 1):
                row["k"] = [rec[COUNT_LEFT], rec[COUNT_RIGHT]]
            rows.append(row)
        return rows

    # ---- expression building ------------------------------------------------

    def _predicate(self, q: Query) -> pl.Expr:
        """Combine all parts of the query into one boolean expression."""
        parts: list[pl.Expr] = []
        if q.duplicates_only:
            parts.append(duplicate_key() if self.result.has_duplicates else pl.lit(False))
        if len(q.statuses) < len(STATUS_ORDER):
            parts.append(pl.col(STATUS).is_in([s.value for s in q.statuses]))
        if q.diff_column is not None:
            differs = pl.col(dcol(self.result.columns[q.diff_column].name))
            # "equal" needs both sides: a row missing on one side is equal in no column
            parts.append(~differs & self._present_l & self._present_r if q.diff_equal else differs)
        if q.search.strip():
            parts.append(self._search(q.search.strip().lower()))
        parts.extend(self._column_filter(f) for f in q.filters)
        return pl.all_horizontal(parts) if parts else pl.lit(True)

    def _sides(self, i: int) -> list[tuple[pl.Expr, pl.Expr, DataType]]:
        """``(value, present, dtype)`` for every side column ``i`` can exist on."""
        c = self.result.columns[i]
        if c.kind == "key":
            return [(pl.col(c.name), pl.lit(True), c.dtype)]
        sides: list[tuple[pl.Expr, pl.Expr, DataType]] = []
        if c.dtype_left is not None:
            sides.append((pl.col(lcol(c.name)), self._present_l, c.dtype_left))
        if c.dtype_right is not None:
            sides.append((pl.col(rcol(c.name)), self._present_r, c.dtype_right))
        return sides

    def _search(self, needle: str) -> pl.Expr:
        """Case-insensitive substring search across all values of both sides.

        Two optimisations keep this fast on millions of rows: non-text columns
        are only searched if the needle could appear in their text form (e.g.
        digits for numbers), and ASCII needles use polars' Aho-Corasick matcher
        instead of lower-casing every value first.
        """
        exprs = [
            _contains(value, dtype, needle)
            for i in range(len(self.result.columns))
            for value, _present, dtype in self._sides(i)
            if _may_contain(dtype, needle)
        ]
        return pl.any_horizontal(exprs).fill_null(False) if exprs else pl.lit(False)

    def _column_filter(self, f: ColumnFilter) -> pl.Expr:
        """A row matches if any *existing* side of the column matches."""
        ftype = self.result.columns[f.column].filter_type
        tests = [
            present & _test(value, dtype, ftype, f)
            for value, present, dtype in self._sides(f.column)
        ]
        return pl.any_horizontal(tests).fill_null(False)

    def _sorted(self, lazy: pl.LazyFrame, q: Query) -> pl.LazyFrame:
        """Apply the sort of ``q`` (stable, nulls last, row id as tie-breaker)."""
        if q.sort_by_status:
            key = pl.col(STATUS).to_physical()
        elif q.sort_column is not None:
            key = self._sort_key(q.sort_column)
        else:
            return lazy.sort(ROW)
        return lazy.sort([key, pl.col(ROW)], descending=[q.descending, False], nulls_last=True)

    def _sort_key(self, i: int) -> pl.Expr:
        """Value to sort column ``i`` by: the left value, else the right one."""
        c = self.result.columns[i]
        if c.kind == "key":
            return pl.col(c.name)
        if c.kind == "left_only":
            return pl.col(lcol(c.name))
        if c.kind == "right_only":
            return pl.col(rcol(c.name))
        left, right = pl.col(lcol(c.name)), pl.col(rcol(c.name))
        if c.compare_dtype is not None:
            left = (
                as_text(left, c.dtype_left)
                if c.compare_dtype == pl.String
                else left.cast(c.compare_dtype)
            )
            right = (
                as_text(right, c.dtype_right)
                if c.compare_dtype == pl.String
                else right.cast(c.compare_dtype)
            )
        return pl.when(self._present_l).then(left).otherwise(right)


# --------------------------------------------------------------------------- #
# Filter tests
# --------------------------------------------------------------------------- #


def _test(value: pl.Expr, dtype: DataType, ftype: str, f: ColumnFilter) -> pl.Expr:  # noqa: PLR0911, PLR0912 - flat operator dispatch
    """Boolean expression for one side of one column filter (nulls → False)."""
    if dtype.is_float():
        value = value.fill_nan(None)  # NaN counts as missing, as in the comparison
    match f.op:
        case "is_null":
            return value.is_null()
        case "not_null":
            return value.is_not_null()
        case "true" | "false":
            return value.eq_missing(pl.lit(f.op == "true"))

    if ftype == "number":
        return _number_test(value, f)
    if ftype == "datetime":
        return _date_test(value, f)

    text = as_text(value, dtype)
    lower, needle = text.str.to_lowercase(), f.a.lower()
    match f.op:
        case "is_empty":
            return value.is_null() | (text == "")
        case "not_empty":
            return value.is_not_null() & (text != "")
        case "equals":
            result = text == f.a
        case "starts":
            result = lower.str.starts_with(needle)
        case "ends":
            result = lower.str.ends_with(needle)
        case "not_contains":
            result = ~lower.str.contains(needle, literal=True)
        case _:
            result = lower.str.contains(needle, literal=True)
    return result.fill_null(False)


_NUMERIC_CHARS: Final = frozenset("0123456789.-+e")
_TEMPORAL_CHARS: Final = frozenset("0123456789-:t .")
_BOOL_WORDS: Final = ("true", "false")


def _may_contain(dtype: DataType, needle: str) -> bool:
    """Can the text form of a ``dtype`` value contain ``needle`` at all?"""
    if dtype == pl.Boolean:
        return any(needle in word for word in _BOOL_WORDS)
    if dtype.is_numeric():
        return set(needle) <= _NUMERIC_CHARS | {"n", "a", "i", "f"}  # also nan / inf
    if dtype.is_temporal():
        return set(needle) <= _TEMPORAL_CHARS
    return True


def _contains(value: pl.Expr, dtype: DataType, needle: str) -> pl.Expr:
    """Case-insensitive ``needle in value`` for one column (needle is lower-case)."""
    text = as_text(value, dtype)
    if needle.isascii():
        return text.str.contains_any([needle], ascii_case_insensitive=True)
    return text.str.to_lowercase().str.contains(needle, literal=True)


def _number_test(value: pl.Expr, f: ColumnFilter) -> pl.Expr:
    a, b = _to_float(f.a), _to_float(f.b)
    if f.op == "eq":
        return (value == a).fill_null(False) if a is not None else pl.lit(False)
    if f.op == "ne":
        return (value != a).fill_null(False) if a is not None else pl.lit(False)
    cond = value.is_not_null()
    if a is not None:
        cond &= value >= a
    if b is not None:
        cond &= value <= b
    return cond.fill_null(False)


def _date_test(value: pl.Expr, f: ColumnFilter) -> pl.Expr:
    as_date = value.cast(pl.Date, strict=False)
    cond = value.is_not_null()
    if (a := _to_date(f.a)) is not None:
        cond &= as_date >= a
    if (b := _to_date(f.b)) is not None:
        cond &= as_date <= b
    return cond.fill_null(False)


def _to_float(text: str) -> float | None:
    try:
        return float(text) if text.strip() else None
    except ValueError:
        return None


def _to_date(text: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(text.strip()[:10]) if text.strip() else None
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# JSON conversion of single values
# --------------------------------------------------------------------------- #


#: largest integer a JavaScript number represents exactly (Number.MAX_SAFE_INTEGER)
_MAX_SAFE_INT: Final = 2**53 - 1


def _json(v: Any) -> Any:  # noqa: PLR0911 - a flat dispatch reads best here
    """Make one value JSON-safe for the frontend.

    Non-finite floats become strings (``"NaN"``, ``"inf"``) so they stay
    distinguishable from null; temporal values use ISO 8601. Integers beyond
    JavaScript's exact range and decimals a float cannot hold exactly become
    strings too - otherwise two different values could show up as equal.
    """
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v if abs(v) <= _MAX_SAFE_INT else str(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, Decimal):
        if not v.is_finite():  # NaN, sNaN, Infinity: float() rejects sNaN
            return str(v)
        f = float(v)
        return f if Decimal(repr(f)) == v else str(v)
    if isinstance(v, dt.timedelta):
        return str(v)
    if isinstance(v, bytes):
        return v.hex()
    return str(v)
