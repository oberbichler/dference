"""Core comparison engine.

Two frames are matched with a single full outer join on the key columns.
Everything else - status, per-column difference flags, statistics - is derived
from that joined frame with vectorised polars expressions, so the cost is
dominated by one hash join and one pass over the compared columns.

The joined frame (``DiffResult.data``) uses reserved internal column names:

* ``__fd_row``          - stable row id (0..n-1), also the position in ``data``
* ``__fd_status``       - :class:`Status` as a polars ``Enum``
* ``__fd_l:<col>``      - value from the left frame
* ``__fd_r:<col>``      - value from the right frame
* ``__fd_d:<col>``      - ``True`` where the compared column differs
* ``<key>``             - key columns, coalesced from both sides
* ``__fd_nl``/``__fd_nr`` - how often the row's key occurs on the left/right
  (only present if a key is not unique on one side)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from polars.datatypes import DataType

__all__ = ["ColumnInfo", "DiffResult", "Status", "Summary", "compare"]

# --------------------------------------------------------------------------- #
# Constants & naming
# --------------------------------------------------------------------------- #

_PREFIX = "__fd_"
ROW = f"{_PREFIX}row"
STATUS = f"{_PREFIX}status"
_IN_LEFT = f"{_PREFIX}in_l"
_IN_RIGHT = f"{_PREFIX}in_r"
_OCC = f"{_PREFIX}occ"
COUNT_LEFT = f"{_PREFIX}nl"
COUNT_RIGHT = f"{_PREFIX}nr"

#: How rows that share a key are paired (see :func:`compare`).
Duplicates = Literal["match", "number", "raise"]
#: Above this many candidate pairs, rows sharing a key are paired by order.
_MAX_CANDIDATES = 2_000_000


def duplicate_key() -> pl.Expr:
    """``True`` for rows whose key occurs more than once on a side.

    Only valid on ``DiffResult.data`` of a result with duplicates.
    """
    return (pl.col(COUNT_LEFT) > 1) | (pl.col(COUNT_RIGHT) > 1)


def lcol(name: str) -> str:
    """Internal name of the left-hand value column for ``name``."""
    return f"{_PREFIX}l:{name}"


def rcol(name: str) -> str:
    """Internal name of the right-hand value column for ``name``."""
    return f"{_PREFIX}r:{name}"


def dcol(name: str) -> str:
    """Internal name of the difference flag for ``name``."""
    return f"{_PREFIX}d:{name}"


class Status(StrEnum):
    """Classification of a row after matching both frames on the key."""

    EQUAL = "equal"
    """Key on both sides, all compared values equal."""
    MISMATCH = "mismatch"
    """Key on both sides, at least one compared value differs."""
    MISSING_LEFT = "missing_left"
    """Key only in the right frame."""
    MISSING_RIGHT = "missing_right"
    """Key only in the left frame."""


#: Display and sort order of the statuses.
#: Display and sort order: left before right, so "only in left" (a row missing on
#: the right) comes before "only in right".
STATUS_ORDER: tuple[Status, ...] = (
    Status.EQUAL,
    Status.MISMATCH,
    Status.MISSING_RIGHT,
    Status.MISSING_LEFT,
)
STATUS_DTYPE = pl.Enum([s.value for s in STATUS_ORDER])

ColumnKind = Literal["key", "compared", "left_only", "right_only"]
FilterType = Literal["number", "boolean", "datetime", "string"]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Summary:
    """Row counts of a comparison."""

    equal: int
    mismatch: int
    missing_left: int
    missing_right: int
    left_rows: int
    right_rows: int
    duplicate_keys_left: int = 0
    """Keys that occur more than once in the left frame."""
    duplicate_keys_right: int = 0
    """Keys that occur more than once in the right frame."""
    duplicates: str = "match"
    """How rows sharing a key were paired (``match`` or ``number``)."""

    @property
    def found(self) -> int:
        """Rows whose key exists on both sides (equal + mismatch)."""
        return self.equal + self.mismatch

    @property
    def not_found(self) -> int:
        """Rows whose key exists on one side only."""
        return self.missing_left + self.missing_right

    @property
    def total(self) -> int:
        """All rows of the joined result."""
        return self.found + self.not_found


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    """Metadata of one output column.

    Attributes:
        name: Column name as in the input frames.
        kind: ``key``, ``compared`` (present on both sides), ``left_only`` or
            ``right_only``.
        dtype_left: polars dtype on the left side, ``None`` if absent.
        dtype_right: polars dtype on the right side, ``None`` if absent.
        mismatches: Number of differing rows (compared columns only).
        compare_dtype: Common dtype both sides were cast to before comparing,
            ``None`` if the dtypes were identical.
    """

    name: str
    kind: ColumnKind
    dtype_left: DataType | None
    dtype_right: DataType | None
    mismatches: int = 0
    compare_dtype: DataType | None = None

    @property
    def dtype(self) -> DataType:
        """Representative dtype (the comparison dtype, else the side that exists)."""
        dtype = self.compare_dtype or self.dtype_left or self.dtype_right
        assert dtype is not None  # every column exists on at least one side
        return dtype

    @property
    def filter_type(self) -> FilterType:
        """Kind of filter the widget offers for this column."""
        return filter_type(self.dtype)


def filter_type(dtype: DataType) -> FilterType:
    """Map a polars dtype to the filter UI used by the widget."""
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_numeric():
        return "number"
    if dtype.is_temporal() and dtype.base_type() not in {pl.Duration, pl.Time}:
        return "datetime"
    return "string"


class DiffResult:
    """Outcome of :func:`compare`.

    The object is immutable in practice; all methods return new polars frames.
    Use :meth:`frame` for a wide view, :meth:`mismatches` for a long list of
    differing cells and :meth:`column_stats` for per-column match rates.
    """

    __slots__ = ("columns", "data", "ignored", "keys", "left_name", "right_name", "summary")

    def __init__(
        self,
        *,
        data: pl.DataFrame,
        keys: tuple[str, ...],
        columns: tuple[ColumnInfo, ...],
        summary: Summary,
        left_name: str,
        right_name: str,
        ignored: tuple[str, ...] = (),
    ) -> None:
        self.data = data
        self.ignored = ignored
        self.keys = keys
        self.columns = columns
        self.summary = summary
        self.left_name = left_name
        self.right_name = right_name

    def __repr__(self) -> str:
        s = self.summary
        return (
            f"DiffResult(keys={list(self.keys)}, equal={s.equal}, mismatch={s.mismatch}, "
            f"missing_left={s.missing_left}, missing_right={s.missing_right})"
        )

    # ---- column groups ------------------------------------------------------

    @property
    def has_duplicates(self) -> bool:
        """Whether a key occurs more than once on one side."""
        return COUNT_LEFT in self.data.columns

    @property
    def duplicate_rows(self) -> int:
        """Rows of the result whose key is not unique on a side."""
        if not self.has_duplicates:
            return 0
        return int(self.data.select(duplicate_key().sum()).item())

    @property
    def duplicates(self) -> pl.DataFrame:
        """Keys that occur more than once on a side, with their count per side.

        Columns: ``<keys>``, ``<left_name>``, ``<right_name>`` (number of rows
        with that key on each side). Empty if every key is unique.
        """
        names = {COUNT_LEFT: self.left_name, COUNT_RIGHT: self.right_name}
        if not self.has_duplicates:
            schema: dict[str, DataType] = {k: self.data.schema[k] for k in self.keys}
            return pl.DataFrame(schema={**schema, **dict.fromkeys(names.values(), pl.UInt32())})
        return (
            self.data.filter(duplicate_key())
            .select(*self.keys, *(pl.col(c).alias(n) for c, n in names.items()))
            .unique(self.keys, maintain_order=True)
        )

    @property
    def compared(self) -> tuple[str, ...]:
        """Names of the columns compared value by value."""
        return tuple(c.name for c in self.columns if c.kind == "compared")

    def side_label(self, name: str, side: Literal["l", "r"]) -> str:
        """Output name of a value column, e.g. ``"city [CRM]"``."""
        return f"{name} [{self.left_name if side == 'l' else self.right_name}]"

    # ---- tables -----------------------------------------------------------------

    def frame(
        self,
        status: Status | str | Iterable[Status | str] | None = None,
        *,
        rows: Sequence[int] | pl.Series | None = None,
    ) -> pl.DataFrame:
        """Wide result: key, status, differing columns and both values per column.

        Args:
            status: Keep only rows with this status (or any of these statuses).
            rows: Keep only these row ids (in the given order), e.g. the
                widget's ``selected_ids``.

        Returns:
            One row per key with columns ``<keys>``, ``status``, ``differing``
            (list of column names) and ``<col> [<left_name>]`` /
            ``<col> [<right_name>]`` for every value column.
        """
        data = self.data
        if rows is not None:
            data = data.select(pl.all().gather(pl.Series(rows, dtype=pl.UInt32)))
        if status is not None:
            wanted = [status] if isinstance(status, str) else list(status)
            data = data.filter(pl.col(STATUS).is_in([Status(s).value for s in wanted]))

        differing = pl.concat_list(
            [pl.when(pl.col(dcol(c))).then(pl.lit(c)) for c in self.compared]
            or [pl.lit(None, dtype=pl.String)]
        ).list.drop_nulls()

        value_cols: list[pl.Expr] = []
        for c in self.columns:
            if c.kind in ("compared", "left_only"):
                value_cols.append(pl.col(lcol(c.name)).alias(self.side_label(c.name, "l")))
            if c.kind in ("compared", "right_only"):
                value_cols.append(pl.col(rcol(c.name)).alias(self.side_label(c.name, "r")))

        return data.select(
            *self.keys,
            pl.col(STATUS).alias("status"),
            differing.alias("differing"),
            *value_cols,
        )

    def mismatches(self) -> pl.DataFrame:
        """Long format: one row per ``(key, column)`` that differs.

        Values are rendered as strings so columns of different dtypes fit into
        one table; nulls stay null.
        """
        parts = [
            self.data.filter(pl.col(dcol(c.name))).select(
                *self.keys,
                pl.lit(c.name).alias("column"),
                as_text(pl.col(lcol(c.name)), c.dtype_left).alias(self.left_name),
                as_text(pl.col(rcol(c.name)), c.dtype_right).alias(self.right_name),
            )
            for c in self.columns
            if c.kind == "compared"
        ]
        if not parts:
            schema: dict[str, DataType] = {k: self.data.schema[k] for k in self.keys}
            for name in ("column", self.left_name, self.right_name):
                schema[name] = pl.String()
            return pl.DataFrame(schema=schema)
        return pl.concat(parts, how="vertical")

    def column_stats(self) -> pl.DataFrame:
        """Per compared column: dtypes, number and share of differing rows.

        Shares are relative to the rows found on both sides.
        """
        found = self.summary.found
        stats = [
            {
                "column": c.name,
                "dtype_left": str(c.dtype_left),
                "dtype_right": str(c.dtype_right),
                "mismatches": c.mismatches,
                "equal_share": (found - c.mismatches) / found if found else None,
                "mismatch_share": c.mismatches / found if found else None,
            }
            for c in self.columns
            if c.kind == "compared"
        ]
        schema = {
            "column": pl.String,
            "dtype_left": pl.String,
            "dtype_right": pl.String,
            "mismatches": pl.Int64,
            "equal_share": pl.Float64,
            "mismatch_share": pl.Float64,
        }
        return pl.DataFrame(stats, schema=schema).sort("mismatches", descending=True)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def to_polars(obj: Any, side: str) -> pl.DataFrame:
    """Convert a supported frame to an eager polars ``DataFrame``.

    Supported: polars ``DataFrame``/``LazyFrame``, pandas ``DataFrame`` (index is
    ignored), objects with ``to_polars()`` (e.g. DuckDB relations, narwhals) and
    anything implementing the Arrow PyCapsule stream interface
    (``__arrow_c_stream__``, e.g. pyarrow tables).

    Raises:
        TypeError: If the object cannot be converted.
    """
    if isinstance(obj, pl.DataFrame):
        return obj
    if isinstance(obj, pl.LazyFrame):
        return obj.collect()
    module = type(obj).__module__.partition(".")[0]
    if module == "pandas":
        return _from_pandas(obj)
    if hasattr(obj, "to_polars"):
        result = obj.to_polars()
        return result.collect() if isinstance(result, pl.LazyFrame) else result
    if hasattr(obj, "__arrow_c_stream__"):
        return pl.DataFrame(obj)
    msg = f"{side}: unsupported frame type {type(obj).__qualname__!r}"
    raise TypeError(msg)


def _from_pandas(df: Any) -> pl.DataFrame:
    """Convert pandas, falling back to strings for mixed-type object columns."""
    try:
        return pl.from_pandas(df)
    except (TypeError, ValueError, pl.exceptions.PolarsError):
        # Object columns holding mixed Python types cannot be converted as-is.
        fixed = df.copy()
        for name in fixed.columns:
            if fixed[name].dtype == object:
                fixed[name] = fixed[name].map(lambda v: v if v is None else str(v))
        return pl.from_pandas(fixed)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


class DtypeMismatchError(ValueError):
    """A column has dtypes on the two sides that cannot be aligned losslessly.

    ``hint`` is a suggestion how to cast one side (a polars expression as text).
    """

    def __init__(self, left: DataType, right: DataType, hint: str) -> None:
        super().__init__(f"{left} vs. {right}")
        self.hint = hint


_TIME_UNITS = ("ms", "us", "ns")  # coarse to fine
_MAX_DECIMAL_PRECISION = 38


def common_dtype(left: DataType, right: DataType) -> DataType | None:  # noqa: PLR0911 - flat dispatch
    """Dtype both sides are cast to before comparing; ``None`` if identical.

    Only lossless alignments happen automatically - the same values stored
    differently: integers of different width (``UInt64`` vs. a signed type is
    refused), ``Float32`` vs. ``Float64``, decimals of different precision,
    ``Datetime``/``Duration`` in different time units (same time zone), text vs.
    ``Categorical``/``Enum`` (compared as ``String``) and an all-null column
    vs. anything. Everything else raises :class:`DtypeMismatchError`: casting
    between different kinds of values is the caller's decision.
    """
    if left == right:
        return None
    if isinstance(left, pl.Null):
        return right
    if isinstance(right, pl.Null):
        return left
    if left.is_integer() and right.is_integer():
        target = _common_integer(left, right)
        if target is not None:
            return target
    elif left.is_float() and right.is_float():
        return pl.Float64()
    elif isinstance(left, pl.Decimal) and isinstance(right, pl.Decimal):
        target = _common_decimal(left, right)
        if target is not None:
            return target
    elif isinstance(left, pl.Datetime) and isinstance(right, pl.Datetime):
        if left.time_zone == right.time_zone:
            return pl.Datetime(_finer(left.time_unit, right.time_unit), left.time_zone)
    elif isinstance(left, pl.Duration) and isinstance(right, pl.Duration):
        return pl.Duration(_finer(left.time_unit, right.time_unit))
    elif _is_text(left) and _is_text(right):
        return pl.String()
    raise DtypeMismatchError(left, right, _cast_hint(left, right))


def _common_integer(left: DataType, right: DataType) -> DataType | None:
    """Smallest integer dtype holding every value of both, if there is one."""
    bits = {
        pl.Int8: 8, pl.Int16: 16, pl.Int32: 32, pl.Int64: 64, pl.Int128: 128,
        pl.UInt8: 8, pl.UInt16: 16, pl.UInt32: 32, pl.UInt64: 64,
    }  # fmt: skip
    lb, rb = bits.get(left.base_type()), bits.get(right.base_type())
    if lb is None or rb is None:
        return None
    lu, ru = left.is_unsigned_integer(), right.is_unsigned_integer()
    if lu == ru:
        width = max(lb, rb)
    else:  # a signed type needs one more bit than the unsigned one it must hold
        unsigned, signed = (lb, rb) if lu else (rb, lb)
        width = max(signed, unsigned * 2)
    names = {8: "Int8", 16: "Int16", 32: "Int32", 64: "Int64", 128: "Int128"}
    if width not in names:
        return None
    return getattr(pl, ("U" if lu and ru else "") + names[width])()


def _common_decimal(left: pl.Decimal, right: pl.Decimal) -> pl.Decimal | None:
    scale = max(left.scale, right.scale)
    digits = max((left.precision or 38) - left.scale, (right.precision or 38) - right.scale)
    if digits + scale > _MAX_DECIMAL_PRECISION:
        return None
    return pl.Decimal(digits + scale, scale)


def _finer(a: str | None, b: str | None) -> Any:
    return max(a or "us", b or "us", key=_TIME_UNITS.index)


def _is_text(dtype: DataType) -> bool:
    return isinstance(dtype, (pl.String, pl.Categorical, pl.Enum))


def _cast_hint(left: DataType, right: DataType) -> str:
    """How to make ``right`` match ``left`` (``{col}`` is the column)."""
    if isinstance(left, pl.Datetime) and isinstance(right, pl.Datetime):
        if left.time_zone and right.time_zone:
            return f'pl.col({{col}}).dt.convert_time_zone("{left.time_zone}")'
        if left.time_zone:
            return f'pl.col({{col}}).dt.replace_time_zone("{left.time_zone}")'
        return "pl.col({col}).dt.replace_time_zone(None)"
    if isinstance(left, pl.Date) and isinstance(right, pl.Datetime):
        return "pl.col({col}).dt.date()"
    return f"pl.col({{col}}).cast(pl.{left!r})"


def _dtype_mismatches(
    lf: pl.DataFrame,
    rf: pl.DataFrame,
    columns: Iterable[str],
    lname: str,
    rname: str,
    *,
    strict: bool,
) -> dict[str, DataType | None]:
    """Common dtype per column; raise one error listing every column that has none.

    With ``strict`` the dtypes must be identical; otherwise lossless differences
    are aligned (:func:`common_dtype`).
    """
    casts: dict[str, DataType | None] = {}
    problems: list[str] = []
    for c in columns:
        ldt, rdt = lf.schema[c], rf.schema[c]
        try:
            casts[c] = target = common_dtype(ldt, rdt)
        except DtypeMismatchError as exc:
            hint, lossless = exc.hint, False
        else:
            if not strict or target is None:
                continue
            hint, lossless = _cast_hint(ldt, rdt), True
        cast = hint.format(col=json.dumps(c))
        problems.append(
            f"  - {c}: {ldt} in {lname}, {rdt} in {rname}"
            f" - e.g. right = right.with_columns({cast})"
            + (" or pass strict=False to align it" if lossless else "")
        )
    if problems:
        rule = (
            "with strict=True, dference requires identical dtypes"
            if strict
            else "dference only aligns lossless differences such as Int32 vs. Int64"
        )
        msg = (
            f"Columns have different dtypes in {lname} and {rname}; cast one side first "
            f"({rule}), or leave them out with ignore_columns=[...]:\n" + "\n".join(problems)
        )
        raise ValueError(msg)
    return casts


def as_text(expr: pl.Expr, dtype: DataType | None) -> pl.Expr:
    """Render a column as String.

    Scalars use polars' cast. Nested types (List, Array, Struct) cannot be cast
    to String and are rendered as JSON in Python - slower, but rare.
    """
    if dtype is not None and dtype.is_nested():
        return expr.map_elements(_nested_to_json, return_dtype=pl.String)
    return expr.cast(pl.String, strict=False)


def _nested_to_json(value: Any) -> str | None:
    """JSON text of a nested value (lists arrive as ``pl.Series``)."""
    if value is None:
        return None
    if isinstance(value, pl.Series):
        value = value.to_list()
    return json.dumps(value, default=str, ensure_ascii=False)


def compare(
    left: Any,
    right: Any,
    key: str | Sequence[str],
    *,
    left_name: str = "left",
    right_name: str = "right",
    ignore_columns: Iterable[str] = (),
    strict: bool = True,
    duplicates: Duplicates = "match",
    order_by: str | Sequence[str] | None = None,
) -> DiffResult:
    """Match two frames on ``key`` and classify every row.

    Values are compared exactly; missing values are equal to each other
    (``null``, and ``NaN`` in float columns). There is deliberately no numeric
    tolerance - round both frames first (``df.with_columns(cs.float().round(2))``)
    if small float differences should not count.

    A column must have the same dtype on both sides (``strict``). With
    ``strict=False`` lossless differences are aligned (see :func:`common_dtype`),
    e.g. ``Int32`` vs. ``Int64``. Anything else is the caller's decision: cast
    one side first or leave the column out with ``ignore_columns``.

    Args:
        left: Left frame (polars, pandas, pyarrow, …).
        right: Right frame.
        key: Column name or list of column names. Null keys match null keys.
            Rows that share a key on one side are paired as ``duplicates``
            says; see :attr:`DiffResult.duplicates`.
        left_name: Display name of the left side.
        right_name: Display name of the right side.
        ignore_columns: Columns to leave out of the comparison entirely.
        strict: Require identical dtypes on both sides (the default). ``False``
            aligns lossless differences such as integer width, ``Float32`` vs.
            ``Float64``, time units within one time zone or text vs.
            ``Categorical``.
        duplicates: How rows that share a key are paired with the rows of
            the other side. ``"match"`` (the default) pairs identical rows
            first, then each remaining row with the most similar one (fewest
            differing columns), regardless of the row order. ``"number"``
            pairs them by position: the first row with a key on the left with
            the first one on the right, and so on (sorted by ``order_by``
            first, if given). ``"raise"`` refuses keys that are not unique.
            Rows left over because a key occurs more often on one side are
            only in that side.
        order_by: Column(s) that sort rows sharing a key before they are
            numbered (``duplicates="number"`` only); default: row order.

    Returns:
        A :class:`DiffResult`.

    Raises:
        ValueError: On missing keys, duplicate keys with ``duplicates="raise"``,
            an unknown ``duplicates`` mode, key or value columns whose
            dtypes differ beyond a lossless alignment (one error lists them all,
            each with a cast), reserved column names or identical side names.
        TypeError: If an input cannot be converted to polars.
    """
    if left_name == right_name:
        msg = "left_name and right_name must differ"
        raise ValueError(msg)

    lf, rf = to_polars(left, left_name), to_polars(right, right_name)
    keys = (key,) if isinstance(key, str) else tuple(key)
    _validate_keys(lf, rf, keys, left_name, right_name)
    order = _check_duplicates(lf, rf, keys, left_name, right_name, duplicates, order_by)

    ignore_list = list(dict.fromkeys(ignore_columns))
    ignored = set(ignore_list) - set(keys)
    lcols = [c for c in lf.columns if c not in keys and c not in ignored]
    rcols = [c for c in rf.columns if c not in keys and c not in ignored]
    rset, lset = set(rcols), set(lcols)
    compared = [c for c in lcols if c in rset]

    # one error for every key and value column whose dtypes differ irreconcilably
    casts = _dtype_mismatches(lf, rf, [*keys, *compared], left_name, right_name, strict=strict)
    key_casts = {k: t for k in keys if (t := casts.pop(k)) is not None}
    if key_casts:
        exprs = [pl.col(k).cast(t) for k, t in key_casts.items()]
        lf, rf = lf.with_columns(exprs), rf.with_columns(exprs)

    columns: list[ColumnInfo] = [ColumnInfo(k, "key", lf.schema[k], rf.schema[k]) for k in keys]

    # ---- one full outer join ----------------------------------------------
    lsel = lf.select(
        *keys, *(pl.col(c).alias(lcol(c)) for c in lcols), pl.lit(True).alias(_IN_LEFT)
    )
    rsel = rf.select(
        *keys, *(pl.col(c).alias(rcol(c)) for c in rcols), pl.lit(True).alias(_IN_RIGHT)
    )
    key_counts = _key_counts(lf, rf, keys)
    on = list(keys)
    if key_counts is not None:
        # keys that are not unique: join on (key, occurrence) instead
        locc, rocc = _occurrences(
            lf, rf, lsel, rsel, keys, key_counts, compared, casts, duplicates, order
        )
        lsel, rsel = lsel.with_columns(locc.alias(_OCC)), rsel.with_columns(rocc.alias(_OCC))
        on.append(_OCC)
    joined = lsel.lazy().join(
        rsel.lazy(),
        on=on,
        how="full",
        coalesce=True,
        nulls_equal=True,
        maintain_order="left_right",
    )
    if key_counts is not None:
        joined = joined.drop(_OCC).join(
            key_counts.lazy(), on=list(keys), how="left", nulls_equal=True, maintain_order="left"
        )

    in_l = pl.col(_IN_LEFT).fill_null(False)
    in_r = pl.col(_IN_RIGHT).fill_null(False)
    diff_exprs = [
        _differs(c, lf.schema[c], rf.schema[c], casts[c]).and_(in_l & in_r).alias(dcol(c))
        for c in compared
    ]
    any_diff = pl.any_horizontal([pl.col(dcol(c)) for c in compared]) if compared else pl.lit(False)
    status = (
        pl.when(~in_r)
        .then(pl.lit(Status.MISSING_RIGHT.value))
        .when(~in_l)
        .then(pl.lit(Status.MISSING_LEFT.value))
        .when(any_diff)
        .then(pl.lit(Status.MISMATCH.value))
        .otherwise(pl.lit(Status.EQUAL.value))
        .cast(STATUS_DTYPE)
    )
    try:
        data = (
            joined.with_columns(diff_exprs)
            .with_columns(status.alias(STATUS))
            .drop(_IN_LEFT, _IN_RIGHT)
            .with_row_index(ROW)
            .collect()
        )
    except pl.exceptions.PolarsError as exc:  # pragma: no cover - defensive
        msg = f"Comparison failed: {exc}"
        raise ValueError(msg) from exc

    # ---- statistics in a single pass -------------------------------------
    counts = data.select(
        *(pl.col(dcol(c)).sum().alias(c) for c in compared),
        *((pl.col(STATUS) == s.value).sum().alias(f"{_PREFIX}n:{s.value}") for s in STATUS_ORDER),
    ).row(0, named=True)

    columns += [
        ColumnInfo(
            c, "compared", lf.schema[c], rf.schema[c], int(counts[c]), compare_dtype=casts[c]
        )
        for c in compared
    ]
    columns += [ColumnInfo(c, "left_only", lf.schema[c], None) for c in lcols if c not in rset]
    columns += [ColumnInfo(c, "right_only", None, rf.schema[c]) for c in rcols if c not in lset]

    summary = Summary(
        equal=int(counts[f"{_PREFIX}n:equal"]),
        mismatch=int(counts[f"{_PREFIX}n:mismatch"]),
        missing_left=int(counts[f"{_PREFIX}n:missing_left"]),
        missing_right=int(counts[f"{_PREFIX}n:missing_right"]),
        left_rows=lf.height,
        right_rows=rf.height,
        duplicate_keys_left=0 if key_counts is None else int((key_counts[COUNT_LEFT] > 1).sum()),
        duplicate_keys_right=0 if key_counts is None else int((key_counts[COUNT_RIGHT] > 1).sum()),
        duplicates="number" if duplicates == "number" else "match",
    )
    return DiffResult(
        data=data,
        keys=keys,
        columns=tuple(columns),
        summary=summary,
        left_name=left_name,
        right_name=right_name,
        ignored=tuple(c for c in ignore_list if c in ignored),
    )


def _differs(name: str, ldtype: DataType, rdtype: DataType, cast: DataType | None) -> pl.Expr:
    """``True`` where the two values of ``name`` differ.

    Missing values are equal to each other: ``null``, and for float columns
    also ``NaN`` (pandas uses NaN as its missing marker, so a frame converted
    from pandas and one from Arrow would otherwise disagree).
    """
    left, right = pl.col(lcol(name)), pl.col(rcol(name))
    if cast == pl.String:
        return ~as_text(left, ldtype).eq_missing(as_text(right, rdtype))
    if cast is not None:
        left, right = left.cast(cast, strict=False), right.cast(cast, strict=False)
    if (cast or ldtype).is_float():
        left, right = left.fill_nan(None), right.fill_nan(None)
    return ~left.eq_missing(right)


def _key_counts(lf: pl.DataFrame, rf: pl.DataFrame, keys: tuple[str, ...]) -> pl.DataFrame | None:
    """Rows per key on each side, or ``None`` if every key is unique on both."""
    if not (lf.select(keys).is_duplicated().any() or rf.select(keys).is_duplicated().any()):
        return None
    gl = lf.group_by(keys).len(COUNT_LEFT)
    gr = rf.group_by(keys).len(COUNT_RIGHT)
    return gl.join(gr, on=list(keys), how="full", coalesce=True, nulls_equal=True).with_columns(
        pl.col(COUNT_LEFT, COUNT_RIGHT).fill_null(0).cast(pl.UInt32)
    )


def _occurrences(  # noqa: PLR0917 - internal helper of compare()
    lf: pl.DataFrame,
    rf: pl.DataFrame,
    lsel: pl.DataFrame,
    rsel: pl.DataFrame,
    keys: tuple[str, ...],
    counts: pl.DataFrame,
    compared: list[str],
    casts: dict[str, DataType | None],
    mode: str,
    order: list[str],
) -> tuple[pl.Series, pl.Series]:
    """Occurrence number per row on each side; the join pairs equal numbers.

    The left rows of a key are numbered 0, 1, …; each right row gets the
    number of the left row it is paired with, unpaired right rows numbers
    after the last left one (so they stay only in the right side).
    """
    on = list(keys)
    number = pl.int_range(pl.len(), dtype=pl.UInt32)
    if mode == "number":
        # lf/rf hold the rows in the same order as lsel/rsel
        numbered = number.over(on, order_by=order) if order else number.over(on)
        return lf.select(numbered).to_series(), rf.select(numbered).to_series()

    li, ri = f"{_PREFIX}li", f"{_PREFIX}ri"
    lnum = lsel.select(number.over(on)).to_series()
    dup = (pl.col(COUNT_LEFT) > 1) | (pl.col(COUNT_RIGHT) > 1)
    dup_keys = counts.filter(dup).select(keys)
    size = int(counts.filter(dup).select((pl.col(COUNT_LEFT) * pl.col(COUNT_RIGHT)).sum()).item())
    ldup = lsel.with_columns(lnum.alias(_OCC)).with_row_index(li)
    ldup = ldup.join(dup_keys, on=on, how="semi", nulls_equal=True)
    rdup = rsel.with_row_index(ri).join(dup_keys, on=on, how="semi", nulls_equal=True)

    paired: dict[int, int] = {}
    if size <= _MAX_CANDIDATES:
        differs = [
            _differs(c, lf.schema[c], rf.schema[c], casts[c]).cast(pl.UInt32) for c in compared
        ]
        cand = (
            ldup.select(*keys, li, _OCC, *(lcol(c) for c in compared))
            .join(rdup.select(*keys, ri, *(rcol(c) for c in compared)), on=on, nulls_equal=True)
            .select(li, ri, _OCC, (pl.sum_horizontal(differs) if differs else pl.lit(0)).alias("n"))
            .sort("n", li, ri)
        )
        used_l: set[int] = set()
        for lrow, rrow, occ, _n in cand.iter_rows():
            if lrow not in used_l and rrow not in paired:
                used_l.add(lrow)
                paired[rrow] = occ
    else:  # too many candidates: pair by position instead
        return lnum, rsel.select(number.over(on)).to_series()

    rocc = (
        rsel.select(keys)
        .with_row_index(ri)
        .join(
            pl.DataFrame(
                {ri: list(paired), _OCC: list(paired.values())},
                schema={ri: pl.UInt32, _OCC: pl.UInt32},
            ),
            on=ri,
            how="left",
            maintain_order="left",
        )
        .join(counts, on=on, how="left", nulls_equal=True, maintain_order="left")
        .select(
            pl.when((pl.col(COUNT_LEFT) <= 1) & (pl.col(COUNT_RIGHT) <= 1))
            .then(0)  # a unique key: joined as usual
            .when(pl.col(_OCC).is_not_null())
            .then(pl.col(_OCC))
            .otherwise(
                pl.col(COUNT_LEFT) + pl.col(_OCC).is_null().cast(pl.UInt32).cum_sum().over(on) - 1
            )
            .cast(pl.UInt32)
        )
        .to_series()
    )
    return lnum, rocc


def _validate_keys(
    lf: pl.DataFrame, rf: pl.DataFrame, keys: tuple[str, ...], lname: str, rname: str
) -> None:
    """Check that the key is non-empty, unique and present on both sides."""
    if not keys:
        msg = "Specify at least one key column."
        raise ValueError(msg)
    if len(set(keys)) != len(keys):
        msg = f"Key columns must be distinct, got {list(keys)}."
        raise ValueError(msg)
    for name, df in ((lname, lf), (rname, rf)):
        missing = [k for k in keys if k not in df.columns]
        if missing:
            msg = f"Key column(s) {missing} missing in {name}."
            raise ValueError(msg)
        reserved = [c for c in df.columns if c.startswith(_PREFIX)]
        if reserved:
            msg = f"{name}: column names starting with {_PREFIX!r} are reserved: {reserved}"
            raise ValueError(msg)


def _check_duplicates(  # noqa: PLR0917 - internal helper of compare()
    lf: pl.DataFrame,
    rf: pl.DataFrame,
    keys: tuple[str, ...],
    lname: str,
    rname: str,
    duplicates: str,
    order_by: str | Sequence[str] | None,
) -> list[str]:
    """Validate ``duplicates`` and ``order_by``; return the ``order_by`` columns."""
    if duplicates not in {"match", "number", "raise"}:
        msg = f"duplicates must be 'match', 'number' or 'raise', got {duplicates!r}"
        raise ValueError(msg)
    order = [] if order_by is None else [order_by] if isinstance(order_by, str) else list(order_by)
    for name, df in ((lname, lf), (rname, rf)):
        if duplicates == "raise":
            _check_unique(df, keys, name)
        missing = [c for c in order if c not in df.columns]
        if missing:
            msg = f"order_by column(s) {missing} missing in {name}."
            raise ValueError(msg)
    return order


def _check_unique(df: pl.DataFrame, keys: tuple[str, ...], name: str) -> None:
    """Raise if a key combination occurs more than once."""
    dup = df.select(keys).is_duplicated()
    n = int(dup.sum())
    if n:
        examples = df.filter(dup).select(keys).unique(maintain_order=True).head(3).to_dicts()
        msg = (
            f"{name}: {n} rows share a key {list(keys)} (e.g. {examples}). "
            "The key must be unique on each side."
        )
        raise ValueError(msg)
