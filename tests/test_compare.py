"""Tests for the comparison engine."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from dference import Status, compare
from dference._compare import DtypeMismatchError, common_dtype


def test_statuses(left: pl.DataFrame, right: pl.DataFrame) -> None:
    r = compare(left, right, "id")
    s = r.summary
    assert (s.equal, s.mismatch, s.missing_left, s.missing_right) == (2, 2, 1, 1)
    assert (s.found, s.not_found, s.total) == (4, 2, 6)
    assert (s.left_rows, s.right_rows) == (5, 5)
    status = dict(r.frame().select("id", "status").iter_rows())
    assert status == {
        1: "equal",
        2: "mismatch",
        3: "equal",
        4: "mismatch",
        5: "missing_right",
        6: "missing_left",
    }


def test_null_and_nan_count_as_equal(left: pl.DataFrame, right: pl.DataFrame) -> None:
    row3 = compare(left, right, "id").frame().filter(pl.col("id") == 3)
    assert row3["status"].item() == Status.EQUAL
    assert row3["differing"].item().to_list() == []


def test_differing_columns_and_stats(left: pl.DataFrame, right: pl.DataFrame) -> None:
    r = compare(left, right, "id")
    differing = dict(r.frame().select("id", "differing").iter_rows())
    assert differing[2] == ["name"]
    assert differing[4] == ["city"]
    stats = {row["column"]: row for row in r.column_stats().iter_rows(named=True)}
    assert stats["city"]["mismatches"] == 1
    assert stats["city"]["mismatch_share"] == pytest.approx(0.25)
    assert stats["revenue"]["mismatches"] == 0


def test_column_kinds(left: pl.DataFrame, right: pl.DataFrame) -> None:
    r = compare(left, right, "id", ignore_columns=["active"])
    kinds = {c.name: c.kind for c in r.columns}
    assert kinds == {
        "id": "key",
        "name": "compared",
        "city": "compared",
        "revenue": "compared",
        "since": "compared",
        "note": "left_only",
        "channel": "right_only",
    }
    assert r.ignored == ("active",)


def test_side_labels(left: pl.DataFrame, right: pl.DataFrame) -> None:
    cols = compare(left, right, "id", left_name="CRM", right_name="ERP").frame().columns
    assert {"city [CRM]", "city [ERP]", "note [CRM]"} <= set(cols)
    assert "note [ERP]" not in cols


def test_frame_filters(left: pl.DataFrame, right: pl.DataFrame) -> None:
    r = compare(left, right, "id")
    assert r.frame(Status.MISMATCH)["id"].to_list() == [2, 4]
    assert sorted(r.frame(["missing_left", "missing_right"])["id"].to_list()) == [5, 6]
    assert r.frame(rows=[3, 0])["id"].to_list() == [4, 1]


def test_mismatches_long_format(left: pl.DataFrame, right: pl.DataFrame) -> None:
    long = compare(left, right, "id", left_name="L", right_name="R").mismatches().sort("id")
    assert long.columns == ["id", "column", "L", "R"]
    assert long.rows() == [(2, "name", "Bob", "Bob "), (4, "city", "Kiel", "Hamburg")]


def test_mismatches_empty(left: pl.DataFrame) -> None:
    r = compare(left, left, "id")
    assert r.mismatches().is_empty()
    assert r.summary.equal == 5


def test_composite_key() -> None:
    a = pl.DataFrame({"k1": ["x", "x", "y"], "k2": [1, 2, 1], "v": [1, 2, 3]})
    b = pl.DataFrame({"k1": ["x", "y", "y"], "k2": [1, 1, 2], "v": [1, 30, 4]})
    s = compare(a, b, ["k1", "k2"]).summary
    assert (s.equal, s.mismatch, s.missing_left, s.missing_right) == (1, 1, 1, 1)


def test_null_keys_match() -> None:
    a = pl.DataFrame({"k": [None, 1], "v": [1, 2]})
    b = pl.DataFrame({"k": [None, 1], "v": [1, 3]})
    s = compare(a, b, "k").summary
    assert (s.equal, s.mismatch) == (1, 1)


def test_numeric_key_dtypes_are_aligned_unless_strict() -> None:
    a = pl.DataFrame({"k": pl.Series([1, 2], dtype=pl.Int32), "v": [1, 2]})
    b = pl.DataFrame({"k": pl.Series([1, 2], dtype=pl.Int64), "v": [1, 2]})
    assert compare(a, b, "k", strict=False).summary.equal == 2
    with pytest.raises(ValueError, match=r"k: Int32 in left, Int64 in right"):
        compare(a, b, "k")


def test_lossless_dtype_differences_are_aligned() -> None:
    a = pl.DataFrame(
        {
            "k": pl.Series([1, 2, 3], dtype=pl.Int32),
            "n": pl.Series([1, 2, 3], dtype=pl.Int16),
            "f": pl.Series([0.5, 1.5, 2.5], dtype=pl.Float32),
            "t": pl.Series(["a", "b", "c"], dtype=pl.Categorical),
            "e": pl.Series([None, None, None], dtype=pl.Null),
            "ts": pl.Series([dt.datetime(2026, 1, 1)] * 3).cast(pl.Datetime("ms")),
        }
    )
    b = pl.DataFrame(
        {
            "k": [1, 2, 3],
            "n": pl.Series([1, 2, 4], dtype=pl.UInt8),
            "f": [0.5, 1.5, 2.5],
            "t": ["a", "x", "c"],
            "e": ["x", None, None],
            "ts": pl.Series([dt.datetime(2026, 1, 1)] * 3).cast(pl.Datetime("ns")),
        }
    )
    r = compare(a, b, "k", strict=False)
    assert dict(r.frame().select("k", "differing").iter_rows()) == {1: ["e"], 2: ["t"], 3: ["n"]}
    info = {c.name: c.compare_dtype for c in r.columns}
    assert info == {
        "k": None,
        "n": pl.Int16,
        "f": pl.Float64,
        "t": pl.String,
        "e": pl.String,
        "ts": pl.Datetime("ns"),
    }


def test_strict_requires_identical_dtypes() -> None:
    a = pl.DataFrame(
        {
            "k": [1, 2],
            "n": pl.Series([1, 2], dtype=pl.Int32),
            "t": pl.Series(["a", "b"], dtype=pl.Categorical),
            "same": [1.5, 2.5],
        }
    )
    b = pl.DataFrame({"k": [1, 2], "n": [1, 2], "t": ["a", "b"], "same": [1.5, 2.5]})
    with pytest.raises(ValueError, match="different dtypes") as info:
        compare(a, b, "k")  # strict is the default
    msg = str(info.value)
    assert "n: Int32 in left, Int64 in right" in msg
    assert "t: Categorical in left, String in right" in msg
    assert "same" not in msg
    # lossless differences point to strict=False
    assert "strict=False" in msg
    s = compare(a, b, "k", strict=False).summary
    assert s.equal == 2


def test_strict_false_hint_only_for_lossless_differences() -> None:
    a = pl.DataFrame({"k": [1], "n": [1]})
    b = pl.DataFrame({"k": [1], "n": [1.0]})
    with pytest.raises(ValueError, match="different dtypes") as info:
        compare(a, b, "k")
    assert "strict=False" not in str(info.value)  # int vs float is never aligned


def test_dtype_mismatches_raise_with_all_columns_and_casts() -> None:
    a = pl.DataFrame(
        {
            "k": [1, 2],
            "n": [1, 2],
            "t": [1, 2],
            "d": [dt.date(2026, 1, 1)] * 2,
            "ok": ["x", "y"],
        }
    )
    b = pl.DataFrame(
        {
            "k": [1, 2],
            "n": [1.0, 2.5],
            "t": ["1", "2"],
            "d": [dt.datetime(2026, 1, 1)] * 2,
            "ok": ["x", "y"],
        }
    )
    with pytest.raises(ValueError, match="different dtypes") as info:
        compare(a, b, "k", left_name="CRM", right_name="ERP")
    msg = str(info.value)
    assert "n: Int64 in CRM, Float64 in ERP" in msg
    assert "t: Int64 in CRM, String in ERP" in msg
    assert "d: Date in CRM, Datetime(time_unit='us', time_zone=None) in ERP" in msg
    assert "ok" not in msg.split("\n", 1)[1]
    assert 'pl.col("n").cast(pl.Int64)' in msg
    assert 'pl.col("d").dt.date()' in msg
    assert "ignore_columns" in msg
    # leaving the columns out makes the comparison work
    assert compare(a, b, "k", ignore_columns=["n", "t", "d"]).summary.equal == 2


@pytest.mark.parametrize(
    ("right_tz", "hint"),
    [
        ("UTC", "dt.replace_time_zone(None)"),  # naive vs aware: make the right side naive
        (None, None),
    ],
)
def test_time_zones_must_match(right_tz: str | None, hint: str | None) -> None:
    naive = pl.DataFrame({"k": [1], "ts": [dt.datetime(2026, 1, 1)]})
    other = naive.with_columns(pl.col("ts").dt.replace_time_zone(right_tz))
    if hint is None:
        assert compare(naive, other, "k").summary.equal == 1
        return
    with pytest.raises(ValueError, match="different dtypes") as info:
        compare(naive, other, "k")
    assert hint in str(info.value)


def test_different_time_zones_raise() -> None:
    berlin = pl.DataFrame({"k": [1], "ts": [dt.datetime(2026, 1, 1)]}).with_columns(
        pl.col("ts").dt.replace_time_zone("Europe/Berlin")
    )
    utc = berlin.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
    with pytest.raises(ValueError, match="convert_time_zone") as info:
        compare(berlin, utc, "k")
    assert "Europe/Berlin" in str(info.value)
    fixed = utc.with_columns(pl.col("ts").dt.convert_time_zone("Europe/Berlin"))
    assert compare(berlin, fixed, "k").summary.equal == 1


def test_nested_columns() -> None:
    a = pl.DataFrame({"k": [1, 2], "tags": [["a"], ["b"]]})
    b = pl.DataFrame({"k": [1, 2], "tags": [["a"], ["c"]]})
    long = compare(a, b, "k").mismatches()
    assert long.rows() == [(2, "tags", '["b"]', '["c"]')]


@pytest.mark.parametrize(
    ("key", "message"),
    [([], "at least one key"), (["id", "id"], "distinct"), ("nope", "missing in left")],
)
def test_invalid_keys(
    left: pl.DataFrame, right: pl.DataFrame, key: str | list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compare(left, right, key)


def test_duplicate_keys_raise_on_request(left: pl.DataFrame, right: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"right: 2 rows share a key"):
        compare(left, pl.concat([right, right.head(1)]), "id", duplicates="raise")


def test_unknown_duplicates_mode_raises(left: pl.DataFrame, right: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="duplicates must be"):
        compare(left, right, "id", duplicates="ignore")  # ty: ignore[invalid-argument-type]


def test_unique_keys_have_no_duplicates(left: pl.DataFrame, right: pl.DataFrame) -> None:
    result = compare(left, right, "id")
    assert not result.has_duplicates
    assert result.duplicates.is_empty()
    assert result.duplicate_rows == 0
    assert result.summary.duplicate_keys_left == result.summary.duplicate_keys_right == 0


DUP_LEFT = pl.DataFrame({"id": [1, 1, 1, 2, 3], "v": ["a", "b", "c", "x", "y"]})
DUP_RIGHT = pl.DataFrame({"id": [1, 1, 2, 2, 3], "v": ["c", "a", "x", "z", "y"]})


def test_duplicates_are_matched_by_content() -> None:
    result = compare(DUP_LEFT, DUP_RIGHT, "id")
    rows = result.frame().sort("id", "v [left]", nulls_last=True)
    # identical rows pair up regardless of their order; the surplus is one-sided
    assert rows.select("id", "status", "v [left]", "v [right]").rows() == [
        (1, "equal", "a", "a"),
        (1, "missing_right", "b", None),
        (1, "equal", "c", "c"),
        (2, "equal", "x", "x"),
        (2, "missing_left", None, "z"),
        (3, "equal", "y", "y"),
    ]
    s = result.summary
    assert (s.duplicate_keys_left, s.duplicate_keys_right, s.duplicates) == (1, 2, "match")
    assert result.duplicates.rows() == [(1, 3, 2), (2, 1, 2)]
    assert result.duplicate_rows == 5


def test_duplicates_pair_the_most_similar_rows() -> None:
    left = pl.DataFrame({"id": [1, 1], "a": ["x", "y"], "b": [1, 2]})
    right = pl.DataFrame({"id": [1, 1], "a": ["y", "x"], "b": [20, 1]})
    rows = compare(left, right, "id").frame().sort("a [left]")
    # (x, 1) matches (x, 1) exactly, so (y, 2) is left for (y, 20): one difference
    assert rows.select("status", "a [right]", "b [right]").rows() == [
        ("equal", "x", 1),
        ("mismatch", "y", 20),
    ]


def test_duplicates_numbered_by_position_and_order_by() -> None:
    left = pl.DataFrame({"id": [1, 1], "t": [2, 1], "v": ["b", "a"]})
    right = pl.DataFrame({"id": [1, 1], "t": [1, 2], "v": ["a", "b"]})
    by_position = compare(left, right, "id", duplicates="number")
    assert by_position.summary.mismatch == 2
    assert by_position.summary.duplicates == "number"
    by_time = compare(left, right, "id", duplicates="number", order_by="t")
    assert by_time.summary.equal == 2
    with pytest.raises(ValueError, match="order_by column"):
        compare(left, right, "id", duplicates="number", order_by="missing")


def test_duplicate_null_keys_pair_with_each_other() -> None:
    left = pl.DataFrame({"id": [None, None], "v": [1, 2]}, schema={"id": pl.Int64, "v": pl.Int64})
    right = pl.DataFrame({"id": [None], "v": [2]}, schema={"id": pl.Int64, "v": pl.Int64})
    result = compare(left, right, "id")
    assert sorted(result.frame().get_column("status").to_list()) == ["equal", "missing_right"]


def test_text_vs_number_key_raises() -> None:
    with pytest.raises(ValueError, match="cast one side first"):
        compare(pl.DataFrame({"k": ["1"]}), pl.DataFrame({"k": [1]}), "k")


def test_int_vs_float_key_raises() -> None:
    with pytest.raises(ValueError, match=r"k: Int64 in left, Float64 in right"):
        compare(pl.DataFrame({"k": [1]}), pl.DataFrame({"k": [1.0]}), "k")


def test_reserved_names_and_side_names(left: pl.DataFrame, right: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="reserved"):
        compare(left.with_columns(__fd_x=1), right, "id")
    with pytest.raises(ValueError, match="must differ"):
        compare(left, right, "id", left_name="a", right_name="a")


def test_pandas_and_arrow_input(left: pl.DataFrame, right: pl.DataFrame) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    assert compare(left.to_pandas(), right.to_pandas(), "id").summary.mismatch == 2
    assert compare(left.to_arrow(), right.to_arrow(), "id").summary.mismatch == 2
    # pandas has no date dtype: its Date column became a datetime, Arrow kept the date
    with pytest.raises(ValueError, match=r"since: Datetime\(.*\) in left, Date in right"):
        compare(left.to_pandas(), right.to_arrow(), "id")


def test_pandas_mixed_object_column() -> None:
    pd = pytest.importorskip("pandas")
    a = pd.DataFrame({"k": [1, 2], "v": [1, "x"]})  # mixed Python types
    b = pd.DataFrame({"k": [1, 2], "v": ["1", "x"]})
    assert compare(a, b, "k").summary.equal == 2


def test_lazyframe_input(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert compare(left.lazy(), right.lazy(), "id").summary.total == 6


def test_unsupported_input() -> None:
    with pytest.raises(TypeError, match="unsupported frame type"):
        compare([1, 2], [1, 2], "id")


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (pl.Int32(), pl.Int32(), None),
        (pl.Int32(), pl.Int64(), pl.Int64()),
        (pl.UInt8(), pl.UInt32(), pl.UInt32()),
        (pl.UInt8(), pl.Int8(), pl.Int16()),
        (pl.UInt32(), pl.Int64(), pl.Int64()),
        (pl.UInt64(), pl.Int64(), pl.Int128()),
        (pl.Float32(), pl.Float64(), pl.Float64()),
        (pl.Decimal(10, 2), pl.Decimal(12, 4), pl.Decimal(12, 4)),
        (pl.Decimal(10, 0), pl.Decimal(6, 4), pl.Decimal(14, 4)),
        (pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("us")),
        (pl.Datetime("ms", "UTC"), pl.Datetime("ns", "UTC"), pl.Datetime("ns", "UTC")),
        (pl.Duration("ms"), pl.Duration("us"), pl.Duration("us")),
        (pl.String(), pl.Categorical(), pl.String()),
        (pl.Enum(["a"]), pl.Enum(["a", "b"]), pl.String()),
        (pl.Null(), pl.Int64(), pl.Int64()),
        (pl.Boolean(), pl.Null(), pl.Boolean()),
    ],
)
def test_common_dtype(a: pl.DataType, b: pl.DataType, expected: pl.DataType | None) -> None:
    assert common_dtype(a, b) == expected
    assert common_dtype(b, a) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (pl.Int64(), pl.Float64()),
        (pl.Int64(), pl.String()),
        (pl.Boolean(), pl.Int8()),
        (pl.Date(), pl.Datetime("us")),
        (pl.Datetime("us"), pl.Datetime("us", "UTC")),
        (pl.Datetime("us", "UTC"), pl.Datetime("us", "Europe/Berlin")),
        (pl.Decimal(38, 0), pl.Decimal(38, 2)),  # would need 40 digits
        (pl.List(pl.Int64), pl.List(pl.Int32)),
        (pl.Duration("us"), pl.Datetime("us")),
    ],
)
def test_common_dtype_refuses_lossy_or_different_types(a: pl.DataType, b: pl.DataType) -> None:
    with pytest.raises(DtypeMismatchError):
        common_dtype(a, b)
    with pytest.raises(DtypeMismatchError):
        common_dtype(b, a)


def test_repr(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert "mismatch=2" in repr(compare(left, right, "id"))
