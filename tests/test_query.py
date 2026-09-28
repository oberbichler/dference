"""Tests for the server-side view engine (filters, sort, paging, serialisation)."""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import Any

import polars as pl
import pytest

from dference import DiffResult, compare
from dference._query import Query, ViewEngine, _json


@pytest.fixture
def result(left: pl.DataFrame, right: pl.DataFrame) -> DiffResult:
    return compare(left, right, "id", left_name="L", right_name="R")


@pytest.fixture
def engine(result: DiffResult) -> ViewEngine:
    return ViewEngine(result)


def col(result: DiffResult, name: str) -> int:
    return next(i for i, c in enumerate(result.columns) if c.name == name)


def ids(engine: ViewEngine, **msg: Any) -> list[int]:
    """Key values (column ``id``) of the view described by ``msg``."""
    query = Query.from_message(msg, engine.result.columns)
    return engine.frame(query)["id"].to_list()


# ---- query parsing -------------------------------------------------------------


def test_invalid_query_parts_are_dropped(result: DiffResult) -> None:
    q = Query.from_message(
        {
            "statuses": ["equal", "bogus"],
            "diff_column": 0,  # key column: not a compared column
            "filters": [
                {"column": 99, "op": "contains", "a": "x"},
                {"column": col(result, "name"), "op": "between"},  # not a string op
                "garbage",
            ],
            "sort": {"column": -1},
        },
        result.columns,
    )
    assert {s.value for s in q.statuses} == {"equal"}
    assert q.diff_column is None
    assert q.diff_equal is False
    assert q.filters == ()
    assert q.sort_column is None


def test_diff_equal_must_be_a_bool(result: DiffResult) -> None:
    city = col(result, "city")
    assert Query.from_message({"diff_column": city, "diff_equal": True}, result.columns).diff_equal
    q = Query.from_message({"diff_column": city, "diff_equal": "yes"}, result.columns)
    assert q.diff_equal is False


def test_queries_are_hashable_and_cached(engine: ViewEngine) -> None:
    q = Query()
    assert engine.view(q) is engine.view(Query())


# ---- filtering -------------------------------------------------------------------


def test_default_view_keeps_join_order(engine: ViewEngine) -> None:
    assert ids(engine) == [1, 2, 3, 4, 5, 6]


def test_status_filter(engine: ViewEngine) -> None:
    assert ids(engine, statuses=["mismatch"]) == [2, 4]
    assert ids(engine, statuses=["missing_left", "missing_right"]) == [5, 6]
    assert ids(engine, statuses=[]) == []


def test_search_is_case_insensitive_and_covers_both_sides(engine: ViewEngine) -> None:
    assert ids(engine, search="HAMBURG") == [4]  # only in the right value
    assert ids(engine, search="kiel") == [4]  # only in the left value
    assert ids(engine, search="shop") == [3, 6]  # right-only column


def test_diff_column(engine: ViewEngine, result: DiffResult) -> None:
    assert ids(engine, diff_column=col(result, "city")) == [4]


def test_diff_column_equal_keeps_found_rows_only(engine: ViewEngine, result: DiffResult) -> None:
    # rows 5 and 6 exist on one side only: never "equal" in a column
    assert ids(engine, diff_column=col(result, "city"), diff_equal=True) == [1, 2, 3]
    assert ids(engine, diff_column=col(result, "name"), diff_equal=True) == [1, 3, 4]


def test_diff_equal_needs_a_diff_column(engine: ViewEngine) -> None:
    assert ids(engine, diff_equal=True) == [1, 2, 3, 4, 5, 6]


def test_filter_matches_left_or_right(engine: ViewEngine, result: DiffResult) -> None:
    city = col(result, "city")
    for value in ("Kiel", "Hamburg"):
        f = {"column": city, "op": "equals", "a": value}
        assert ids(engine, filters=[f]) == [4]


def test_missing_side_never_matches(engine: ViewEngine, result: DiffResult) -> None:
    # Row 6 has no left side and row 5 no right side: their absent values are
    # null in the joined frame but must not satisfy "is null".
    f = {"column": col(result, "city"), "op": "is_null"}
    assert ids(engine, filters=[f]) == [3]
    f = {"column": col(result, "note"), "op": "is_null"}
    assert ids(engine, filters=[f]) == []


@pytest.mark.parametrize(
    ("column", "flt", "expected"),
    [
        ("name", {"op": "contains", "a": "o"}, [2]),
        ("name", {"op": "not_contains", "a": "e"}, [1, 2, 3, 6]),
        ("name", {"op": "starts", "a": "d"}, [4]),
        ("name", {"op": "ends", "a": "b "}, [2]),
        ("city", {"op": "is_empty"}, [3]),
        ("city", {"op": "not_empty"}, [1, 2, 4, 5, 6]),
        ("revenue", {"op": "between", "a": "15", "b": "45"}, [2, 4]),
        ("revenue", {"op": "between", "a": "55"}, [6]),
        ("revenue", {"op": "eq", "a": "10"}, [1]),
        ("revenue", {"op": "ne", "a": "10"}, [2, 4, 5, 6]),
        ("active", {"op": "false"}, [2]),
        ("active", {"op": "is_null"}, [5]),
        ("since", {"op": "between", "a": "2024-01-05"}, [5, 6]),
        ("since", {"op": "between", "b": "2024-01-01"}, [1]),
        ("id", {"op": "between", "a": "6"}, [6]),
    ],
)
def test_column_filters(
    engine: ViewEngine, result: DiffResult, column: str, flt: dict[str, str], expected: list[int]
) -> None:
    assert ids(engine, filters=[{"column": col(result, column), **flt}]) == expected


def test_filters_combine_with_and(engine: ViewEngine, result: DiffResult) -> None:
    filters = [
        {"column": col(result, "revenue"), "op": "between", "a": "15"},
        {"column": col(result, "name"), "op": "contains", "a": "e"},
    ]
    assert ids(engine, filters=filters) == [4, 5]


# ---- sorting ---------------------------------------------------------------------


def test_sort_by_column(engine: ViewEngine, result: DiffResult) -> None:
    rev = col(result, "revenue")
    # NaN sorts as the largest float in polars; nulls would go last
    assert ids(engine, sort={"column": rev}) == [1, 2, 4, 5, 6, 3]
    assert ids(engine, sort={"column": rev, "descending": True})[:3] == [3, 6, 5]


def test_sort_by_status_is_stable(engine: ViewEngine) -> None:
    # equal, mismatch, only left (missing right), only right (missing left)
    assert ids(engine, sort={"column": "status"}) == [1, 3, 2, 4, 5, 6]


def test_sort_right_only_column(engine: ViewEngine, result: DiffResult) -> None:
    order = ids(engine, sort={"column": col(result, "channel")})
    assert order[-1] == 5  # no right side -> null -> last


# ---- paging & serialisation -----------------------------------------------------


def test_page_clamps(engine: ViewEngine) -> None:
    page = engine.page(Query(), page=99, page_size=4)
    assert page["page"] == 1
    assert page["filtered"] == 6
    assert len(page["rows"]) == 2
    assert engine.page(Query(), page=0, page_size=0)["page_size"] == 1


def test_row_format(engine: ViewEngine, result: DiffResult) -> None:
    rows = {r["l"][0] if r["l"] else r["r"][0]: r for r in engine.page(Query(), 0, 10)["rows"]}
    city, note, channel = (col(result, c) for c in ("city", "note", "channel"))
    assert rows[4]["s"] == "mismatch"
    assert rows[4]["d"] == [city]
    assert (rows[4]["l"][city], rows[4]["r"][city]) == ("Kiel", "Hamburg")
    assert rows[4]["r"][note] is None  # column does not exist on the right
    assert rows[5]["r"] is None  # missing right
    assert rows[6]["l"] is None  # missing left
    assert rows[6]["r"][channel] == "shop"
    assert rows[1]["l"][col(result, "since")] == "2024-01-01"


def test_diff_columns_reflect_view(engine: ViewEngine, result: DiffResult) -> None:
    q = Query.from_message({"statuses": ["mismatch"]}, result.columns)
    assert set(engine.view(q).diff_columns) == {col(result, "name"), col(result, "city")}
    q = Query.from_message({"statuses": ["equal"]}, result.columns)
    assert engine.view(q).diff_columns == ()


def test_export_csv(engine: ViewEngine) -> None:
    q = Query.from_message({"statuses": ["mismatch"]}, engine.result.columns)
    lines = engine.export_csv(q).decode().splitlines()
    assert lines[0].startswith("id,status,differing,")
    assert len(lines) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (1.5, 1.5),
        (math.nan, "nan"),
        (math.inf, "inf"),
        (dt.date(2024, 1, 2), "2024-01-02"),
        (dt.datetime(2024, 1, 2, 3, 4), "2024-01-02T03:04:00"),
        (dt.timedelta(seconds=90), "0:01:30"),
        (b"\x01", "01"),
        (True, True),
        (2**53 - 1, 2**53 - 1),
        # beyond 2**53 JavaScript numbers round: 2**62 and 2**62 + 1 would look equal
        (2**62 + 1, "4611686018427387905"),
        (-(2**53), "-9007199254740992"),
        (Decimal("1.25"), 1.25),
        (Decimal("0.1000000000000000000001"), "0.1000000000000000000001"),
        (Decimal("NaN"), "NaN"),
        (Decimal("sNaN"), "sNaN"),
    ],
)
def test_json_values(value: object, expected: object) -> None:
    assert _json(value) == expected


def test_large_frame_pages_quickly() -> None:
    """Smoke test: paging a 200k-row diff only serialises the page."""
    n = 200_000
    a = pl.DataFrame({"k": range(n), "v": range(n)})
    b = a.with_columns(pl.when(pl.col("k") % 10 == 0).then(-1).otherwise(pl.col("v")).alias("v"))
    engine = ViewEngine(compare(a, b, "k"))
    q = Query.from_message({"statuses": ["mismatch"], "sort": {"column": 1}}, engine.result.columns)
    page = engine.page(q, page=3, page_size=50)
    assert page["filtered"] == n // 10
    assert len(page["rows"]) == 50


def test_nan_counts_as_null_in_filters(engine: ViewEngine, result: DiffResult) -> None:
    f = {"column": col(result, "revenue"), "op": "is_null"}
    assert ids(engine, filters=[f]) == [3]


@pytest.mark.parametrize(
    ("needle", "expected"),
    [
        ("40", [4]),  # number column (40.0)
        ("2024-01-06", [6]),  # date column
        ("FALSE", [2]),  # boolean column
        ("xyz", []),
    ],
)
def test_search_non_text_columns(engine: ViewEngine, needle: str, expected: list[int]) -> None:
    assert ids(engine, search=needle) == expected


def test_search_non_ascii() -> None:
    a = pl.DataFrame({"id": [1, 2], "city": ["München", "Köln"]})
    engine = ViewEngine(compare(a, a, "id"))
    assert ids(engine, search="MÜN") == [1]
