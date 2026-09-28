"""Property-based tests: invariants of ``compare`` and the view engine on random frames."""

from __future__ import annotations

import math
import os
from typing import Any

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dference import DiffResult, compare
from dference._query import Query, ViewEngine

SETTINGS = settings(
    max_examples=int(os.environ.get("DFD_EXAMPLES", "150")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

ints = st.one_of(st.none(), st.integers(-3, 3))
floats = st.one_of(st.none(), st.just(math.nan), st.sampled_from([0.0, -0.0, 1.5, 2.0, 1e300]))
texts = st.one_of(st.none(), st.sampled_from(["", " ", "a", "a ", "b", "\u00a0", "ä"]))
bools = st.one_of(st.none(), st.booleans())


@st.composite
def pair(draw: st.DrawFn) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two frames with a unique ``id`` key; overlapping keys, random values."""
    keys = draw(st.lists(st.integers(0, 40), unique=True, max_size=25))
    in_left = draw(st.lists(st.booleans(), min_size=len(keys), max_size=len(keys)))
    in_right = draw(st.lists(st.booleans(), min_size=len(keys), max_size=len(keys)))

    def side(mask: list[bool]) -> pl.DataFrame:
        ids = [k for k, m in zip(keys, mask, strict=True) if m]
        n = len(ids)
        return pl.DataFrame(
            {
                "id": ids,
                "i": draw(st.lists(ints, min_size=n, max_size=n)),
                "f": draw(st.lists(floats, min_size=n, max_size=n)),
                "s": draw(st.lists(texts, min_size=n, max_size=n)),
                "b": draw(st.lists(bools, min_size=n, max_size=n)),
            },
            schema={
                "id": pl.Int64,
                "i": pl.Int64,
                "f": pl.Float64,
                "s": pl.String,
                "b": pl.Boolean,
            },
        )

    return side(in_left), side(in_right)


def statuses(result: DiffResult) -> dict[Any, str]:
    frame = result.frame()
    return dict(zip(frame["id"].to_list(), frame["status"].cast(pl.String).to_list(), strict=True))


def ids(engine: ViewEngine, query: Query) -> list[int]:
    return engine.frame(query)["id"].to_list()


# ---- compare ------------------------------------------------------------------


@SETTINGS
@given(pair())
def test_counts_add_up(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    left, right = frames
    s = compare(left, right, "id").summary
    lk, rk = set(left["id"]), set(right["id"])
    assert s.total == len(lk | rk) == s.equal + s.mismatch + s.missing_left + s.missing_right
    assert s.found == len(lk & rk) == s.equal + s.mismatch
    assert s.missing_right == len(lk - rk)
    assert s.missing_left == len(rk - lk)
    assert (s.left_rows, s.right_rows) == (len(left), len(right))


@SETTINGS
@given(pair())
def test_swapping_sides_swaps_the_missing_statuses(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    left, right = frames
    swap = {"missing_left": "missing_right", "missing_right": "missing_left"}
    forward = statuses(compare(left, right, "id"))
    backward = statuses(compare(right, left, "id"))
    assert backward == {k: swap.get(v, v) for k, v in forward.items()}


@SETTINGS
@given(pair())
def test_a_frame_equals_itself(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    left, _ = frames
    s = compare(left, left, "id").summary
    assert s.equal == len(left)
    assert s.mismatch == s.missing_left == s.missing_right == 0


@SETTINGS
@given(pair())
def test_mismatches_agree_with_statuses_and_column_counts(
    frames: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    left, right = frames
    result = compare(left, right, "id")
    long = result.mismatches()
    per_column = dict(zip(long["column"].to_list(), [0] * len(long), strict=False))
    for c in long["column"].to_list():
        per_column[c] += 1
    for c in result.columns:
        if c.kind == "compared":
            assert per_column.get(c.name, 0) == c.mismatches
    status = statuses(result)
    assert {k for k, v in status.items() if v == "mismatch"} == set(long["id"].to_list())
    wide = result.frame()
    for row in wide.iter_rows(named=True):
        assert (row["status"] == "mismatch") == bool(row["differing"])


@SETTINGS
@given(pair())
def test_missing_values_compare_equal(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    # null and NaN on both sides never count as a difference
    left, _ = frames
    blank = left.with_columns(pl.col("f").fill_nan(None))
    nan = left.with_columns(pl.when(pl.col("f").is_null()).then(math.nan).otherwise(pl.col("f")))
    assert compare(blank, nan, "id").summary.mismatch == 0


# ---- view engine ----------------------------------------------------------------

filters = st.lists(
    st.one_of(
        st.fixed_dictionaries(
            {
                "column": st.sampled_from([1, 2]),
                "op": st.sampled_from(["between", "eq", "ne", "is_null", "not_null"]),
                "a": st.sampled_from(["0", "1.5", "", "x"]),
                "b": st.sampled_from(["2", "", "-1"]),
            }
        ),
        st.fixed_dictionaries(
            {
                "column": st.just(3),
                "op": st.sampled_from(["contains", "equals", "starts", "is_empty", "is_null"]),
                "a": st.sampled_from(["a", "", " ", "A"]),
            }
        ),
        st.fixed_dictionaries(
            {"column": st.just(4), "op": st.sampled_from(["true", "false", "is_null"])}
        ),
    ),
    max_size=2,
)
messages = st.fixed_dictionaries(
    {
        "statuses": st.lists(
            st.sampled_from(["equal", "mismatch", "missing_left", "missing_right"]), unique=True
        ),
        "search": st.sampled_from(["", "a", "1", "true", "ä"]),
        "diff_column": st.sampled_from([None, 1, 2, 3, 4]),
        "diff_equal": st.booleans(),
        "filters": filters,
        "sort": st.one_of(
            st.none(),
            st.fixed_dictionaries(
                {"column": st.sampled_from([0, 1, 2, 3, 4, "status"]), "descending": st.booleans()}
            ),
        ),
    }
)


@SETTINGS
@given(pair(), messages)
def test_views_are_subsets_and_sorting_keeps_the_rows(
    frames: tuple[pl.DataFrame, pl.DataFrame], msg: dict[str, Any]
) -> None:
    left, right = frames
    result = compare(left, right, "id")
    engine = ViewEngine(result)
    query = Query.from_message(msg, result.columns)
    assert len(query.filters) == len(msg["filters"])  # all generated filters are valid
    view = ids(engine, query)
    assert len(view) == len(set(view))
    assert set(view) <= set(result.frame()["id"].to_list())
    unsorted = Query.from_message({**msg, "sort": None}, result.columns)
    assert sorted(view) == sorted(ids(engine, unsorted))


@SETTINGS
@given(pair(), messages, st.sampled_from([1, 3, 10]))
def test_pages_concatenate_to_the_view(
    frames: tuple[pl.DataFrame, pl.DataFrame], msg: dict[str, Any], size: int
) -> None:
    left, right = frames
    result = compare(left, right, "id")
    engine = ViewEngine(result)
    query = Query.from_message(msg, result.columns)
    view = ids(engine, query)
    first = engine.page(query, 0, size)
    assert first["filtered"] == len(view)
    key = [c.name for c in result.columns].index("id")
    seen: list[int] = []
    pages = max(1, math.ceil(len(view) / size))
    for p in range(pages):
        rows = engine.page(query, p, size)["rows"]
        seen += [(r["l"] or r["r"])[key] for r in rows]
    assert seen == view
    # a page past the end is clamped to the last page
    assert engine.page(query, pages + 5, size)["page"] == pages - 1


@SETTINGS
@given(pair(), st.sampled_from([1, 2, 3, 4]))
def test_equal_and_differs_split_the_found_rows(
    frames: tuple[pl.DataFrame, pl.DataFrame], column: int
) -> None:
    left, right = frames
    result = compare(left, right, "id")
    engine = ViewEngine(result)
    differs = set(ids(engine, Query(diff_column=column)))
    equal = set(ids(engine, Query(diff_column=column, diff_equal=True)))
    status = statuses(result)
    found = {k for k, v in status.items() if v in ("equal", "mismatch")}
    assert differs.isdisjoint(equal)
    assert differs | equal == found
    assert len(differs) == result.columns[column].mismatches


@SETTINGS
@given(pair())
def test_status_filters_partition_the_rows(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    left, right = frames
    result = compare(left, right, "id")
    engine = ViewEngine(result)
    parts = [
        set(ids(engine, Query.from_message({"statuses": [s]}, result.columns)))
        for s in ("equal", "mismatch", "missing_left", "missing_right")
    ]
    assert sum(len(p) for p in parts) == result.summary.total
    assert set().union(*parts) == set(result.frame()["id"].to_list())


@SETTINGS
@given(pair(), messages)
def test_export_matches_the_view(
    frames: tuple[pl.DataFrame, pl.DataFrame], msg: dict[str, Any]
) -> None:
    left, right = frames
    result = compare(left, right, "id")
    engine = ViewEngine(result)
    query = Query.from_message(msg, result.columns)
    csv = bytes(engine.export_csv(query)).decode()
    exported = pl.read_csv(csv.encode(), infer_schema=False) if csv.strip() else pl.DataFrame()
    assert len(exported) == len(ids(engine, query))


@pytest.mark.parametrize("size", [0, -1])
def test_page_size_is_at_least_one(size: int) -> None:
    result = compare(pl.DataFrame({"id": [1, 2]}), pl.DataFrame({"id": [1, 2]}), "id")
    page = ViewEngine(result).page(Query(), 0, size)
    assert page["page_size"] >= 1
