"""Rough timings for large inputs.

Usage::

    uv run python benchmarks/bench.py            # 1 million rows per side
    uv run python benchmarks/bench.py 5_000_000
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Any

import polars as pl

from dference import compare
from dference._query import Query, ViewEngine

if TYPE_CHECKING:
    from collections.abc import Callable


def make(n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two frames with ~5% mismatches and 1% missing rows on each side."""
    ids = pl.int_range(n, eager=True).alias("id")
    base = pl.DataFrame(ids).with_columns(
        pl.format("customer {}", "id").alias("name"),
        pl.lit("Berlin").alias("city"),
        (pl.col("id").cast(pl.Float64) * 1.5).alias("revenue"),
        (pl.col("id") % 3 == 0).alias("active"),
    )
    right = base.with_columns(
        pl.when(pl.col("id") % 20 == 0)
        .then(pl.col("revenue") + 1)
        .otherwise("revenue")
        .alias("revenue"),
        pl.when(pl.col("id") % 97 == 0).then(pl.lit("Kiel")).otherwise("city").alias("city"),
    )
    return base.filter(pl.col("id") % 100 != 1), right.filter(pl.col("id") % 100 != 2)


def timed(label: str, fn: Callable[[], Any]) -> Any:
    t0 = time.perf_counter()
    out = fn()
    print(f"{label:<44} {1000 * (time.perf_counter() - t0):8.0f} ms")
    return out


def main(n: int) -> None:
    left, right = timed(f"generate 2 x {n:,} rows", lambda: make(n))
    result = timed("compare (join + diff flags + stats)", lambda: compare(left, right, "id"))
    print(f"  {result!r}")
    engine = ViewEngine(result)
    cols = result.columns
    q_all = Query()
    q_mm = Query.from_message({"statuses": ["mismatch"]}, cols)
    q_search = Query.from_message({"search": "kiel"}, cols)
    q_filter = Query.from_message(
        {"filters": [{"column": 3, "op": "between", "a": "1000", "b": "200000"}]}, cols
    )
    q_sort = Query.from_message({"sort": {"column": 3, "descending": True}}, cols)
    timed("first page, no filter", lambda: engine.page(q_all, 0, 25))
    timed("first page, status = mismatch", lambda: engine.page(q_mm, 0, 25))
    timed("next page, same query (cached view)", lambda: engine.page(q_mm, 1, 25))
    timed("first page, full-text search", lambda: engine.page(q_search, 0, 25))
    timed("first page, numeric range filter (L or R)", lambda: engine.page(q_filter, 0, 25))
    timed("first page, sort by revenue desc", lambda: engine.page(q_sort, 0, 25))
    timed("mismatches() long format", result.mismatches)

    # keys that are not unique: every 10th key occurs three times on both sides,
    # in a different order on the right
    copies = pl.col("id") % 10 == 0
    dup_left = pl.concat([left, left.filter(copies), left.filter(copies)])
    dup_right = pl.concat([right, right.filter(copies), right.filter(copies)]).reverse()
    for mode in ("match", "number"):
        dup = timed(
            f"compare, 10% keys x3, duplicates={mode!r}",
            lambda mode=mode: compare(dup_left, dup_right, "id", duplicates=mode),
        )
        print(f"  {dup!r}")
    q_dup = Query(duplicates_only=True)
    timed("first page, duplicate keys only", lambda: ViewEngine(dup).page(q_dup, 0, 25))


if __name__ == "__main__":
    main(int(sys.argv[1].replace("_", "")) if len(sys.argv) > 1 else 1_000_000)
