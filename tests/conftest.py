"""Shared fixtures: two small frames covering every status and edge case."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest


@pytest.fixture
def left() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Ann", "Bob", "Cid", "Dee", "Eve"],
            "city": ["Berlin", "Munich", None, "Kiel", "Bonn"],
            "revenue": [10.0, 20.0, float("nan"), 40.0, 50.0],
            "active": [True, False, True, True, None],
            "since": [dt.date(2024, 1, d) for d in (1, 2, 3, 4, 5)],
            "note": ["a", "b", "c", "d", "e"],  # left only
        }
    )


@pytest.fixture
def right() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 6],
            "name": ["Ann", "Bob ", "Cid", "Dee", "Fay"],  # trailing space on 2
            "city": ["Berlin", "Munich", None, "Hamburg", "Ulm"],  # 4 differs
            "revenue": [10.0, 20.0, float("nan"), 40.0, 60.0],  # NaN == NaN
            "active": [True, False, True, True, True],
            "since": [dt.date(2024, 1, d) for d in (1, 2, 3, 4, 6)],
            "channel": ["web", "web", "shop", "web", "shop"],  # right only
        }
    )
