"""Tests for the anywidget wrapper and its message protocol."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Any

import pytest

from dference import DataFrameDiff, compare
from dference._widget import short_names

if TYPE_CHECKING:
    import polars as pl


class Recorder:
    """Captures what the widget would send to the frontend."""

    def __init__(self) -> None:
        self.messages: list[tuple[dict[str, Any], list[bytes] | None]] = []

    def __call__(self, content: dict[str, Any], buffers: list[bytes] | None = None) -> None:
        self.messages.append((content, buffers))


@pytest.fixture
def widget(
    left: pl.DataFrame, right: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> DataFrameDiff:
    w = DataFrameDiff(left, right, key="id", left_name="CRM", right_name="ERP", page_size=2)
    monkeypatch.setattr(w, "send", Recorder())
    return w


def sent(widget: DataFrameDiff) -> Recorder:
    recorder = widget.send
    assert isinstance(recorder, Recorder)
    return recorder


def test_static_assets_are_packaged() -> None:
    static = files("dference") / "static"
    # explicit encoding: the files contain non-ASCII characters such as "≠", and
    # Windows defaults to cp1252; anywidget itself reads them as UTF-8 too
    assert (static / "widget.js").read_text(encoding="utf-8").startswith("// dference")
    assert ".dfd" in (static / "widget.css").read_text(encoding="utf-8")


def test_traits(widget: DataFrameDiff) -> None:
    assert (widget.left_short, widget.right_short) == ("C", "E")
    assert widget.summary["mismatch"] == 2
    assert widget.summary["found"] == 4
    assert widget.summary["keys"] == ["id"]
    city = next(c for c in widget.columns if c["name"] == "city")
    assert city == {
        "name": "city",
        "kind": "compared",
        "dtype": "String",
        "numeric": False,
        "ftype": "string",
        "mismatches": 1,
    }


def test_query_message(widget: DataFrameDiff) -> None:
    widget._handle_message(widget, {"type": "query", "req": 7, "statuses": ["mismatch"]}, [])
    content, buffers = sent(widget).messages[-1]
    assert buffers is None
    assert content["type"] == "page"
    assert content["req"] == 7
    assert content["filtered"] == 2
    assert content["page_size"] == 2
    assert [r["s"] for r in content["rows"]] == ["mismatch", "mismatch"]


def test_view_frame_follows_last_query(widget: DataFrameDiff) -> None:
    widget._handle_message(widget, {"type": "query", "req": 1, "search": "hamburg"}, [])
    assert widget.view_frame()["id"].to_list() == [4]


def test_export_message(widget: DataFrameDiff) -> None:
    widget._handle_message(widget, {"type": "export", "req": "export"}, [])
    content, buffers = sent(widget).messages[-1]
    assert content["type"] == "export"
    assert buffers is not None
    assert buffers[0].decode().startswith("id,status,differing")


def test_errors_are_reported_not_raised(
    widget: DataFrameDiff, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object) -> None:
        raise RuntimeError("kaputt")

    monkeypatch.setattr(widget._engine, "page", boom)
    widget._handle_message(widget, {"type": "query", "req": 3}, [])
    content, _ = sent(widget).messages[-1]
    assert content == {"type": "error", "req": 3, "message": "RuntimeError: kaputt"}


def test_unknown_messages_are_ignored(widget: DataFrameDiff) -> None:
    widget._handle_message(widget, {"type": "nope"}, [])
    widget._handle_message(widget, "not a mapping", [])
    assert sent(widget).messages == []


def test_selected_frame(widget: DataFrameDiff) -> None:
    widget.selected_ids = [3, 0, 3]
    assert widget.selected_frame()["id"].to_list() == [1, 4]


def test_from_result(left: pl.DataFrame, right: pl.DataFrame) -> None:
    result = compare(left, right, "id")
    w = DataFrameDiff.from_result(result, left_short="A", right_short="B")
    assert w.result is result
    assert (w.left_short, w.right_short) == ("A", "B")


@pytest.mark.parametrize(
    ("names", "overrides", "expected"),
    [
        (("CRM", "ERP"), (None, None), ("C", "E")),
        (("Prod", "Preview"), (None, None), ("L", "R")),
        (("Prod", "Preview"), ("PR", "PV"), ("PR", "PV")),
        (("", "x"), (None, None), ("L", "X")),
        (("left", "right"), (None, None), ("L", "R")),
    ],
)
def test_short_names(
    names: tuple[str, str], overrides: tuple[str | None, str | None], expected: tuple[str, str]
) -> None:
    assert short_names(*names, *overrides) == expected


def test_widget_passes_strict_on(left: pl.DataFrame, right: pl.DataFrame) -> None:
    import polars as pl

    narrow = left.with_columns(pl.col("id").cast(pl.Int32))
    with pytest.raises(ValueError, match="strict=False"):
        DataFrameDiff(narrow, right, key="id")
    assert DataFrameDiff(narrow, right, key="id", strict=False).summary["equal"] == 2


def test_default_page_size(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert DataFrameDiff(left, right, key="id").page_size == 10


def test_text_diff_default_and_option(left: pl.DataFrame, right: pl.DataFrame) -> None:
    assert DataFrameDiff(left, right, key="id").text_diff is True
    assert DataFrameDiff(left, right, key="id", text_diff=False).text_diff is False
    result = compare(left, right, key="id")
    assert DataFrameDiff.from_result(result, text_diff=False).text_diff is False
