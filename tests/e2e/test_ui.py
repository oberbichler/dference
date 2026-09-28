"""End-to-end tests of the widget in a real browser (see ``conftest.py``)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from dference import DataFrameDiff

from .conftest import Harness, css

if TYPE_CHECKING:
    from playwright.sync_api import Page

ALL = list(range(1, 45))
FOUND = list(range(1, 37))
MISMATCH = list(range(1, 14))


def toolbar(ui: Harness) -> str:
    """Pills in the toolbar as one line of text."""
    return " ".join(ui.page.locator(".dfd-toolbar-dyn").inner_text().split())


def status_menu(ui: Harness) -> None:
    ui.click('.dfd-scroll [data-colmenu="status"]')
    assert ui.page.locator(".dfd-colmenu.status").is_visible()


def all_keys(ui: Harness) -> list[int]:
    """Keys of every row in the current view, paging through it."""
    keys = ui.keys()
    while ui.page.locator('[data-page="next"]').is_enabled():
        ui.click('[data-page="next"]')
        keys += ui.keys()
    ui.click('[data-page="first"]')
    return keys


def show_all_rows(ui: Harness) -> None:
    ui.page.select_option('[data-act="page-size"]', "100")
    ui.settle()


# ---- initial render -----------------------------------------------------------


def test_first_page(ui: Harness) -> None:
    assert ui.rows.count() == 10
    assert ui.keys() == list(range(1, 11))
    assert ui.footer() == "1–10 of 44"
    assert "Page 1 of 5" in ui.page.locator(".dfd-pager").inner_text()
    stats = ui.page.locator(".dfd-stats").first.inner_text()
    for text in ("Equal", "23", "Mismatch", "13", "Only in ERP", "Only in CRM"):
        assert text in stats
    ui.screenshot("01-first-page")


def test_toolbar_has_no_status_button(ui: Harness) -> None:
    assert ui.page.locator(".dfd-toolbar .dfd-status-filter, .dfd-toolbar details").count() == 0


# ---- status filter in the column header ------------------------------------------


def test_status_menu_opens_from_the_header(ui: Harness) -> None:
    status_menu(ui)
    menu = ui.page.locator(".dfd-colmenu.status")
    assert menu.locator("[data-status]").count() == 4
    assert menu.locator("input[data-status]:checked").count() == 4
    text = menu.inner_text()
    assert "Only in CRM" in text
    assert "Only in ERP" in text
    ui.screenshot("02-status-menu")


def test_unchecking_a_status_filters_and_shows_a_pill(ui: Harness) -> None:
    status_menu(ui)
    ui.page.locator('input[data-status="equal"]').uncheck()
    ui.settle()
    show_all_rows(ui)
    assert sorted(ui.keys()) == [*MISMATCH, *range(37, 45)]
    # menu stays open and in sync, button marked active, pill in the toolbar
    assert ui.page.locator(".dfd-colmenu.status").is_visible()
    assert not ui.page.locator('input[data-status="equal"]').is_checked()
    assert "on" in (
        ui.page.locator('.dfd-colbtn[data-colmenu="status"]').get_attribute("class") or ""
    )
    pill = ui.page.locator('.dfd-toolbar-dyn [data-colmenu="status"]')
    assert pill.inner_text() == "Status All differences"


def test_presets_and_only(ui: Harness) -> None:
    status_menu(ui)
    ui.click('[data-preset="missing"]')
    assert ui.footer().startswith("1–8 of 8")
    checked = ui.page.locator("input[data-status]:checked")
    assert checked.count() == 2
    ui.page.locator('.dfd-opt:has([data-status="missing_left"])').hover()
    ui.click('[data-only="missing_left"]')
    assert ui.keys() == [41, 42, 43, 44]
    assert ui.page.locator("input[data-status]:checked").count() == 1
    ui.click('[data-preset="all"]')
    assert ui.footer() == "1–10 of 44"
    assert ui.page.locator('.dfd-toolbar-dyn [data-colmenu="status"]').count() == 0


def test_pill_reopens_the_menu_and_clears_the_filter(ui: Harness) -> None:
    status_menu(ui)
    ui.click('[data-preset="diff"]')
    ui.page.keyboard.press("Escape")
    assert ui.page.locator(".dfd-colmenu").count() == 0
    ui.click('.dfd-toolbar-dyn [data-colmenu="status"]')
    assert ui.page.locator(".dfd-colmenu.status").is_visible()
    ui.page.keyboard.press("Escape")
    ui.click('[data-act="clear-status"]')
    assert ui.footer() == "1–10 of 44"


def test_status_menu_closes_on_outside_click_and_escape(ui: Harness) -> None:
    status_menu(ui)
    ui.page.mouse.click(5, 880)
    assert ui.page.locator(".dfd-colmenu").count() == 0
    status_menu(ui)
    ui.page.keyboard.press("Escape")
    assert ui.page.locator(".dfd-colmenu").count() == 0
    # the header button toggles
    status_menu(ui)
    ui.click('.dfd-scroll [data-colmenu="status"]')
    assert ui.page.locator(".dfd-colmenu").count() == 0


def test_only_one_header_menu_is_open(ui: Harness) -> None:
    status_menu(ui)
    ui.click(f'.dfd-scroll [data-colmenu="{ui.col("city")}"]')
    assert ui.page.locator(".dfd-colmenu").count() == 1
    assert ui.page.locator(".dfd-colmenu.status").count() == 0
    status_menu(ui)
    assert ui.page.locator(".dfd-colmenu").count() == 1


def test_sort_by_status_from_the_menu(ui: Harness) -> None:
    status_menu(ui)
    ui.click('.dfd-colmenu [data-cm="desc"]')
    assert ui.page.locator(".dfd-colmenu").count() == 0
    statuses = ui.page.eval_on_selector_all(
        ".dfd-scroll tbody tr[data-id]", "trs => trs.map(tr => tr.className.match(/st-(\\w+)/)[1])"
    )
    # left before right: "only in ERP" (missing_left) sorts last, so first when descending
    assert statuses == ["missing_left"] * 4 + ["missing_right"] * 4 + ["mismatch"] * 2
    assert "↓" in ui.page.locator('.dfd-sort[data-sort="status"]').inner_text()
    status_menu(ui)
    ui.click('.dfd-colmenu [data-cm="unsort"]')
    assert ui.keys() == list(range(1, 11))


# ---- sorting, column filters, search ----------------------------------------------


def test_header_click_cycles_sort(ui: Harness) -> None:
    sort = f'.dfd-sort[data-sort="{ui.col("amount")}"]'
    ui.click(sort)
    assert ui.keys()[:3] == [1, 2, 3]
    ui.click(sort)
    assert ui.keys()[:2] == [44, 43]  # right-only rows carry the largest amounts
    ui.click(sort)
    assert ui.keys() == list(range(1, 11))


def test_column_filter_matches_left_or_right(ui: Harness) -> None:
    city = ui.col("city")
    for value, expected in (("Hamburg", [1, 5]), ("Munich", [4, 8])):
        ui.click(f'.dfd-scroll [data-colmenu="{city}"]')
        ui.page.select_option('.dfd-colmenu select[data-f="op"]', "equals")
        ui.page.fill('.dfd-colmenu input[data-f="a"]', value)
        ui.page.keyboard.press("Enter")  # Enter applies
        ui.settle()
        show_all_rows(ui)
        keys = ui.keys()
        assert set(expected) <= set(keys), (value, keys)
        for text in ui.column_texts(city):
            assert value in text
        assert f"city equals “{value}”" in toolbar(ui)
    ui.click(f'[data-rmfilter="{city}"]')
    assert ui.footer().endswith("of 44")


def test_number_range_filter(ui: Harness) -> None:
    amount = ui.col("amount")
    ui.click(f'.dfd-scroll [data-colmenu="{amount}"]')
    ui.page.select_option('.dfd-colmenu select[data-f="op"]', "between")
    ui.page.fill('.dfd-colmenu input[data-f="a"]', "100")
    ui.page.fill('.dfd-colmenu input[data-f="b"]', "120")
    ui.click('.dfd-colmenu [data-cm="apply"]')
    # 100.5 (right side of id 10) and 110.5, 120.5 differ; 110/120 left
    assert ui.keys() == [10, 11, 12]


def test_search_is_debounced_and_covers_both_sides(ui: Harness) -> None:
    ui.page.fill(".dfd-search input", "CUSTOMER 3")
    ui.page.wait_for_timeout(300)
    ui.settle()
    assert ui.keys() == list(range(30, 40))
    ui.page.fill(".dfd-search input", "web")  # right-only column
    ui.page.wait_for_timeout(300)
    ui.settle()
    assert ui.footer() == "1–10 of 40 (filtered from 44)"


# ---- quick filters ≠ / = -----------------------------------------------------------


def test_quick_filter_differs(ui: Harness) -> None:
    city = ui.col("city")
    ui.click(f'.dfd-delta[data-diffcol="{city}"]:not([data-eq])')
    assert ui.keys() == list(range(1, 9))
    assert "Differs in city" in toolbar(ui)


def test_quick_filter_equal_and_switching(ui: Harness) -> None:
    city = ui.col("city")
    eq = f'.dfd-delta[data-diffcol="{city}"][data-eq]'
    ne = f'.dfd-delta[data-diffcol="{city}"]:not([data-eq])'
    ui.click(eq)
    show_all_rows(ui)
    assert ui.keys() == list(range(9, 37))  # found rows only, city equal
    assert "Equal in city" in toolbar(ui)
    assert "on" in (ui.page.locator(eq).get_attribute("class") or "")
    ui.click(ne)  # the other symbol switches
    assert ui.keys() == list(range(1, 9))
    ui.click(ne)  # the active one turns it off
    assert ui.footer().endswith("of 44")
    ui.click(eq)
    ui.click('[data-act="clear-colfilter"]')
    assert ui.footer().endswith("of 44")


def test_equal_share_of_zero_is_not_a_button(ui: Harness) -> None:
    # "active" never differs: 100 % = is a button, 0 % ≠ is plain text
    active = ui.col("active")
    assert ui.page.locator(f'.dfd-delta[data-diffcol="{active}"][data-eq]').count() == 1
    assert ui.page.locator(f'.dfd-delta[data-diffcol="{active}"]:not([data-eq])').count() == 0


def test_only_differing_columns(ui: Harness) -> None:
    ui.page.check('[data-act="only-diff"]')
    headers = ui.page.locator(".dfd-scroll thead .dfd-sort").all_inner_texts()
    names = [h.rstrip("↑↓") for h in headers]
    # columns that differ anywhere in the view (not just on this page) stay
    assert names == ["Status", "id", "name", "city", "amount", "note", "channel"]
    # narrowing the view narrows the columns
    ui.click(f'.dfd-delta[data-diffcol="{ui.col("city")}"]:not([data-eq])')
    headers = ui.page.locator(".dfd-scroll thead .dfd-sort").all_inner_texts()
    assert [h.rstrip("↑↓") for h in headers] == [
        "Status",
        "id",
        "city",
        "amount",
        "note",
        "channel",
    ]


# ---- paging, fixed heights --------------------------------------------------------


def test_paging_and_page_size(ui: Harness) -> None:
    ui.click('[data-page="next"]')
    assert ui.keys() == list(range(11, 21))
    ui.click('[data-page="last"]')
    assert ui.footer() == "41–44 of 44"
    ui.click('[data-page="prev"]')
    assert ui.footer() == "31–40 of 44"
    ui.page.select_option('[data-act="page-size"]', "25")
    ui.settle()
    assert ui.footer() == "1–25 of 44"  # a new page size starts at page 1


def test_all_rows_have_the_same_height(ui: Harness) -> None:
    show_all_rows(ui)
    heights = ui.page.eval_on_selector_all(
        ".dfd-scroll tbody tr[data-id]", "trs => trs.map(tr => tr.getBoundingClientRect().height)"
    )
    assert len(heights) == 44
    assert set(heights) == {48}


def next_frames(page: Page) -> None:
    """Let the browser render two frames."""
    page.evaluate("() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")


def table_height(page: Page) -> float:
    return page.locator(".dfd-scroll").evaluate("el => el.getBoundingClientRect().height")


def test_table_height_does_not_depend_on_the_rows(ui: Harness) -> None:
    full = table_height(ui.page)
    head = ui.page.locator(".dfd-scroll thead").evaluate("el => el.getBoundingClientRect().height")
    assert full == pytest.approx(head + 10 * 48 + 2, abs=1)  # + frame
    ui.click('[data-page="last"]')  # 4 rows
    assert table_height(ui.page) == full
    ui.page.fill(".dfd-search input", "no such value")
    ui.page.wait_for_timeout(300)
    ui.settle()
    assert "No rows match" in ui.page.locator(".dfd-scroll").inner_text()
    assert table_height(ui.page) == full
    ui.screenshot("03-no-rows")
    # no inner scrollbar: the page fits exactly
    assert ui.page.locator(".dfd-scroll").evaluate("el => el.scrollHeight <= el.clientHeight")


def test_table_height_follows_the_page_size(ui: Harness) -> None:
    ten = table_height(ui.page)
    ui.page.select_option('[data-act="page-size"]', "25")
    ui.settle()
    assert table_height(ui.page) == pytest.approx(ten + 15 * 48, abs=1)


def test_last_row_line(ui: Harness) -> None:
    last = ".dfd-scroll tbody tr[data-id]:last-child td.cb"
    # a full page ends on the frame: no second line
    assert css(ui.page.locator(last), "border-bottom-width") == "0px"
    ui.click('[data-page="last"]')
    assert css(ui.page.locator(last), "border-bottom-width") == "1px"


# ---- cells, colours, chips ------------------------------------------------------------


def test_difference_cells_show_both_sides_with_chips(ui: Harness) -> None:
    cell = ui.page.locator(".dfd-scroll tbody tr[data-id]").first.locator("td.diff").first
    assert cell.locator(".dfd-val").count() == 2
    assert cell.locator(".dfd-side").all_inner_texts() == ["C", "E"]
    assert "CRM: " in (cell.get_attribute("title") or "")


def test_side_colours_are_consistent(ui: Harness) -> None:
    page = ui.page
    chip_l = css(page.locator(".dfd-side.l"), "background-color")
    chip_r = css(page.locator(".dfd-side.r"), "background-color")
    assert chip_l != chip_r
    # a row that exists only in CRM (left) wears the left colour and chip
    only_left = page.locator(".dfd-stat .dfd-badge.st-missing_right")
    only_right = page.locator(".dfd-stat .dfd-badge.st-missing_left")
    assert css(only_left, "background-color") == chip_l
    assert css(only_right, "background-color") == chip_r
    assert only_left.inner_text() == "C"
    assert only_right.inner_text() == "E"
    # one-sided column tags, overview line and detail view use the same chips
    assert page.locator(".dfd-tag .dfd-side.l").count() == 1
    assert page.locator(".dfd-tag .dfd-side.r").count() == 1
    assert page.locator(".dfd-meta .dfd-side").count() == 2


def test_difference_frames_are_uniform_and_shared(ui: Harness) -> None:
    city = ui.col("city")
    cells = ui.page.locator(f".dfd-scroll tbody tr[data-id] td:nth-child({city + 3})")
    first, second = cells.nth(0), cells.nth(1)
    amber = css(first, "--st-mismatch").strip()
    # the grey grid line between two differences takes the frame colour
    assert css(first, "border-bottom-color") == css(second, "border-bottom-color")
    assert css(first, "border-bottom-width") == "1px"
    assert css(first, "box-shadow").count("inset") == 2  # top (first row) and right
    assert css(second, "box-shadow").count("inset") == 1  # right only
    assert amber.lower() in ("#e69f00", "rgb(230, 159, 0)")


def test_single_values_are_vertically_centred(ui: Harness) -> None:
    td = ui.page.locator(".dfd-scroll tbody tr[data-id]").first.locator("td").nth(3)
    assert css(td, "vertical-align") == "middle"


def test_invisible_characters(ui: Harness) -> None:
    ui.page.fill(".dfd-search input", "customer 13")
    ui.page.wait_for_timeout(300)
    ui.settle()
    cell = ui.page.locator(".dfd-scroll tbody tr[data-id] td.diff").first
    assert "␣" in cell.inner_text()
    assert "only in invisible characters" in (cell.get_attribute("title") or "")
    ui.page.uncheck('[data-act="show-inv"]')
    assert "␣" not in cell.inner_text()


# ---- detail view -----------------------------------------------------------------------


def test_detail_view_and_stepping_across_pages(ui: Harness) -> None:
    ui.click('.dfd-scroll tbody tr[data-id="9"] td.status')
    detail = ui.page.locator(".dfd-detail")
    assert detail.is_visible()
    assert "10 / 44" in detail.inner_text()
    heads = detail.locator("thead th").all_inner_texts()
    assert heads[1:3] == ["C\nCRM (left)", "E\nERP (right)"] or "CRM (left)" in heads[1]
    assert detail.locator(".dfd-side.l").count() == 1
    ui.screenshot("04-detail")
    ui.click('[data-detail="next"]')  # row 11 is on page 2
    assert "Page 2 of 5" in ui.page.locator(".dfd-pager").first.inner_text()
    assert "11 / 44" in detail.inner_text()
    ui.click('[data-detail="prev"]')
    assert "Page 1 of 5" in ui.page.locator(".dfd-footer .dfd-pager").inner_text()
    ui.click('[data-act="close-detail"]')
    assert ui.page.locator(".dfd-detail").count() == 0


def test_detail_shows_deltas_and_one_sided_rows(ui: Harness) -> None:
    ui.click('.dfd-scroll tbody tr[data-id="9"] td.status')  # id 5: city and amount
    ui.page.locator(".dfd-scroll tbody tr[data-id]").nth(4).locator("td.status").click()
    ui.settle()
    text = ui.page.locator(".dfd-detail").inner_text()
    assert "Δ" in text
    ui.click('[data-page="last"]')
    ui.page.locator(".dfd-scroll tbody tr.st-missing_left").first.locator("td.status").click()
    ui.settle()
    assert "Only in ERP" in ui.page.locator(".dfd-detail header").inner_text()


# ---- selection, export, errors -------------------------------------------------------


def test_selection_syncs_to_python_and_back(ui: Harness) -> None:
    ui.page.locator(".dfd-scroll tbody tr[data-id]").nth(1).locator("input[data-sel]").check()
    ui.settle()
    row_id = int(
        ui.page.locator(".dfd-scroll tbody tr[data-id]").nth(1).get_attribute("data-id") or -1
    )
    assert ui.widget.selected_ids == [row_id]
    assert ui.widget.selected_frame()["id"].to_list() == [2]
    assert "1 selected" in toolbar(ui)
    ui.page.check('[data-act="page-all"]')
    assert len(ui.widget.selected_ids) == 10
    # Python changes the selection: the checkboxes follow
    ui.set_trait("selected_ids", [0])
    checked = ui.page.locator(".dfd-scroll tbody input[data-sel]:checked")
    assert checked.count() == 1
    ui.click('[data-act="clear-sel"]')
    assert ui.widget.selected_ids == []


def test_csv_export_uses_the_current_filters(ui: Harness) -> None:
    status_menu(ui)
    ui.click('[data-preset="missing"]')
    ui.page.keyboard.press("Escape")
    with ui.page.expect_download() as info:
        ui.page.click('[data-act="export"]')
    download = info.value
    assert download.suggested_filename == "dataframe-diff.csv"
    lines = Path(download.path()).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 8
    assert lines[0].startswith("id,")
    ui.settle()
    assert ui.page.locator('[data-act="export"]').is_enabled()


def test_view_frame_follows_the_widget(ui: Harness) -> None:
    ui.click(f'.dfd-delta[data-diffcol="{ui.col("amount")}"]:not([data-eq])')
    assert ui.widget.view_frame()["id"].to_list() == list(range(5, 13))


def test_python_errors_are_shown_not_thrown(ui: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(ui.widget._engine, "page", boom)
    ui.click('[data-page="next"]')
    box = ui.page.locator(".dfd-error")
    assert box.is_visible()
    assert "engine exploded" in box.inner_text()


def test_new_result_from_python_resets_the_view(ui: Harness) -> None:
    status_menu(ui)
    ui.click('[data-preset="missing"]')
    ui.page.keyboard.press("Escape")
    ui.set_trait("left_name", "Shop")
    assert re.search(r"Only in Shop", ui.page.locator(".dfd-stats").first.inner_text())


def test_narrow_viewport_scrolls_sideways_without_a_vertical_scrollbar(ui: Harness) -> None:
    ui.page.set_viewport_size({"width": 700, "height": 900})
    ui.click('[data-page="next"]')  # re-render at the new width
    scroll = ui.page.locator(".dfd-scroll")
    assert scroll.evaluate("el => el.scrollWidth > el.clientWidth")
    assert scroll.evaluate("el => el.scrollHeight <= el.clientHeight")
    ui.screenshot("05-narrow")


def test_height_is_fixed_once_a_hidden_widget_is_shown(ui: Harness) -> None:
    ten = table_height(ui.page)
    ui.page.evaluate("document.getElementById('app').style.display = 'none'")
    next_frames(ui.page)  # a resize observer only reports sizes it saw in a frame
    # renders while hidden (Playwright does not operate hidden controls)
    ui.page.evaluate(
        """() => {
            const sel = document.querySelector('[data-act="page-size"]');
            sel.value = "25";
            sel.dispatchEvent(new Event("change", { bubbles: true }));
        }"""
    )
    ui.settle()
    assert ui.rows.count() == 25
    ui.page.evaluate("document.getElementById('app').style.display = ''")
    # the resize observer measures in the next frame
    expected = ten + 15 * 48
    for _ in range(50):
        if table_height(ui.page) == pytest.approx(expected, abs=1):
            break
        ui.page.wait_for_timeout(20)
    assert table_height(ui.page) == pytest.approx(expected, abs=1)


def test_large_integers_keep_all_digits(page: Page) -> None:
    left = pl.DataFrame({"id": [1], "v": [2**62]})
    right = pl.DataFrame({"id": [1], "v": [2**62 + 1]})
    ui = Harness(page, DataFrameDiff(left, right, key="id")).mount()
    cell = ui.page.locator(".dfd-scroll td.diff")
    assert cell.locator(".dfd-v").all_inner_texts() == [
        "4611686018427387904",
        "4611686018427387905",
    ]
    # the delta is exact too (BigInt), not lost because the values arrive as text
    assert "Δ +1" in (cell.get_attribute("title") or "")


def test_quick_filter_column_stays_visible_with_only_differing_columns(ui: Harness) -> None:
    ui.page.check('[data-act="only-diff"]')
    city = ui.col("city")
    ui.click(f'.dfd-delta[data-diffcol="{city}"][data-eq]')
    # the view has no difference in "city" by construction, but it is the filtered column
    headers = [
        h.rstrip("↑↓") for h in ui.page.locator(".dfd-scroll thead .dfd-sort").all_inner_texts()
    ]
    assert "city" in headers
    assert ui.page.locator(f'.dfd-delta.on[data-diffcol="{city}"]').count() == 1
    # so is a column with a filter
    active = ui.col("active")
    ui.click('[data-act="clear-colfilter"]')
    ui.page.uncheck('[data-act="only-diff"]')
    ui.click(f'.dfd-scroll [data-colmenu="{active}"]')
    ui.page.select_option('.dfd-colmenu select[data-f="op"]', "true")
    ui.click('.dfd-colmenu [data-cm="apply"]')
    ui.page.check('[data-act="only-diff"]')
    headers = [
        h.rstrip("↑↓") for h in ui.page.locator(".dfd-scroll thead .dfd-sort").all_inner_texts()
    ]
    assert "active" in headers


def test_status_order_is_left_before_right(ui: Harness) -> None:
    labels = (
        ui.page.locator(".dfd-stats")
        .first.locator(".dfd-stat span:not(.dfd-badge)")
        .all_inner_texts()
    )
    assert labels == ["Equal", "Mismatch", "Only in CRM", "Only in ERP"]
    status_menu(ui)
    menu = ui.page.locator(".dfd-colmenu [data-status]")
    assert [menu.nth(i).get_attribute("data-status") for i in range(4)] == [
        "equal",
        "mismatch",
        "missing_right",
        "missing_left",
    ]


def test_small_diff_is_only_as_high_as_its_rows(page: Page) -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "v": [1, 2, 3]})
    right = pl.DataFrame({"id": [1, 2, 4], "v": [1, 5, 4]})
    ui = Harness(page, DataFrameDiff(left, right, key="id", page_size=25)).mount()
    head = ui.page.locator(".dfd-scroll thead").evaluate("el => el.getBoundingClientRect().height")
    full = table_height(ui.page)
    assert full == pytest.approx(head + 4 * 48 + 2, abs=1)  # 4 rows in total, not 25
    last = ".dfd-scroll tbody tr[data-id]:last-child td.cb"
    assert css(ui.page.locator(last), "border-bottom-width") == "0px"  # the rows fill the box
    # filtering still does not change the height
    status_menu(ui)
    ui.page.locator('.dfd-opt:has([data-status="equal"])').hover()
    ui.click('[data-only="equal"]')
    assert ui.rows.count() == 1
    assert table_height(ui.page) == full
    assert css(ui.page.locator(last), "border-bottom-width") == "1px"


def test_empty_diff_keeps_room_for_the_message(page: Page) -> None:
    empty = pl.DataFrame({"id": [], "v": []}, schema={"id": pl.Int64, "v": pl.Int64})
    ui = Harness(page, DataFrameDiff(empty, empty, key="id")).mount()
    assert "No rows match" in ui.page.locator(".dfd-scroll").inner_text()
    assert ui.page.locator(".dfd-scroll").evaluate("el => el.scrollHeight <= el.clientHeight")
