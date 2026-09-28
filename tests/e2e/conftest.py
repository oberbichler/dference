"""Browser tests of the widget: the real ``widget.js``/``widget.css`` against a real
:class:`~dference.DataFrameDiff`, without marimo or Jupyter.

A small page stands in for anywidget: its model keeps the synced traits, and
``model.send`` / ``model.save_changes`` call into Python through Playwright's
``expose_function``. Python's answers are delivered back as ``msg:custom``
events, exactly like anywidget does.

Needs the ``e2e`` dependency group and a browser: ``uv run playwright install
chromium``. Without them the tests are skipped.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from dference import DataFrameDiff

sync_api = pytest.importorskip("playwright.sync_api")

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Locator, Page

STATIC = files("dference") / "static"
ORIGIN = "http://dference.test"
TRAITS = (
    "columns",
    "summary",
    "left_name",
    "right_name",
    "left_short",
    "right_short",
    "page_size",
    "selected_ids",
)

# The page: stand-in for anywidget's model around the real widget module.
INDEX = """<!doctype html>
<html><head><meta charset="utf-8"><link rel="stylesheet" href="widget.css">
<style>body { margin: 16px; background: #fff; color: #111; font: 13px sans-serif; }</style>
</head><body><div id="app"></div>
<script type="module">
import widget from "./widget.js";
const traits = await window.pyTraits();
const listeners = new Map();
let pending = {};
const emit = (event, ...args) => (listeners.get(event) || []).slice().forEach((cb) => cb(...args));
const toView = (b64) => {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new DataView(bytes.buffer);
};
const api = {
  inflight: 0,
  saved: [],
  emit,
  deliver(out) {
    for (const [content, buffers] of out) emit("msg:custom", content, buffers.map(toView));
  },
  setTrait(name, value) {
    traits[name] = value;
    emit(`change:${name}`);
  },
};
const model = {
  get: (name) => traits[name],
  set: (name, value) => { traits[name] = value; pending[name] = value; },
  save_changes() {
    const changes = pending;
    pending = {};
    api.saved.push(changes);
    window.pySave(changes);
  },
  on(event, cb) { listeners.set(event, [...(listeners.get(event) || []), cb]); },
  off(event, cb) { listeners.set(event, (listeners.get(event) || []).filter((x) => x !== cb)); },
  async send(msg) {
    api.inflight++;
    try { api.deliver(await window.pySend(msg)); } finally { api.inflight--; }
  },
};
window.__dfd = api;
widget.render({ model, el: document.getElementById("app") });
window.__dfdReady = true;
</script></body></html>"""


def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """44 keys covering every status on more than one page (page size 10).

    * ids 1-8: ``city`` differs, ids 5-12: ``amount`` differs (5-8 both)
    * id 13: ``name`` differs by a trailing space only
    * ids 37-40 only left (missing right), ids 41-44 only right (missing left)
    * all other ids 14-36 are equal
    """
    n_left = list(range(1, 41))
    n_right = list(range(1, 37)) + list(range(41, 45))
    cities = ["Berlin", "Munich", "Hamburg", "Cologne"]

    def side(ids: list[int], *, right: bool) -> pl.DataFrame:
        rows = []
        for i in ids:
            city = cities[i % 4]
            amount = float(i * 10)
            name = f"Customer {i:02d}"
            if right and i <= 8:
                city = cities[(i + 1) % 4]
            if right and 5 <= i <= 12:
                amount += 0.5
            if right and i == 13:
                name += " "
            rows.append(
                {
                    "id": i,
                    "name": name,
                    "city": city,
                    "amount": amount,
                    "active": i % 3 != 0,
                    "day": dt.date(2026, 1, 1) + dt.timedelta(days=i),
                }
            )
        df = pl.DataFrame(rows)
        if right:
            return df.with_columns(channel=pl.lit("web"))
        return df.with_columns(note=pl.format("n{}", pl.col("id")))

    return side(n_left, right=False), side(n_right, right=True)


class Harness:
    """A mounted widget: the Playwright page plus the Python widget behind it."""

    def __init__(self, page: Page, widget: DataFrameDiff) -> None:
        self.page = page
        self.widget = widget
        self._outbox: list[tuple[Any, list[bytes] | None]] = []
        # everything the widget sends ends up in the outbox, which pySend returns
        widget.send = self._collect  # ty: ignore[invalid-assignment]

    # ---- Python side of the fake anywidget model ---------------------------

    def _collect(self, content: Any, buffers: list[bytes] | None = None) -> None:
        self._outbox.append((content, buffers))

    def _traits(self) -> dict[str, Any]:
        return {name: getattr(self.widget, name) for name in TRAITS}

    def _send(self, msg: Any) -> list[list[Any]]:
        self._outbox = []
        self.widget._handle_message(self.widget, msg, [])
        return [
            [content, [base64.b64encode(bytes(b)).decode() for b in buffers or []]]
            for content, buffers in self._outbox
        ]

    def _save(self, changes: dict[str, Any]) -> None:
        for name, value in changes.items():
            setattr(self.widget, name, value)

    def mount(self) -> Harness:
        assets = {"widget.js": "text/javascript", "widget.css": "text/css"}

        def serve(route: Any) -> None:
            name = route.request.url.removeprefix(f"{ORIGIN}/")
            if name in assets:
                route.fulfill(body=(STATIC / name).read_text(), content_type=assets[name])
            else:
                route.fulfill(body=INDEX, content_type="text/html")

        self.page.route(f"{ORIGIN}/**", serve)
        self.page.expose_function("pyTraits", self._traits)
        self.page.expose_function("pySend", self._send)
        self.page.expose_function("pySave", self._save)
        self.page.goto(f"{ORIGIN}/index.html")
        self.page.wait_for_function("window.__dfdReady === true")
        self.settle()
        return self

    # ---- helpers for tests -------------------------------------------------

    def settle(self) -> None:
        """Wait until no request is in flight and the table shows its result."""
        self.page.wait_for_function(
            "window.__dfd.inflight === 0 && !document.querySelector('.dfd.loading')"
        )

    def set_trait(self, name: str, value: Any) -> None:
        """Change a trait in Python and tell the frontend, like a comm update."""
        setattr(self.widget, name, value)
        self.page.evaluate("([n, v]) => window.__dfd.setTrait(n, v)", [name, value])

    @property
    def rows(self) -> Locator:
        return self.page.locator(".dfd-scroll tbody tr[data-id]")

    def keys(self) -> list[int]:
        """``id`` column of the visible rows, in display order."""
        idx = [c.name for c in self.widget.result.columns].index("id")
        return [int(v) for v in self.column_texts(idx)]

    def column_texts(self, col: int) -> list[str]:
        """Text of column ``col`` (index into ``result.columns``) per visible row."""
        # body cells: checkbox, status, then the columns in order
        return self.page.eval_on_selector_all(
            ".dfd-scroll tbody tr[data-id]",
            "(trs, i) => trs.map((tr) => tr.children[i + 2].textContent.trim())",
            col,
        )

    def col(self, name: str) -> int:
        return [c.name for c in self.widget.result.columns].index(name)

    def footer(self) -> str:
        return self.page.locator(".dfd-footer > span").first.inner_text()

    def click(self, selector: str) -> None:
        self.page.locator(selector).first.click()
        self.settle()

    def screenshot(self, name: str) -> None:
        """Save a screenshot if ``DFD_SCREENSHOTS`` names a directory."""
        target = os.environ.get("DFD_SCREENSHOTS")
        if target:
            Path(target).mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(Path(target) / f"{name}.png"), full_page=True)


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except sync_api.Error as exc:  # browser not installed
            pytest.skip(f"no Chromium for Playwright ({exc.message.splitlines()[0]})")
        yield b
        b.close()


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    context = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    pg = context.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    yield pg
    context.close()
    assert not errors, f"JavaScript errors: {errors}"


@pytest.fixture
def ui(page: Page) -> Harness:
    left, right = frames()
    widget = DataFrameDiff(left, right, key="id", left_name="CRM", right_name="ERP", page_size=10)
    return Harness(page, widget).mount()


def css(locator: Locator, prop: str) -> str:
    """Computed style ``prop`` of the first element of ``locator``."""
    return locator.first.evaluate("(el, p) => getComputedStyle(el).getPropertyValue(p)", prop)


def js(page: Page, expr: str) -> Any:
    return json.loads(page.evaluate(f"JSON.stringify({expr})"))
