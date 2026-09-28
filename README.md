<!-- absolute URL: relative image paths do not render on PyPI -->
<p align="center">
  <img src="https://raw.githubusercontent.com/oberbichler/dference/main/docs/logo.svg" alt="dference" width="480">
</p>

<p align="center">
  <a href="https://pypi.org/project/dference/"><img src="https://img.shields.io/pypi/v/dference" alt="PyPI"></a>
  <a href="https://pypi.org/project/dference/"><img src="https://img.shields.io/pypi/pyversions/dference" alt="Python versions"></a>
  <a href="https://pypistats.org/packages/dference"><img src="https://img.shields.io/pypi/dm/dference" alt="Downloads"></a>
  <a href="https://github.com/oberbichler/dference/actions/workflows/ci.yml"><img src="https://github.com/oberbichler/dference/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/oberbichler/dference/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-ISC-blue" alt="License: ISC"></a>
  <br>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/astral-sh/ty"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" alt="ty"></a>
  <a href="https://molab.marimo.io/github/oberbichler/dference/blob/main/examples/demo.py"><img src="https://marimo.io/molab-shield.svg" alt="Open the demo in molab"></a>
</p>

**Compare two DataFrames by key and explore every difference – in an interactive
widget for [marimo](https://marimo.io) and Jupyter, or as plain polars tables.**

Rows are matched on a key (one or several columns) and classified as

| Status | Meaning |
| --- | --- |
| `equal` | key on both sides, all compared values equal |
| `mismatch` | key on both sides, at least one value differs |
| `missing_right` | key only in the left frame (shown as `Only in {left_name}`) |
| `missing_left` | key only in the right frame (shown as `Only in {right_name}`) |

The widget is modelled on `marimo.ui.table`: paging, sorting, search, a status
filter, marimo-style column filters, per-column match rates, a side-by-side
detail view, invisible-character highlighting and CSV export. It stays fast on
millions of rows because filtering, sorting and paging run in polars on the
Python side – the browser only ever receives the visible page.

## Installation

```bash
uv add dference            # or: pip install dference
uv add "dference[pandas]"  # if your inputs are pandas DataFrames
```

`dference` depends on `polars`, `anywidget` and `traitlets`. Inputs can be polars
(eager or lazy), pandas, pyarrow, or anything implementing the Arrow PyCapsule
interface or `to_polars()` (DuckDB relations, narwhals, …).

## Quick start

### marimo

```python
import marimo as mo
from dference import DataFrameDiff

diff = DataFrameDiff(crm, erp, key="customer_id", left_name="CRM", right_name="ERP")
view = mo.ui.anywidget(diff)
view
```

In another cell, work with the rows checked in the widget. The cell re-runs
when the selection changes – and only then; paging, sorting and filtering
never trigger re-execution:

```python
view.value["selected_ids"]  # reactive dependency
diff.selected_frame()  # the checked rows as a polars DataFrame
```

### Jupyter

```python
from dference import DataFrameDiff

DataFrameDiff(crm, erp, key=["region", "customer_id"])
```

### Without a UI

```python
import dference

result = dference.compare(crm, erp, key="customer_id", left_name="CRM", right_name="ERP")

result.summary  # Summary(equal=158, mismatch=78, missing_left=10, missing_right=8, ...)
result.frame()  # wide: key, status, differing, "<col> [CRM]", "<col> [ERP]", ...
result.frame("mismatch")
result.mismatches()  # long: one row per (key, column) that differs
result.column_stats()  # per column: dtypes, mismatches, equal/mismatch share
```

## How values are compared

- **Exact equality.** There is deliberately no numeric tolerance. If small float
  differences should not count, round both frames first, e.g.
  `df.with_columns(cs.float().round(2))`.
- **Missing values are equal to each other**: `null == null`, and in float
  columns `NaN` counts as missing too (pandas uses NaN as its null marker, so
  a frame from pandas and one from Arrow would otherwise disagree).
- **Same column, same dtype.** By default (`strict=True`) a column must have
  the identical dtype on both sides.
- **`strict=False`** aligns differences that lose nothing – the same values
  stored differently: integers of different width, `Float32` vs. `Float64`,
  decimals of different precision, `Datetime` / `Duration` in different time
  units (same time zone), text vs. `Categorical` / `Enum`, and an all-null
  column vs. anything. `ColumnInfo.compare_dtype` tells you which dtype was
  used.
- **Everything else is always your decision.** Int vs. float, `Date` vs. `Datetime`,
  naive vs. time-zone aware, different time zones, text vs. numbers, different
  nested types: `compare()` raises one `ValueError` that lists every such column
  (keys included) with a cast to fix it, e.g.
  `right = right.with_columns(pl.col("qty").cast(pl.Int64))`. Note that pandas
  has no date dtype – a `Date` column from pandas arrives as a `Datetime`.
- **Keys** must be unique on each side; null keys match null keys.
- **Columns present on one side only** are shown but not compared. Use
  `ignore_columns=[...]` to leave columns out entirely.

## Widget features

- Status overview on a shared scale (equal / mismatch / only left / only
  right, plus found vs. not found); filter by status from the menu in the
  status column header.
- Each side has one colour and one chip – the first letter of its name
  (`left_short=` / `right_short=` to override) – used everywhere: differences
  show both values with equal weight, each marked by its side's chip, and rows
  on one side only carry the chip of the side they exist on.
- Per-column match rate in the header; click `≠` or `=` to show only rows
  that differ in, or are equal in, that column.
- Column menu per header: sort and type-aware filters (text, number range,
  date range, boolean, null checks). For compared columns **a row matches if
  the left or the right value matches**; a side that does not exist in a row
  never matches.
- Invisible characters are made visible: `␣` leading/trailing/repeated spaces,
  `→` tab, `↵` line feed, `⍽` no-break spaces, `ZWSP`, `BOM`, … Cells that
  differ *only* in invisible characters are flagged.
- Side-by-side detail view with numeric deltas; step through the filtered rows
  across page boundaries.
- CSV export of all rows matching the current filters.

## Python API

| Object | Purpose |
| --- | --- |
| `compare(left, right, key, *, left_name, right_name, ignore_columns, strict=True)` | Run a comparison, returns `DiffResult`. |
| `DiffResult.summary` | `Summary` with counts per status plus `found`, `not_found`, `total`. |
| `DiffResult.columns` | Tuple of `ColumnInfo` (kind, dtypes, mismatches, compare dtype). |
| `DiffResult.frame(status=None, *, rows=None)` | Wide result, optionally filtered by status or row ids. |
| `DiffResult.mismatches()` | Long format of all differing cells (values as text). |
| `DiffResult.column_stats()` | Match rates per compared column. |
| `DataFrameDiff(left, right, key, ...)` | The widget; same arguments as `compare` plus `left_short`, `right_short`, `page_size` (default 10). |
| `DataFrameDiff.from_result(result, ...)` | Widget for an existing `DiffResult`. |
| `DataFrameDiff.result` | The underlying `DiffResult`. |
| `DataFrameDiff.selected_ids` | Synced trait with the checked row ids. |
| `DataFrameDiff.selected_frame()` | Checked rows as a wide frame. |
| `DataFrameDiff.view_frame()` | Rows matching the widget's current filters (not reactive). |
| `Status` | `StrEnum` of the four statuses. |

## Performance

The comparison is one hash join plus vectorised expressions; the widget keeps
the ordered row ids of the last few views in a small cache, so paging through
a view is a slice. Timings from `benchmarks/bench.py` on a **single CPU core**
(polars parallelises, so more cores are faster):

| 5 million rows per side | time |
| --- | ---: |
| `compare` (join, difference flags, statistics) | 2.5 s |
| first page / next page | < 1 ms |
| filter by status | 19 ms |
| numeric range filter (left or right) | 45 ms |
| sort by a column | 0.4 s |
| full-text search across all columns | 0.4 s |

Memory: the joined frame holds both sides once, plus one boolean flag per
compared column.

## Limitations

- The widget needs a running Python kernel; in a static HTML export it shows no
  rows.
- CSV export sends all matching rows to the browser in one message – filter
  first for very large exports, or use `view_frame().write_csv(...)`.
- `view_frame()` reflects the filters last sent by the widget but is not a
  reactive value in marimo.

## Development

```bash
git clone https://github.com/oberbichler/dference && cd dference
uv sync                                # creates .venv with all dev tools

uv run ruff format .                   # format
uv run ruff check .                    # lint
uv run ty check                        # type check
uv run pytest --cov                    # tests with coverage
uv run playwright install chromium     # once: browser for the widget tests
uv run pytest tests/e2e                # widget in a real browser (skipped without Chromium)

uv run marimo edit examples/demo.py    # interactive demo
uv run python benchmarks/bench.py 1_000_000
```

The frontend is a single dependency-free ES module
(`src/dference/static/widget.js`) with its stylesheet next to it – no build step.
anywidget reloads both on change while you develop.

Node.js is only needed for frontend tooling, never at runtime:

```bash
npm ci                                 # Biome + TypeScript (dev only)

npm run fix                            # format + auto-fix JS, CSS and JSON (Biome)
npm run check                          # lint (Biome) and type-check the JSDoc (tsc)
```

Types live in JSDoc comments and are checked with `tsc --checkJs` in strict
mode (see `tsconfig.json`); the widget API is typed via `@anywidget/types`.
Nothing is compiled – the file that ships is the file you edit.

### Releasing

The version is derived from the git tag (`v1.2.3`) by
[hatch-vcs](https://github.com/ofek/hatch-vcs) – there is no version to bump in
`pyproject.toml` or `uv.lock`. A checkout between releases reports a dev version
such as `0.1.1.dev3+g94570d5`.

For every release:

1. Rename `## [Unreleased]` in `CHANGELOG.md` to `## [x.y.z] - <date>` (add a new
   empty `Unreleased` section above it) and merge that to `main`.
2. Run *Actions → Release → Run workflow* on `main` with the version `x.y.z`, or
   `gh workflow run release.yml -f version=x.y.z`.

The workflow refuses versions that already exist as tag, release or on PyPI,
runs the full CI on the commit, builds with that version, publishes to PyPI via
[trusted publishing](https://docs.pypi.org/trusted-publishers/), and only then
tags the commit `vx.y.z` and creates the GitHub release with the changelog
section as notes and the wheel and sdist attached. If a step fails, *Re-run
failed jobs* continues from there.

Once, before the first release: on PyPI add a
[pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for the project `dference` – owner `oberbichler`, repository `dference`,
workflow `release.yml`, environment `pypi` – and create the environment `pypi`
in the GitHub repository settings (optionally with required reviewers).

## License

ISC © Thomas Oberbichler
