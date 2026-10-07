# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-07

### Changed

- Table rows keep one height with multi-line text: a cell shows only the first
  line plus a `⋯` button that opens the full text in a flyover. For a
  difference the flyover shows both sides next to each other, with numbered
  lines and the differing lines highlighted.

## [0.1.0] - 2026-10-06

First release.

### Added

- `compare()` – match two DataFrames on a (composite) key with a single polars
  join and classify every row as equal, mismatch, only in left
  (`missing_right`) or only in right (`missing_left`).
- `DiffResult` with `summary`, `columns`, `frame()`, `mismatches()` and
  `column_stats()`.
- Exact comparison without tolerance; `null` and `NaN` count as equal missing
  values.
- Strict dtypes: `strict=True` (the default) requires identical dtypes on both
  sides; `strict=False` aligns lossless differences (integer width,
  `Float32`/`Float64`, decimal precision, time units within one time zone, text
  vs. `Categorical`/`Enum`, all-null columns). Anything else – including naive
  vs. time-zone-aware and different time zones – raises one error listing each
  column with a cast.
- Input support for polars (eager and lazy), pandas, pyarrow and objects with
  the Arrow PyCapsule interface or `to_polars()`.
- `DataFrameDiff` anywidget for marimo and Jupyter. Filtering, sorting and
  paging run in polars on the Python side; the browser only receives the
  visible page. It has:
  - a status overview on a shared scale, ordered left before right
  - a status filter in the header of the status column; an active filter shows
    as a removable pill in the toolbar
  - quick filters `≠` and `=` per column (only rows that differ in, or are
    equal in, that column) next to per-column match rates
  - column menus with sorting and type-aware filters, where a row matches if
    the left *or* the right value matches
  - one colour and one chip (the side's short name) per side, used for
    difference markers, statuses, one-sided columns, overview and detail view
  - a side-by-side detail view with numeric deltas, stepping across pages
  - invisible-character highlighting
  - search, CSV export of the filtered rows, row selection synced to Python
    (`selected_ids`, `selected_frame()`) and `view_frame()`
  - a fixed row height and a table as high as a full page (default 10 rows) or
    as all rows of a smaller diff, so filtering never makes it jump
  - exact display of integers beyond 2^53 and of decimals a float cannot hold
- Demo notebook that tours every feature.
- Tests: unit tests, property-based tests (Hypothesis) and browser tests of the
  widget (Playwright).
- Tooling: ruff and ty for Python, Biome and a strict JSDoc type check with
  TypeScript for the frontend; all run in CI.
- Release workflow, started by hand with the version: runs the full CI, builds
  and checks the distributions, publishes them to PyPI via trusted publishing
  (with attestations), then tags the commit and creates the GitHub release. The
  version comes from the git tag (hatch-vcs), so nothing needs to be bumped.
- ISC license.

[Unreleased]: https://github.com/oberbichler/dference/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/oberbichler/dference/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/oberbichler/dference/releases/tag/v0.1.0
