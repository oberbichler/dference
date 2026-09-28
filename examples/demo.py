"""dference demo notebook.

Run from a checkout with ``uv run marimo edit examples/demo.py``.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime as dt
    import time

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    import dference
    from dference import DataFrameDiff

    return DataFrameDiff, dference, dt, mo, np, pa, pd, pl, time


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # dference – a tour

    Two customer lists – from the **CRM** (left) and the **ERP** (right) – matched on a
    key. This notebook walks through every feature:

    1. **The widget** – explore the differences interactively
    2. **Continue in Python** – use the checked rows and the current view
    3. **Without a UI** – the same comparison as polars tables
    4. **How values are compared** – exact equality, missing values, mixed dtypes
    5. **Keys, names and markers** – composite keys, `left_short` / `right_short`
    6. **Inputs** – polars, lazy frames, pandas, pyarrow
    7. **Large data** – millions of rows
    """)
    return


@app.cell
def _(np, pd):
    _rng = np.random.default_rng(7)
    _n = 250
    _ids = np.arange(1000, 1000 + _n)
    _cities = ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Stuttgart", "Leipzig"]
    _tiers = ["Bronze", "Silver", "Gold"]

    crm = pd.DataFrame(
        {
            "customer_id": _ids,
            "region": _rng.choice(["North", "South", "West", "East"], _n),
            "name": [f"Customer {i}" for i in _ids],
            "city": _rng.choice(_cities, _n),
            "revenue": _rng.gamma(2.0, 5000, _n).round(2),
            "tier": _rng.choice(_tiers, _n),
            "active": _rng.random(_n) > 0.15,
            "last_contact": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(_rng.integers(0, 270, _n), unit="D"),
        }
    )
    erp = crm.copy()
    _p = _rng.permutation(_n)

    # rounding noise – disappears when rounding to 2 decimals
    erp.loc[_p[:30], "revenue"] += _rng.normal(0, 1e-6, 30)
    # real revenue differences
    erp.loc[_p[30:40], "revenue"] = (erp.loc[_p[30:40], "revenue"] * 1.1).round(2)
    # different city
    erp.loc[_p[40:52], "city"] = erp.loc[_p[40:52], "city"].map(
        lambda s: _cities[(_cities.index(s) + 1) % len(_cities)]
    )
    # tier upgrade
    erp.loc[_p[52:58], "tier"] = "Gold"
    # values missing in the ERP
    erp.loc[_p[58:62], "city"] = None
    # contact date shifted by one day
    erp.loc[_p[62:67], "last_contact"] += pd.Timedelta(days=1)
    # several columns at once
    erp.loc[_p[67:70], ["city", "tier"]] = ["Dresden", "Platinum"]
    erp.loc[_p[67:70], "active"] = ~erp.loc[_p[67:70], "active"]

    # invisible characters – they cause mismatches that are hard to spot
    erp.loc[_p[84:87], "name"] = erp.loc[_p[84:87], "name"] + " "  # trailing space
    erp.loc[_p[87:89], "name"] = erp.loc[_p[87:89], "name"].str.replace(
        " ", "\u00a0"
    )  # no-break space
    erp.loc[_p[89:91], "name"] = "\u200b" + erp.loc[_p[89:91], "name"]  # zero-width space
    erp.loc[_p[91:92], "region"] = erp.loc[_p[91:92], "region"] + "\t"  # tab
    crm.loc[_p[92:93], "city"] = ""  # empty string …
    erp.loc[_p[92:93], "city"] = None  # … vs. null
    crm.loc[_p[93:95], "name"] = crm.loc[_p[93:95], "name"].str.replace(" ", "  ")  # double space,
    erp.loc[_p[93:95], "name"] = erp.loc[_p[93:95], "name"].str.replace(
        " ", "  "
    )  # same on both sides

    # missing right (only in CRM) / missing left (only in ERP)
    erp = erp.drop(index=_p[70:78])
    crm = crm.drop(index=_p[78:84])
    _new = pd.DataFrame(
        {
            "customer_id": [2000, 2001, 2002, 2003],
            "region": ["North", "East", "South", "West"],
            "name": ["Customer 2000", "Customer 2001", "Customer 2002", "Customer 2003"],
            "city": ["Kiel", "Rostock", "Passau", "Aachen"],
            "revenue": [1200.0, 540.5, 9800.0, 77.0],
            "tier": ["Bronze", "Bronze", "Silver", "Bronze"],
            "active": [True, True, False, True],
            "last_contact": pd.to_datetime(
                ["2026-09-01", "2026-09-03", "2026-09-10", "2026-09-12"]
            ),
        }
    )
    erp = pd.concat([erp, _new], ignore_index=True)

    # a column that only exists in the ERP; shuffle row order
    erp["channel"] = _rng.choice(["Online", "Store", "Field sales"], len(erp))
    erp = erp.sample(frac=1, random_state=1).reset_index(drop=True)
    return crm, erp


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The widget
    """)
    return


@app.cell
def _(crm, erp, mo):
    _common = [c for c in crm.columns if c in erp.columns]
    key_select = mo.ui.multiselect(options=_common, value=["customer_id"], label="Key")
    rounding = mo.ui.dropdown(
        options={
            "no rounding": None,
            "0 decimals": 0,
            "1 decimal": 1,
            "2 decimals": 2,
            "4 decimals": 4,
        },
        value="no rounding",
        label="Round numeric columns before comparing",
    )
    ignore = mo.ui.multiselect(options=_common, value=[], label="Ignore columns")
    mo.hstack([key_select, rounding, ignore], justify="start", gap=2)
    return ignore, key_select, rounding


@app.cell(expand_output=True)
def _(DataFrameDiff, crm, erp, ignore, key_select, mo, rounding):
    mo.stop(
        not key_select.value,
        mo.callout("Select at least one key column.", kind="warn"),
    )

    # no tolerance in the comparison – rounding happens here, on both sides
    def _round(df):
        if rounding.value is None:
            return df
        return df.round(dict.fromkeys(df.select_dtypes("number").columns, rounding.value))

    _left, _right = _round(crm), _round(erp)
    try:
        diff = DataFrameDiff(
            _left,
            _right,
            key=key_select.value,
            left_name="CRM",
            right_name="ERP",
            ignore_columns=ignore.value,
        )
        _error = None
    except ValueError as _e:
        diff, _error = None, str(_e)

    mo.stop(_error is not None, mo.callout(mo.md(f"**Cannot compare.** {_error}"), kind="danger"))

    diff_view = mo.ui.anywidget(diff)
    diff_view
    return diff, diff_view


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Things to try in the widget": mo.md("""
    - **Overview** – the bars share one scale: equal / mismatch / only in ERP / only in
      CRM, and found vs. not found. Each side has one colour and one chip (**C**, **E**)
      used everywhere.
    - **Status filter** – the funnel in the **Status** column header: presets (*All
      differences*, *One side only*), one checkbox per status, *only* to pick one. An
      active status filter shows as a pill in the toolbar; × removes it.
    - **Quick filters** – in each column header, click **≠** for rows that differ in
      that column or **=** for rows that are equal in it. Click again to turn it off.
    - **Column menus** – the funnel next to each column name: sort, and type-aware
      filters (text, number range, date range, boolean, null checks). For compared
      columns a row matches if the **CRM or the ERP value** matches – try *city equals
      Berlin*.
    - **Invisible characters** – search for `Customer 10` or sort by *name*: trailing
      spaces (`␣`), no-break spaces (`⍽`), zero-width spaces and tabs are made visible.
      Toggle *Show invisible characters* to compare.
    - **Detail view** – click a row: both sides next to each other with numeric deltas;
      step through all filtered rows with ‹ ›, across page boundaries.
    - **Differing columns only** – hides compared columns without a difference in the
      current view.
    - **Rows per page** – the table height stays fixed while you filter; it only
      changes with the page size.
    - **Export CSV** – downloads all rows matching the current filters.
    - **Check rows** – they show up in Python below.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Continue in Python

    `diff_view.value["selected_ids"]` is the only reactive value: this cell re-runs when
    you check or uncheck rows – paging, sorting and filtering never re-run anything.
    """)
    return


@app.cell
def _(diff, diff_view, mo):
    # reacts to the selection in the widget
    # re-runs whenever the selection in the widget changes (and only then)
    diff_view.value.get("selected_ids")
    selection = diff.selected_frame()

    mo.ui.tabs(
        {
            f"Selected ({len(selection)})": selection
            if len(selection)
            else mo.md("_Check rows in the widget to see them here._"),
            "Mismatches (long)": mo.ui.table(diff.result.mismatches(), page_size=10),
            "Per column": mo.ui.table(diff.result.column_stats(), selection=None),
            "Result (wide)": mo.ui.table(diff.result.frame(), page_size=10),
        }
    )
    return


@app.cell
def _(mo):
    read_view = mo.ui.run_button(label="Read the widget's current view")
    mo.md(f"""
    `diff.view_frame()` returns the rows matching the widget's current filters, in view
    order. It is not reactive (filters do not re-run cells), so read it on demand:
    {read_view}
    """)
    return (read_view,)


@app.cell
def _(diff, mo, read_view):
    mo.stop(not read_view.value, mo.md("_Press the button after filtering the widget._"))
    _view = diff.view_frame()
    mo.vstack(
        [mo.md(f"**{len(_view)} rows** in the current view"), mo.ui.table(_view, page_size=5)]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Without a UI

    `dference.compare()` runs the same comparison and returns a `DiffResult` with plain
    polars tables – for tests, pipelines and reports.
    """)
    return


@app.cell
def _(crm, dference, erp):
    result = dference.compare(crm, erp, key="customer_id", left_name="CRM", right_name="ERP")
    result.summary
    return (result,)


@app.cell
def _(mo, pl, result):
    _columns = pl.DataFrame(
        [
            {
                "name": c.name,
                "kind": c.kind,
                "dtype_left": str(c.dtype_left),
                "dtype_right": str(c.dtype_right),
                "compare_dtype": str(c.compare_dtype),
                "mismatches": c.mismatches,
            }
            for c in result.columns
        ]
    )
    mo.ui.tabs(
        {
            "frame('mismatch')": mo.ui.table(result.frame("mismatch"), page_size=5),
            "frame('missing_right')": mo.ui.table(result.frame("missing_right"), page_size=5),
            "mismatches()": mo.ui.table(result.mismatches(), page_size=5),
            "column_stats()": mo.ui.table(result.column_stats(), selection=None),
            "columns": mo.ui.table(_columns, selection=None),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · How values are compared

    - **Exact equality** – no tolerance; round first if small float differences should
      not count (the *Round* dropdown above does exactly that).
    - **Missing values are equal** – `null == null`, and `NaN` counts as missing in float
      columns.
    - **Same column, same dtype** – by default (`strict=True`) the dtypes must be
      identical on both sides.
    - **`strict=False`** aligns lossless differences – the same values stored
      differently: integers of different width, `Float32` vs. `Float64`, decimals of
      different precision, `Datetime` in different time units (same time zone), text
      vs. `Categorical`/`Enum`, and an all-null column. `ColumnInfo.compare_dtype` says
      which dtype was used.
    - **Everything else is always your decision** – int vs. float, `Date` vs.
      `Datetime`, naive vs. time-zone aware, different time zones, text vs. numbers:
      `compare()` raises one error that lists every such column with a cast to fix it.
    """)
    return


@app.cell
def _(dt, pl):
    # the same data, stored differently on the two sides
    storage_left = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3, 4], dtype=pl.Int32),
            "qty": pl.Series([1, 2, 3, None], dtype=pl.Int16),
            "price": pl.Series([9.5, None, float("nan"), 5.0], dtype=pl.Float32),
            "seen": pl.Series([dt.datetime(2026, 1, d) for d in (1, 2, 3, 4)]).cast(
                pl.Datetime("ms")
            ),
            "code": pl.Series(["A", "B", "C", "D"], dtype=pl.Categorical),
        }
    )
    storage_right = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],  # Int64
            "qty": pl.Series([1, 2, 3, None], dtype=pl.UInt8),
            "price": [9.5, float("nan"), None, 5.0],  # Float64; NaN and null are both missing
            "seen": [dt.datetime(2026, 1, d, 12 if d == 4 else 0) for d in (1, 2, 3, 4)],  # µs
            "code": ["A", "B", "C", "X"],  # String
        }
    )
    return storage_left, storage_right


@app.cell
def _(mo):
    strict = mo.ui.switch(value=False, label="strict (identical dtypes)")
    strict
    return (strict,)


@app.cell
def _(DataFrameDiff, mo, storage_left, storage_right, strict):
    try:
        storage = DataFrameDiff(
            storage_left,
            storage_right,
            key="id",
            left_name="Before",
            right_name="After",
            strict=strict.value,
            page_size=5,
        )
        _error = None
    except ValueError as _e:
        storage, _error = None, str(_e)
    mo.stop(_error is not None, mo.callout(mo.md(f"```\n{_error}\n```"), kind="warn"))
    storage
    return (storage,)


@app.cell
def _(mo, pl, storage):
    mo.ui.table(
        pl.DataFrame(
            [
                {
                    "column": c.name,
                    "left": str(c.dtype_left),
                    "right": str(c.dtype_right),
                    "compared as": str(c.compare_dtype or c.dtype_left),
                }
                for c in storage.result.columns
            ]
        ),
        selection=None,
    )
    return


@app.cell
def _(dference, dt, mo, pl):
    # different kinds of values: dference refuses to guess
    kinds_left = pl.DataFrame(
        {
            "id": [1, 2],
            "qty": [1, 2],
            "day": [dt.date(2026, 1, 1), dt.date(2026, 1, 2)],
            "ts": [dt.datetime(2026, 1, 1, 8), dt.datetime(2026, 1, 2, 8)],
        }
    )
    kinds_right = pl.DataFrame(
        {
            "id": [1, 2],
            "qty": [1.0, 2.5],
            "day": [dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2)],
            "ts": [dt.datetime(2026, 1, 1, 8), dt.datetime(2026, 1, 2, 8)],
        }
    ).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    try:
        dference.compare(kinds_left, kinds_right, key="id", left_name="CRM", right_name="ERP")
        _msg = "no error"
    except ValueError as _e:
        _msg = str(_e)
    mo.callout(mo.md(f"```\n{_msg}\n```"), kind="warn")
    return kinds_left, kinds_right


@app.cell
def _(DataFrameDiff, kinds_left, kinds_right, pl):
    # decide how the values should be compared, then compare
    DataFrameDiff(
        kinds_left,
        kinds_right.with_columns(
            pl.col("qty").round().cast(pl.Int64),  # or cast the left side to Float64
            pl.col("day").dt.date(),
            pl.col("ts").dt.replace_time_zone(None),
        ),
        key="id",
        left_name="CRM",
        right_name="ERP",
        page_size=5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Keys, names and markers

    Keys can span several columns; the combination must be unique on each side. The side
    markers default to the first letter of each name (`L`/`R` if both start alike) and can
    be set with `left_short` / `right_short`. `DataFrameDiff.from_result()` reuses a
    comparison you already ran.
    """)
    return


@app.cell
def _(DataFrameDiff, pl):
    _prod = pl.DataFrame(
        {
            "region": ["EU", "EU", "US", "US"],
            "sku": [1, 2, 1, 2],
            "stock": [10, 5, 7, 0],
        }
    )
    _preview = pl.DataFrame(
        {
            "region": ["EU", "EU", "US", "APAC"],
            "sku": [1, 2, 1, 1],
            "stock": [10, 6, 7, 3],
        }
    )
    # "Prod" and "Preview" both start with P: markers would clash, so set them
    DataFrameDiff(
        _prod,
        _preview,
        key=["region", "sku"],
        left_name="Prod",
        right_name="Preview",
        left_short="PR",
        right_short="PV",
        page_size=5,
    )
    return


@app.cell
def _(DataFrameDiff, result):
    # reuse the result from section 3, only the mismatching columns matter here
    DataFrameDiff.from_result(result, page_size=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Inputs

    polars (eager or lazy), pandas, pyarrow – or anything with the Arrow PyCapsule
    interface or `to_polars()` (DuckDB relations, narwhals, …). The result is the same.
    """)
    return


@app.cell
def _(crm, dference, erp, mo, pa, pl):
    _inputs = {
        "pandas": (crm, erp),
        "polars": (pl.from_pandas(crm), pl.from_pandas(erp)),
        "polars lazy": (pl.from_pandas(crm).lazy(), pl.from_pandas(erp).lazy()),
        "pyarrow": (
            pa.Table.from_pandas(crm, preserve_index=False),
            pa.Table.from_pandas(erp, preserve_index=False),
        ),
    }
    _rows = []
    for _name, (_l, _r) in _inputs.items():
        _s = dference.compare(_l, _r, key="customer_id").summary
        _rows.append(
            {
                "input": _name,
                "equal": _s.equal,
                "mismatch": _s.mismatch,
                "missing_left": _s.missing_left,
                "missing_right": _s.missing_right,
            }
        )
    mo.ui.table(pl.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Large data

    Filtering, sorting and paging run in polars; the browser only ever receives the
    visible page. Pick a size and run it – the widget stays responsive.
    """)
    return


@app.cell
def _(mo):
    big_rows = mo.ui.dropdown(
        options={"100 000": 100_000, "1 million": 1_000_000, "5 million": 5_000_000},
        value="1 million",
        label="Rows per side",
    )
    big_run = mo.ui.run_button(label="Generate and compare")
    mo.hstack([big_rows, big_run], justify="start")
    return big_rows, big_run


@app.cell
def _(DataFrameDiff, big_rows, big_run, mo, np, pl, time):
    mo.stop(not big_run.value, mo.md("_Press **Generate and compare**._"))
    _n = big_rows.value
    _rng = np.random.default_rng(0)
    _left = pl.DataFrame(
        {
            "id": np.arange(_n),
            "value": _rng.normal(size=_n).round(3),
            "group": _rng.integers(0, 50, _n),
            "label": pl.Series(_rng.integers(0, 1000, _n)).cast(pl.String),
        }
    )
    _changed = _rng.random(_n) < 0.02
    _right = _left.with_columns(
        value=pl.when(pl.Series(_changed)).then(pl.col("value") + 1).otherwise(pl.col("value"))
    ).filter(pl.col("id") % 97 != 0)
    _t0 = time.perf_counter()
    big = DataFrameDiff(_left, _right, key="id", left_name="Before", right_name="After")
    _took = time.perf_counter() - _t0
    mo.vstack([mo.md(f"Compared **{_n:,}** rows per side in **{_took:.2f} s**."), big])
    return


if __name__ == "__main__":
    app.run()
