// dference – frontend (anywidget ESM), modelled on marimo.ui.table.
//
// The table is paged on the Python side: every change to status, search,
// filters, sort or page sends a `query` message; Python answers with the rows
// of that page only. Static metadata (columns, counts, names) arrives as
// synced traits. The checked rows are synced back as `selected_ids`.

// Types are checked with `tsc --checkJs` (see tsconfig.json); nothing is compiled.

/** @typedef {"equal"|"mismatch"|"missing_left"|"missing_right"} Status */
/** @typedef {"key"|"compared"|"left_only"|"right_only"} ColumnKind */
/** @typedef {"number"|"boolean"|"datetime"|"string"} FilterType */
/**
 * A cell value after JSON conversion in Python (`_query._json`).
 * @typedef {string|number|boolean|null} Value
 */
/**
 * @typedef {object} Column
 * @property {string} name
 * @property {ColumnKind} kind
 * @property {string} dtype
 * @property {boolean} numeric
 * @property {FilterType} ftype
 * @property {number} mismatches
 */
/**
 * Counts and names synced from Python (`_widget._summary_payload`).
 * @typedef {object} Summary
 * @property {number} equal
 * @property {number} mismatch
 * @property {number} missing_left
 * @property {number} missing_right
 * @property {number} total
 * @property {number} found
 * @property {number} left_rows
 * @property {number} right_rows
 * @property {string[]} keys
 * @property {string[]} compared
 * @property {string[]} left_only_cols
 * @property {string[]} right_only_cols
 * @property {string[]} ignored
 * @property {number} duplicate_keys_left Keys that occur more than once on the left.
 * @property {number} duplicate_keys_right Keys that occur more than once on the right.
 * @property {number} duplicate_rows Rows whose key is not unique on a side.
 * @property {"match"|"number"} duplicates How rows sharing a key were paired.
 */
/**
 * Traits synced with the Python `DataFrameDiff` widget.
 * @typedef {object} Model
 * @property {Column[]} columns
 * @property {Summary} summary
 * @property {string} left_name
 * @property {string} right_name
 * @property {string} left_short
 * @property {string} right_short
 * @property {number} page_size
 * @property {boolean} text_diff
 * @property {number[]} selected_ids
 */
/**
 * One row as sent by Python. `l`/`r` are aligned with `columns` and `null`
 * for a side that does not exist in this row; `d` lists differing columns;
 * `k` is how often the key occurs on the left and right, if it is not unique.
 * @typedef {{id: number, s: Status, d: number[], l: Value[]|null, r: Value[]|null, k?: [number, number]}} Row
 */
/** @typedef {{op: string, a: string, b: string}} Filter */
/** @typedef {{col: number|"status", dir: 1|-1}} Sort */

// left before right: "only in left" (missing_right) comes before "only in right"
const STATUS_ORDER = /** @type {const} */ (["equal", "mismatch", "missing_right", "missing_left"]);
const FUNNEL = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 3.5h11l-4.2 5v4l-2.6 1.2V8.5z"/></svg>`;
const PAGE_SIZES = [10, 25, 50, 100];
const LOCALE = "en-US";
const SEARCH_DEBOUNCE_MS = 200;
const INT_TEXT = /^-?\d+$/;

// ---------------------------------------------------------------- formatting

/** @type {Readonly<Record<string, string>>} */
const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
/**
 * Escape text for use in HTML content and attributes.
 * @param {unknown} s
 */
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => HTML_ESCAPES[c] ?? c);
/** @param {Value|undefined} v */
const plain = (v) => (v === null || v === undefined ? "" : String(v));
/** @param {number} x */
const n = (x) => x.toLocaleString(LOCALE);

/**
 * Percentage with one decimal; 0% / 100% only when exact.
 * @param {number} part
 * @param {number} whole
 */
function pct(part, whole) {
  if (!whole) return "–";
  if (part === 0) return "0%";
  if (part === whole) return "100%";
  const p = (100 * part) / whole;
  if (p < 0.1) return "<0.1%";
  if (p > 99.9) return ">99.9%";
  return `${p.toLocaleString(LOCALE, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
}

/**
 * Signed difference right − left for two numbers, else null.
 * @param {Value|undefined} a
 * @param {Value|undefined} b
 * @returns {string|null}
 */
function delta(a, b) {
  // integers beyond 2**53 arrive as text (see `_json`): subtract them exactly
  if (INT_TEXT.test(String(a)) && INT_TEXT.test(String(b)) && (typeof a === "string" || typeof b === "string")) {
    const d = BigInt(String(b)) - BigInt(String(a));
    return (d > 0n ? "+" : "") + d.toLocaleString(LOCALE);
  }
  if (typeof a !== "number" || typeof b !== "number") return null;
  const d = b - a;
  const s =
    Math.abs(d) >= 0.01 || d === 0 ? d.toLocaleString(LOCALE, { maximumFractionDigits: 2 }) : d.toExponential(2);
  return (d > 0 ? "+" : "") + s;
}

// ------------------------------------------------------ invisible characters
// Characters that render as nothing (or like a normal space) but still make
// two strings differ. Regular spaces (␣) are only marked when leading,
// trailing or repeated; single inner spaces stay untouched.

/** @type {Readonly<Record<string, [symbol: string, name: string]>>} */
const INV_NAMES = {
  "\t": ["→", "tab"],
  "\n": ["↵", "line feed"],
  "\r": ["␍", "carriage return"],
  "\u00a0": ["⍽", "no-break space"],
  "\u202f": ["⍽", "narrow no-break space"],
  "\u2007": ["⍽", "figure space"],
  "\u2009": ["⍽", "thin space"],
  "\u3000": ["⍽", "ideographic space"],
  "\u200b": ["ZWSP", "zero-width space"],
  "\u200c": ["ZWNJ", "zero-width non-joiner"],
  "\u200d": ["ZWJ", "zero-width joiner"],
  "\u2060": ["WJ", "word joiner"],
  "\ufeff": ["BOM", "byte order mark"],
  "\u00ad": ["SHY", "soft hyphen"],
  "\u200e": ["LRM", "left-to-right mark"],
  "\u200f": ["RLM", "right-to-left mark"],
};
// Control characters are exactly what these patterns are meant to find.
// biome-ignore lint/suspicious/noControlCharactersInRegex: detecting invisible characters is the point
const INV_RE = /[\u0000-\u001f\u007f-\u00a0\u00ad\u1680\u180e\u2000-\u200f\u2028-\u202f\u205f-\u2064\u3000\ufeff]/;
const INV_SPACE_RE = /[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]/g;
// biome-ignore lint/suspicious/noControlCharactersInRegex: detecting invisible characters is the point
const INV_ZERO_RE = /[\u0000-\u001f\u007f-\u009f\u00ad\u180e\u200b-\u200f\u2028-\u202e\u2060-\u2064\ufeff]/g;

/**
 * Code point of a single character as hex digits, e.g. `"200B"`.
 * @param {string} c
 */
const codePoint = (c) => (c.codePointAt(0) ?? 0).toString(16).toUpperCase().padStart(4, "0");
/** @param {string} c */
const hex = (c) => `U+${codePoint(c)}`;

/**
 * Flags for regular spaces that should be made visible.
 * @param {string[]} chars
 * @returns {boolean[]}
 */
function markedSpaces(chars) {
  const mark = new Array(chars.length).fill(false);
  for (let i = 0; i < chars.length; ) {
    if (chars[i] !== " ") {
      i++;
      continue;
    }
    let j = i;
    while (j < chars.length && chars[j] === " ") j++;
    if (i === 0 || j === chars.length || j - i > 1) mark.fill(true, i, j);
    i = j;
  }
  return mark;
}

/**
 * Does the value contain anything worth highlighting?
 * @param {Value|undefined} s
 * @returns {s is string}
 */
function hasInvisible(s) {
  if (typeof s !== "string") return false;
  return s === "" || INV_RE.test(s) || markedSpaces(Array.from(s)).some(Boolean);
}

/**
 * Grey marker for one invisible character.
 * @param {string} c
 */
function invMark(c) {
  const [sym, name] = INV_NAMES[c] ?? [hex(c), "control / invisible character"];
  const cls = sym.length > 1 ? "dfd-inv code" : "dfd-inv";
  return `<span class="${cls}" title="${hex(c)} ${name}">${esc(sym)}</span>`;
}

/**
 * HTML for a string with its invisible characters rendered as grey markers.
 * @param {string} s
 */
function visualize(s) {
  if (s === "") return '<span class="dfd-inv" title="Empty string (not null)">""</span>';
  const chars = Array.from(s);
  const mark = markedSpaces(chars);
  return chars
    .map((c, i) => {
      if (c === " ") return mark[i] ? '<span class="dfd-inv" title="U+0020 space">␣</span>' : " ";
      if (!INV_RE.test(c)) return esc(c);
      return `${invMark(c)}${c === "\n" ? "<br>" : ""}`;
    })
    .join("");
}

/**
 * Text after removing invisible characters and normalising whitespace.
 * @param {string} s
 */
const normalise = (s) => s.replace(INV_SPACE_RE, " ").replace(INV_ZERO_RE, "").replace(/\s+/g, " ").trim();

/**
 * Do two strings differ only in invisible characters?
 * @param {Value|undefined} a
 * @param {Value|undefined} b
 */
const onlyInvisible = (a, b) =>
  typeof a === "string" && typeof b === "string" && a !== b && normalise(a) === normalise(b);

/**
 * Tooltip form of a value: strings quoted, invisible characters as `\u` escapes.
 * @param {Value|undefined} v
 */
const literal = (v) =>
  typeof v === "string"
    ? `"${Array.from(v)
        .map((c) => (INV_RE.test(c) ? `\\u${codePoint(c).toLowerCase()}` : c))
        .join("")}"`
    : plain(v);

/**
 * HTML for one cell value.
 * @param {Value|undefined} v
 * @param {boolean} showInvisible
 */
function fmt(v, showInvisible) {
  if (v === null || v === undefined) return '<span class="dfd-null">null</span>';
  if (typeof v === "boolean") return String(v);
  if (typeof v === "string" && /^\d{4}-\d{2}-\d{2}T00:00:00$/.test(v)) return esc(v.slice(0, 10));
  if (showInvisible && hasInvisible(v)) return visualize(v);
  return esc(v);
}

// --------------------------------------------------------------- line breaks
// A table row is always one line high: a multi-line string shows its first line
// and a button that opens the whole text in a flyover.

const LINE_BREAK_RE = /\r\n|[\n\r\u2028\u2029]/;

/**
 * Lines of a string with line breaks, or `null` for any other value.
 * @param {Value|undefined} v
 * @returns {string[]|null}
 */
const linesOf = (v) => (typeof v === "string" && LINE_BREAK_RE.test(v) ? v.split(LINE_BREAK_RE) : null);

/**
 * The line breaks of a string, in order (`"\r\n"` counts as one).
 * @param {string} v
 */
const breaksOf = (v) => v.match(new RegExp(LINE_BREAK_RE.source, "g")) ?? [];

/**
 * HTML for the first line of a multi-line string, followed by its line break
 * (as a marker when invisible characters are shown).
 * @param {string} v
 * @param {boolean} showInvisible
 */
function firstLine(v, showInvisible) {
  const m = /** @type {RegExpExecArray} */ (LINE_BREAK_RE.exec(v));
  const head = v.slice(0, m.index);
  if (!showInvisible) return esc(head);
  return (head && hasInvisible(head) ? visualize(head) : esc(head)) + Array.from(m[0]).map(invMark).join("");
}

/**
 * HTML for one line inside the flyover: like `visualize`, but an empty line
 * stays empty (it is a line, not an empty string).
 * @param {string} line
 * @param {boolean} showInvisible
 */
const fmtLine = (line, showInvisible) => (showInvisible && line && hasInvisible(line) ? visualize(line) : esc(line));

// ---------------------------------------------------------------- text diff
// Two differing texts are shown as a unified diff, like `git diff`: lines are
// matched with Myers' algorithm (git's default), changed lines are paired and
// the words that differ within them are highlighted.

/** Above this many lines or edits a text is not diffed (the page must stay responsive). */
const DIFF_MAX_LINES = 2000;
const DIFF_MAX_EDITS = 500;
/** Unchanged lines kept around each change; longer runs collapse. */
const DIFF_CONTEXT = 3;

/**
 * Edit script between two sequences (Myers' O(ND) algorithm).
 * @template T
 * @param {readonly T[]} a
 * @param {readonly T[]} b
 * @returns {("="|"-"|"+")[]|null} One op per element of `a` and `b` in order, or
 *   `null` if the sequences need more than `DIFF_MAX_EDITS` edits.
 */
function diffSeq(a, b) {
  const n = a.length;
  const m = b.length;
  const off = n + m + 1;
  const v = new Int32Array(2 * off + 1);
  /** `v` at the start of each round, sliced to the diagonals that round can read. */
  const trace = [];
  for (let d = 0; d <= Math.min(n + m, DIFF_MAX_EDITS); d++) {
    trace.push(v.slice(off - d, off + d + 1));
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && v[off + k - 1] < v[off + k + 1]) ? v[off + k + 1] : v[off + k - 1] + 1;
      let y = x - k;
      while (x < n && y < m && a[x] === b[y]) {
        x++;
        y++;
      }
      v[off + k] = x;
      if (x >= n && y >= m) return backtrack(trace, n, m);
    }
  }
  return null;
}

/**
 * Walk the rounds of `diffSeq` back from the end to the edit script.
 * @param {Int32Array[]} trace
 * @param {number} n
 * @param {number} m
 * @returns {("="|"-"|"+")[]}
 */
function backtrack(trace, n, m) {
  /** @type {("="|"-"|"+")[]} */
  const ops = [];
  let x = n;
  let y = m;
  for (let d = trace.length - 1; d > 0; d--) {
    const snap = trace[d];
    /** @param {number} k */
    const V = (k) => snap[k + d];
    const k = x - y;
    const prevK = k === -d || (k !== d && V(k - 1) < V(k + 1)) ? k + 1 : k - 1;
    const prevX = V(prevK);
    const prevY = prevX - prevK;
    while (x > prevX && y > prevY) {
      ops.push("=");
      x--;
      y--;
    }
    ops.push(prevK === k + 1 ? "+" : "-");
    x = prevX;
    y = prevY;
  }
  while (x-- > 0) ops.push("=");
  return ops.reverse();
}

const LINE_RE = /[^\r\n\u2028\u2029]*(?:\r\n|[\n\r\u2028\u2029])|[^\r\n\u2028\u2029]+$/g;
/**
 * Lines of a text, each with its line break: texts that differ only in their
 * line breaks (`\r\n` vs `\n`) differ in the diff too.
 * @param {string} s
 */
const splitLines = (s) => s.match(LINE_RE) ?? [];
/** Words, runs of whitespace and single other characters. */
const TOKEN_RE = /[\p{L}\p{N}_]+|\s+|[^\p{L}\p{N}_\s]/gu;

/**
 * Which code points of `a` and `b` belong to words that differ, and how
 * similar the two are (share of characters in common words, 0–1).
 * @param {string} a
 * @param {string} b
 * @returns {{marks: [boolean[], boolean[]], score: number}|null} `null` if the
 *   two have no word in common.
 */
function wordDiff(a, b) {
  const ta = a.match(TOKEN_RE) ?? [];
  const tb = b.match(TOKEN_RE) ?? [];
  const ops = diffSeq(ta, tb);
  if (!ops) return null;
  /** @type {boolean[]} */
  const fa = [];
  /** @type {boolean[]} */
  const fb = [];
  let i = 0;
  let j = 0;
  let same = 0;
  for (const op of ops) {
    if (op !== "+") {
      const len = Array.from(ta[i++]).length;
      fa.push(...new Array(len).fill(op === "-"));
      if (op === "=" && /[\p{L}\p{N}]/u.test(ta[i - 1])) same += len;
    }
    if (op !== "-") {
      const len = Array.from(tb[j++]).length;
      fb.push(...new Array(len).fill(op === "+"));
    }
  }
  const score = same / Math.max(fa.length, fb.length, 1);
  return same ? { marks: [fa, fb], score } : null;
}

/**
 * HTML of one line of a diff: invisible characters as markers (when shown, the
 * line break too), code points flagged in `marked` highlighted.
 * @param {string} line Text of the line, including its line break.
 * @param {boolean[]|null} marked
 * @param {boolean} showInvisible
 */
function diffLineHtml(line, marked, showInvisible) {
  const chars = Array.from(line);
  let end = chars.length;
  while (end > 0 && /[\n\r\u2028\u2029]/.test(chars[end - 1])) end--;
  const spaces = markedSpaces(chars.slice(0, end));
  const parts = chars.map((c, i) => {
    if (i >= end) return showInvisible ? invMark(c) : "";
    if (!showInvisible) return esc(c);
    if (c === " ") return spaces[i] ? '<span class="dfd-inv" title="U+0020 space">␣</span>' : " ";
    return INV_RE.test(c) ? invMark(c) : esc(c);
  });
  let html = "";
  for (let i = 0; i < parts.length; ) {
    const on = !!marked?.[i];
    let j = i;
    while (j < parts.length && !!marked?.[j] === on) j++;
    const run = parts.slice(i, j).join("");
    html += on ? `<mark class="dfd-dx">${run}</mark>` : run;
    i = j;
  }
  return html;
}

/**
 * One line of a unified diff.
 * @typedef {object} DiffLine
 * @property {"="|"-"|"+"|"…"} op Unchanged, only left, only right, or collapsed lines.
 * @property {string} text
 * @property {number} a Line number on the left (0 if none).
 * @property {number} b Line number on the right (0 if none).
 * @property {boolean[]|null} marked Code points of `text` that differ.
 * @property {number} [hidden] Number of lines a collapsed line stands for.
 */

/**
 * Unified diff of two texts, or `null` if they are too large to diff.
 * @param {string} a
 * @param {string} b
 * @returns {DiffLine[]|null}
 */
function unifiedDiff(a, b) {
  const la = splitLines(a);
  const lb = splitLines(b);
  if (la.length > DIFF_MAX_LINES || lb.length > DIFF_MAX_LINES) return null;
  const ops = diffSeq(la, lb);
  if (!ops) return null;
  /** @type {DiffLine[]} */
  const out = [];
  let i = 0;
  let j = 0;
  for (let p = 0; p < ops.length; ) {
    if (ops[p] === "=") {
      out.push({ op: "=", text: la[i], a: ++i, b: ++j, marked: null });
      p++;
      continue;
    }
    // a block of changes: removed lines first, then added ones (like git); for
    // the word diff each removed line is paired with the most similar added
    // line after the previous pair
    /** @type {DiffLine[]} */
    const del = [];
    /** @type {DiffLine[]} */
    const add = [];
    for (; p < ops.length && ops[p] !== "="; p++) {
      if (ops[p] === "-") del.push({ op: "-", text: la[i], a: ++i, b: 0, marked: null });
      else add.push({ op: "+", text: lb[j], a: 0, b: ++j, marked: null });
    }
    let next = 0;
    for (const d of del) {
      /** @type {{k: number, w: NonNullable<ReturnType<typeof wordDiff>>}|null} */
      let best = null;
      for (let k = next; k < add.length; k++) {
        const w = wordDiff(d.text, add[k].text);
        if (w && w.score >= 0.3 && (!best || w.score > best.w.score)) best = { k, w };
      }
      if (best) {
        [d.marked, add[best.k].marked] = best.w.marks;
        next = best.k + 1;
      }
    }
    out.push(...del, ...add);
  }
  return collapse(out);
}

/**
 * Collapse runs of unchanged lines to `DIFF_CONTEXT` lines around each change.
 * @param {DiffLine[]} lines
 * @returns {DiffLine[]}
 */
function collapse(lines) {
  /** @type {DiffLine[]} */
  const out = [];
  for (let i = 0; i < lines.length; ) {
    if (lines[i].op !== "=") {
      out.push(lines[i++]);
      continue;
    }
    let j = i;
    while (j < lines.length && lines[j].op === "=") j++;
    const keepHead = i === 0 ? 0 : DIFF_CONTEXT;
    const keepTail = j === lines.length ? 0 : DIFF_CONTEXT;
    if (j - i > keepHead + keepTail + 1) {
      out.push(...lines.slice(i, i + keepHead));
      out.push({ op: "…", text: "", a: 0, b: 0, marked: null, hidden: j - i - keepHead - keepTail });
      out.push(...lines.slice(j - keepTail, j));
    } else out.push(...lines.slice(i, j));
    i = j;
  }
  return out;
}

// ------------------------------------------------------------ column filters

/**
 * Operators per filter type – must match `OPERATORS` in `_query.py`.
 * @type {Readonly<Record<FilterType, [op: string, label: string][]>>}
 */
const OPS = {
  string: [
    ["contains", "contains"],
    ["not_contains", "does not contain"],
    ["equals", "equals"],
    ["starts", "starts with"],
    ["ends", "ends with"],
    ["is_empty", "is empty"],
    ["not_empty", "is not empty"],
    ["is_null", "is null"],
    ["not_null", "is not null"],
  ],
  number: [
    ["between", "between"],
    ["eq", "="],
    ["ne", "≠"],
    ["is_null", "is null"],
    ["not_null", "is not null"],
  ],
  datetime: [
    ["between", "between"],
    ["is_null", "is null"],
    ["not_null", "is not null"],
  ],
  boolean: [
    ["true", "is true"],
    ["false", "is false"],
    ["is_null", "is null"],
  ],
};
const NO_VALUE = new Set(["is_empty", "not_empty", "is_null", "not_null", "true", "false"]);

/**
 * Short text for a filter pill, e.g. `contains “kiel”` or `10 – 20`.
 * @param {FilterType} ftype
 * @param {Filter} f
 */
function describeFilter(ftype, f) {
  const label = OPS[ftype].find(([k]) => k === f.op)?.[1] ?? f.op;
  if (NO_VALUE.has(f.op)) return label;
  if (f.op === "between") return f.a && f.b ? `${f.a} – ${f.b}` : f.a ? `≥ ${f.a}` : `≤ ${f.b}`;
  return ftype === "string" ? `${label} “${f.a}”` : `${label} ${f.a}`;
}

/**
 * Surrounding background colour, so sticky headers & popovers are opaque.
 * Walks up through shadow roots, since widgets may live inside one.
 * @param {Element} node
 * @returns {string|null}
 */
function detectBg(node) {
  /** @type {Element|null} */
  let el = node;
  while (el) {
    const bg = getComputedStyle(el).backgroundColor;
    if (bg && bg !== "transparent" && !/rgba\([^)]*,\s*0\)$/.test(bg)) return bg;
    const root = el.getRootNode();
    el = el.parentElement ?? (root instanceof ShadowRoot ? root.host : null);
  }
  return null;
}

// ==================================================================== render

/** @type {import("@anywidget/types").Render<Model>} */
function render({ model, el }) {
  /** UI state; everything in `query()` is evaluated on the Python side. */
  const state = {
    statuses: /** @type {Set<Status>} */ (new Set(STATUS_ORDER)),
    /** Only rows whose key is not unique on a side. */
    dupOnly: false,
    search: "",
    sort: /** @type {Sort|null} */ (null),
    page: 0,
    pageSize: model.get("page_size") || 10,
    onlyDiffCols: false,
    showInvisible: true,
    /** Compare differing texts like `git diff` (detail view and flyover). */
    textDiff: model.get("text_diff") ?? true,
    /** Quick filter from a column header: rows that differ (`≠`) or are equal (`=`) in it. */
    diffColumn: /** @type {number|null} */ (null),
    diffEqual: false,
    filters: /** @type {Map<number, Filter>} */ (new Map()),
    /** Open header popover: a column index or the status column. */
    colMenu: /** @type {number|"status"|null} */ (null),
    /** Cell whose full text is shown in the flyover. */
    flyover: /** @type {{row: Row, col: number}|null} */ (null),
    /** Row shown in the detail panel and its position in the view. */
    detail: /** @type {{row: Row, pos: number}|null} */ (null),
  };
  /** Last page received from Python. */
  let view = { rows: /** @type {Row[]} */ ([]), filtered: 0, page: 0, diffColumns: /** @type {number[]} */ ([]) };
  let loading = true;
  let error = "";
  let reqSeq = 0;
  /** After a page change triggered by detail navigation: which row to open. */
  let pendingDetail = /** @type {"first"|"last"|null} */ (null);
  /** @type {Set<number>} */
  let selected = new Set(model.get("selected_ids") || []);
  let M = meta();

  /** Static metadata of the comparison, re-read when Python replaces it. */
  function meta() {
    return {
      columns: model.get("columns"),
      summary: model.get("summary"),
      L: model.get("left_name"),
      R: model.get("right_name"),
      LS: model.get("left_short") || "L",
      RS: model.get("right_short") || "R",
    };
  }

  // Each side has one colour and one chip (its short name) everywhere. A row that
  // exists on one side only is shown with the chip of the side it exists on.
  /** @param {Status} s */
  const label = (s) =>
    ({ equal: "Equal", mismatch: "Mismatch", missing_left: `Only in ${M.R}`, missing_right: `Only in ${M.L}` })[s];
  /** @param {Status} s */
  const hint = (s) =>
    ({
      equal: "Key in both, all values equal",
      mismatch: "Key in both, at least one value differs",
      missing_left: `Key only in ${M.R}, missing in ${M.L}`,
      missing_right: `Key only in ${M.L}, missing in ${M.R}`,
    })[s];
  /** @param {Status} s */
  const symbol = (s) => ({ equal: "=", mismatch: "≠", missing_left: M.RS, missing_right: M.LS })[s];
  /** @param {Status} s */
  const badge = (s) => `<span class="dfd-badge st-${s}" aria-hidden="true">${esc(symbol(s))}</span>`;
  /**
   * How often the key of a row occurs per side, e.g. "Key occurs 3× in CRM and 2× in ERP".
   * @param {[number, number]} k
   */
  const dupText = ([l, r]) => `Key occurs ${n(l)}× in ${M.L} and ${n(r)}× in ${M.R}`;
  /**
   * Chip for a row whose key is not unique: the larger count, e.g. `×3`.
   * @param {Row} row
   */
  const dupChip = (row) =>
    row.k ? `<span class="dfd-dup" title="${esc(dupText(row.k))}">×${n(Math.max(...row.k))}</span>` : "";
  /**
   * Chip in the status column; rows with a unique key keep its room free, so
   * all badges stay in line.
   * @param {Row} row
   */
  const dupSlot = (row) =>
    row.k ? dupChip(row) : M.summary.duplicate_rows ? '<span class="dfd-dup empty" aria-hidden="true"></span>' : "";
  /**
   * Chip of a side: its short name on the side colour.
   * @param {"l"|"r"} side
   */
  const chip = (side) =>
    side === "l"
      ? `<span class="dfd-side l" title="${esc(M.L)} (left)">${esc(M.LS)}</span>`
      : `<span class="dfd-side r" title="${esc(M.R)} (right)">${esc(M.RS)}</span>`;
  /** @param {Value|undefined} v */
  const F = (v) => fmt(v, state.showInvisible);
  /**
   * Unified diff of two strings as HTML, `null` if text diff is off or the texts are too large.
   * Removed lines carry the left side's colour, added lines the right side's.
   * @param {string} a Left value.
   * @param {string} b Right value.
   */
  function diffHtml(a, b) {
    if (!state.textDiff) return null;
    const lines = unifiedDiff(a, b);
    if (!lines) return null;
    const sign = { "=": " ", "-": "−", "+": "+", "…": "" };
    const cls = { "=": "ctx", "-": "del", "+": "add", "…": "gap" };
    const body = lines
      .map((d) =>
        d.op === "…"
          ? `<div class="dfd-ud-line gap"><span class="dfd-ln"></span><span class="dfd-ln"></span><span class="dfd-ud-sign"></span><span class="dfd-ud-text">⋯ ${n(d.hidden ?? 0)} unchanged line${d.hidden === 1 ? "" : "s"}</span></div>`
          : `<div class="dfd-ud-line ${cls[d.op]}"><span class="dfd-ln">${d.a || ""}</span><span class="dfd-ln">${d.b || ""}</span><span class="dfd-ud-sign">${sign[d.op]}</span><span class="dfd-ud-text">${diffLineHtml(d.text, d.marked, state.showInvisible)}</span></div>`,
      )
      .join("");
    return `<div class="dfd-ud">
      <div class="dfd-ud-legend"><span class="dfd-sidename del">− ${chip("l")}${esc(M.L)}</span><span class="dfd-sidename add">+ ${chip("r")}${esc(M.R)}</span></div>
      <div class="dfd-ud-body">${body}</div></div>`;
  }

  /**
   * Like `F`, but a multi-line string shows its first line only, with a button
   * that opens the whole text: every row of the table stays one line high.
   * @param {Value|undefined} v
   * @param {Row} row
   * @param {number} i Column of the value.
   */
  function cellHtml(v, row, i) {
    const lines = linesOf(v);
    if (!lines) return F(v);
    const open = state.flyover?.row.id === row.id && state.flyover.col === i;
    return `${firstLine(/** @type {string} */ (v), state.showInvisible)}<button class="dfd-more ${open ? "open" : ""}"
      data-more="${row.id}:${i}" aria-haspopup="dialog" aria-label="Show full text (${n(lines.length)} lines)"
      title="Show full text (${n(lines.length)} lines)">⋯</button>`;
  }

  /**
   * Value of column `i` in a row (`undefined` if the column cannot exist there).
   * @param {Row} row
   * @param {number} i
   * @returns {Value|undefined}
   */
  function cellValue(row, i) {
    const kind = M.columns[i].kind;
    if (kind === "left_only") return row.l ? row.l[i] : undefined;
    if (kind === "right_only") return row.r ? row.r[i] : undefined;
    return row.l ? row.l[i] : row.r?.[i];
  }
  /**
   * Key columns of a row as `[name, value]` pairs.
   * @param {Row} row
   * @returns {[string, Value|undefined][]}
   */
  const keyOf = (row) =>
    M.columns.flatMap((c, i) =>
      c.kind === "key" ? [/** @type {[string, Value|undefined]} */ ([c.name, cellValue(row, i)])] : [],
    );

  // ------------------------------------------------------------ skeleton
  el.innerHTML = `
    <div class="dfd">
      <div class="dfd-overview"></div>
      <div class="dfd-toolbar">
        <label class="dfd-search">
          <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/></svg>
          <input type="search" placeholder="Search…" aria-label="Search all values">
        </label>
        <label class="dfd-toggle"><input type="checkbox" data-act="only-diff"> Differing columns only</label>
        <label class="dfd-toggle" title="Mark spaces at the start/end or repeated, tabs, line breaks, no-break and zero-width characters">
          <input type="checkbox" data-act="show-inv" checked> Show invisible characters</label>
        <label class="dfd-toggle" title="Compare differing texts line by line like git diff, with the words that differ highlighted">
          <input type="checkbox" data-act="text-diff" ${state.textDiff ? "checked" : ""}> Text diff</label>
        <span class="dfd-toolbar-dyn"></span>
        <span class="dfd-spacer"></span>
        <button class="dfd-btn" data-act="export" title="Download all filtered rows as CSV">Export CSV</button>
      </div>
      <div class="dfd-error" role="alert" hidden></div>
      <div class="dfd-scroll"></div>
      <div class="dfd-footer"></div>
      <div class="dfd-detail-wrap"></div>
      <div class="dfd-colmenu-wrap"></div>
      <div class="dfd-flyover-wrap"></div>
    </div>`;
  const root = /** @type {HTMLElement} */ (el.firstElementChild);

  /**
   * Element of the skeleton above; these always exist, so a miss is a bug.
   * @template {Element} [T=HTMLElement]
   * @param {string} selector
   * @returns {T}
   */
  function $(selector) {
    const found = root.querySelector(selector);
    if (!found) throw new Error(`dference: missing element ${selector}`);
    return /** @type {T} */ (found);
  }
  /**
   * Element that may be absent (menus and panels that are rendered on demand).
   * @template {Element} [T=HTMLElement]
   * @param {string} selector
   * @returns {T|null}
   */
  const $opt = (selector) => /** @type {T|null} */ (root.querySelector(selector));
  const searchInput = /** @type {HTMLInputElement} */ ($(".dfd-search input"));

  // ------------------------------------------------------------ messaging

  /** Serialisable query – mirrors `Query.from_message` in `_query.py`. */
  function queryPayload() {
    return {
      statuses: [...state.statuses],
      duplicates_only: state.dupOnly,
      search: state.search,
      diff_column: state.diffColumn,
      diff_equal: state.diffEqual,
      filters: [...state.filters].map(([column, f]) => ({ column, ...f })),
      sort: state.sort ? { column: state.sort.col, descending: state.sort.dir < 0 } : null,
    };
  }

  /** Ask Python for the current page; stale answers are ignored via `req`. */
  function query() {
    loading = true;
    root.classList.add("loading");
    model.send({ type: "query", req: ++reqSeq, page: state.page, page_size: state.pageSize, ...queryPayload() });
  }

  /**
   * Custom message from Python (`page`, `export` or `error`).
   * @param {any} msg
   * @param {DataView[]} buffers
   */
  function onMessage(msg, buffers) {
    if (!msg || typeof msg !== "object") return;
    if (msg.type === "error") {
      error = msg.message || "Unknown error";
      loading = false;
      root.classList.remove("loading");
      return renderError();
    }
    if (msg.type === "export" && buffers && buffers.length) return download(buffers[0], msg.filename);
    if (msg.type !== "page" || msg.req !== reqSeq) return;
    error = "";
    loading = false;
    root.classList.remove("loading");
    view = { rows: msg.rows, filtered: msg.filtered, page: msg.page, diffColumns: msg.diff_columns };
    state.page = msg.page;
    if (pendingDetail && view.rows.length) {
      const idx = pendingDetail === "first" ? 0 : view.rows.length - 1;
      state.detail = { row: view.rows[idx], pos: state.page * state.pageSize + idx };
    }
    pendingDetail = null;
    // keep the detail position in sync if the row is on this page
    const detail = state.detail;
    if (detail) {
      const idx = view.rows.findIndex((r) => r.id === detail.row.id);
      detail.pos = idx >= 0 ? state.page * state.pageSize + idx : -1;
      if (idx >= 0) detail.row = view.rows[idx];
    }
    renderError();
    renderTable();
    renderDetail();
  }

  /**
   * Save an exported CSV buffer as a file.
   * @param {DataView} buffer
   * @param {string} [filename]
   */
  function download(buffer, filename) {
    // copy into a plain ArrayBuffer: a DataView may sit on a SharedArrayBuffer
    const bytes = new Uint8Array(buffer.byteLength);
    bytes.set(new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength));
    const blob = new Blob([bytes], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename || "dataframe-diff.csv";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 0);
    const btn = /** @type {HTMLButtonElement} */ ($('[data-act="export"]'));
    btn.disabled = false;
    btn.textContent = "Export CSV";
  }

  // ------------------------------------------------------------ overview

  /** Status bars, legends and metadata line (static until the filter changes). */
  function renderOverview() {
    const S = M.summary;
    const total = Math.max(1, S.total);
    /** @param {number} c */
    const share = (c) => ((100 * c) / total).toLocaleString(LOCALE, { maximumFractionDigits: 1 });
    /** @param {Status} s */
    const off = (s) => (state.statuses.has(s) ? "" : "off");
    // both bars share one scale: segments use flex-grow = count, no gaps
    const seg = STATUS_ORDER.map((s) =>
      S[s] ? `<span class="st-${s} ${off(s)}" style="flex:${S[s]}" title="${esc(label(s))}: ${n(S[s])}"></span>` : "",
    ).join("");
    /** @type {{key: string, title: string, tip: string, count: number, statuses: Status[]}[]} */
    const groups = [
      {
        key: "found",
        title: "Found",
        tip: "Key on both sides (equal + mismatch)",
        count: S.equal + S.mismatch,
        statuses: ["equal", "mismatch"],
      },
      {
        key: "notfound",
        title: "Not found",
        tip: `Key on one side only (only in ${M.L} + only in ${M.R})`,
        count: S.missing_left + S.missing_right,
        statuses: ["missing_left", "missing_right"],
      },
    ];
    /** @param {Status[]} sts */
    const gOff = (sts) => (sts.some((x) => state.statuses.has(x)) ? "" : "off");
    const seg2 = groups
      .map((g) =>
        g.count
          ? `<span class="g-${g.key} ${gOff(g.statuses)}" style="flex:${g.count}" title="${g.title}: ${n(g.count)}"></span>`
          : "",
      )
      .join("");
    const legend = STATUS_ORDER.map(
      (s) => `
      <div class="dfd-stat ${off(s)}" title="${esc(hint(s))}">
        ${badge(s)}<span>${esc(label(s))}</span><b>${n(S[s])}</b><small>${share(S[s])}%</small></div>`,
    ).join("");
    const legend2 = groups
      .map(
        (g) => `
      <div class="dfd-stat ${gOff(g.statuses)}" title="${esc(g.tip)}">
        <span class="dfd-swatch g-${g.key}" aria-hidden="true"></span><span>${g.title}</span><b>${n(g.count)}</b><small>${share(g.count)}%</small></div>`,
      )
      .join("");
    const extra = [];
    if (S.left_only_cols.length) extra.push(`only in ${esc(M.L)}: ${S.left_only_cols.map(esc).join(", ")}`);
    if (S.right_only_cols.length) extra.push(`only in ${esc(M.R)}: ${S.right_only_cols.map(esc).join(", ")}`);
    if (S.ignored?.length) extra.push(`ignored: ${S.ignored.map(esc).join(", ")}`);
    const dupKeys = S.duplicate_keys_left + S.duplicate_keys_right;
    const notice = dupKeys
      ? `<div class="dfd-notice" role="status"><b>Keys are not unique:</b>
          ${[
            [S.duplicate_keys_left, M.L],
            [S.duplicate_keys_right, M.R],
          ]
            .filter(([c]) => c)
            .map(([c, side]) => `${n(Number(c))} key${c === 1 ? " occurs" : "s occur"} more than once in ${esc(side)}`)
            .join(", ")}.
          ${
            S.duplicates === "number"
              ? "Rows with the same key were paired in order."
              : "Rows with the same key were paired by content: identical rows first, then the most similar ones."
          }
          <button class="dfd-link" data-act="dup-only">${state.dupOnly ? "Show all rows" : `Show these ${n(S.duplicate_rows)} rows`}</button></div>`
      : "";
    $(".dfd-overview").innerHTML = `${notice}
      <div class="dfd-bars" aria-hidden="true">
        <div class="dfd-dist">${seg}</div>
        <div class="dfd-dist">${seg2}</div>
      </div>
      <div class="dfd-stats">${legend}</div>
      <div class="dfd-stats">${legend2}</div>
      <div class="dfd-meta">
        <span class="dfd-sidename">${chip("l")}<b>${esc(M.L)}</b> (left) ${n(S.left_rows)} rows</span>
        <span class="dfd-sidename">${chip("r")}<b>${esc(M.R)}</b> (right) ${n(S.right_rows)} rows</span>
        <span>Key: ${S.keys.map((k) => `<code>${esc(k)}</code>`).join(" + ")}</span>
        <span>${S.compared.length} columns compared</span>
        ${extra.map((e) => `<span>${e}</span>`).join("")}
      </div>`;
  }

  // -------------------------------------------------------- status filter

  function statusSummaryText() {
    if (state.statuses.size === STATUS_ORDER.length) return "All";
    if (state.statuses.size === 0) return "None";
    if (state.statuses.size === 3 && !state.statuses.has("equal")) return "All differences";
    return STATUS_ORDER.filter((s) => state.statuses.has(s))
      .map(label)
      .join(", ");
  }

  /** Popover of the status column: sort, presets and one checkbox per status. */
  function statusMenuHtml() {
    const S = M.summary;
    const presets = [
      ["all", "All"],
      ["diff", "All differences"],
      ["missing", "One side only"],
    ];
    const sorted = state.sort && state.sort.col === "status" ? state.sort.dir : 0;
    return `
      <div class="dfd-colmenu status" role="dialog" aria-label="Sort and filter by status">
        <div class="dfd-cm-title"><b>Status</b></div>
        <div class="dfd-cm-sort">
          <button data-cm="asc" class="${sorted === 1 ? "on" : ""}">↑ Sort ascending</button>
          <button data-cm="desc" class="${sorted === -1 ? "on" : ""}">↓ Sort descending</button>
          ${sorted ? '<button data-cm="unsort">Clear sort</button>' : ""}
        </div>
        <div role="group" aria-label="Filter by status">
          <div class="dfd-presets">${presets.map(([k, t]) => `<button data-preset="${k}">${t}</button>`).join("")}</div>
          ${STATUS_ORDER.map(
            (s) => `
            <label class="dfd-opt">
              <input type="checkbox" data-status="${s}" ${state.statuses.has(s) ? "checked" : ""}>
              ${badge(s)}
              <span class="dfd-opt-text"><span>${esc(label(s))}</span><small>${esc(hint(s))}</small></span>
              <b>${n(S[s])}</b>
              <button class="dfd-only" data-only="${s}" title="Show only ${esc(label(s))}">only</button>
            </label>`,
          ).join("")}
          ${
            S.duplicate_rows
              ? `<label class="dfd-opt dfd-opt-dup">
              <input type="checkbox" data-dup-only ${state.dupOnly ? "checked" : ""}>
              <span class="dfd-dup" aria-hidden="true">×</span>
              <span class="dfd-opt-text"><span>Duplicate keys only</span><small>Rows whose key occurs more than once on a side</small></span>
              <b>${n(S.duplicate_rows)}</b>
            </label>`
              : ""
          }
        </div>
      </div>`;
  }

  /** Update the checkboxes of an open status menu in place (keeps the focus). */
  function syncStatusMenu() {
    const dup = root.querySelector(".dfd-colmenu [data-dup-only]");
    if (dup instanceof HTMLInputElement) dup.checked = state.dupOnly;
    for (const cb of root.querySelectorAll(".dfd-colmenu [data-status]")) {
      if (cb instanceof HTMLInputElement)
        cb.checked = isStatus(cb.dataset.status) && state.statuses.has(/** @type {Status} */ (cb.dataset.status));
    }
  }

  function renderToolbarDyn() {
    const parts = [];
    if (state.statuses.size !== STATUS_ORDER.length)
      parts.push(`<span class="dfd-pillfilter col">
        <button class="dfd-pill-edit" data-colmenu="status" title="Edit status filter"><b>Status</b> ${esc(statusSummaryText())}</button>
        <button data-act="clear-status" aria-label="Remove status filter">×</button></span>`);
    if (state.dupOnly)
      parts.push(`<span class="dfd-pillfilter col">
        <button class="dfd-pill-edit" data-colmenu="status" title="Edit status filter"><b>Duplicate keys</b> only</button>
        <button data-act="clear-dup" aria-label="Remove duplicate key filter">×</button></span>`);
    for (const [i, f] of state.filters) {
      const c = M.columns[i];
      parts.push(`<span class="dfd-pillfilter col">
        <button class="dfd-pill-edit" data-colmenu="${i}" title="Edit filter"><b>${esc(c.name)}</b> ${esc(describeFilter(c.ftype, f))}</button>
        <button data-rmfilter="${i}" aria-label="Remove filter on ${esc(c.name)}">×</button></span>`);
    }
    if (state.filters.size > 1)
      parts.push(`<button class="dfd-link" data-act="clear-filters">Clear all filters</button>`);
    if (state.diffColumn !== null)
      parts.push(`<span class="dfd-pillfilter">${state.diffEqual ? "Equal" : "Differs"} in <b>${esc(M.columns[state.diffColumn].name)}</b>
        <button data-act="clear-colfilter" aria-label="Remove column filter">×</button></span>`);
    if (selected.size)
      parts.push(`<span class="dfd-pillfilter sel">${n(selected.size)} selected
        <button data-act="clear-sel" aria-label="Clear selection">×</button></span>`);
    $(".dfd-toolbar-dyn").innerHTML = parts.join("");
  }

  function renderError() {
    const box = $(".dfd-error");
    box.hidden = !error;
    box.textContent = error ? `Could not load rows – ${error}` : "";
  }

  // ---------------------------------------------------------------- table

  function visibleCols() {
    const idx = M.columns.map((_, i) => i);
    if (!state.onlyDiffCols) return idx;
    // columns the view is filtered on stay: "=" on a column means no difference in it
    const hit = new Set([...view.diffColumns, ...state.filters.keys()]);
    if (state.diffColumn !== null) hit.add(state.diffColumn);
    return idx.filter((i) => M.columns[i].kind !== "compared" || hit.has(i));
  }

  /**
   * Header cell of column `i`: name, sort/filter button, dtype and match rate.
   * @param {number} i
   */
  function th(i) {
    const c = M.columns[i];
    const arrow = state.sort && state.sort.col === i ? (state.sort.dir > 0 ? "↑" : "↓") : "";
    let tag = "";
    if (c.kind === "key") tag = `<span class="dfd-tag">key</span>`;
    else if (c.kind === "left_only") tag = `<span class="dfd-tag dfd-sidename">only ${chip("l")}${esc(M.L)}</span>`;
    else if (c.kind === "right_only") tag = `<span class="dfd-tag dfd-sidename">only ${chip("r")}${esc(M.R)}</span>`;
    let stats = "";
    if (c.kind === "compared") {
      // shares relative to the rows whose key exists on both sides
      const found = M.summary.found;
      const mm = c.mismatches,
        eq = found - mm;
      const on = state.diffColumn === i;
      const tip = `${n(eq)} of ${n(found)} matched rows equal (${pct(eq, found)}), ${n(mm)} differ (${pct(mm, found)})`;
      stats = `<div class="dfd-colstats" title="${esc(tip)}">
        <span class="dfd-bar2" aria-hidden="true"><span class="eq" style="flex:${eq}"></span><span class="mm" style="flex:${mm}"></span></span>
        ${
          eq
            ? `<button class="dfd-delta eq ${on && state.diffEqual ? "on" : ""}" data-diffcol="${i}" data-eq
                  title="${esc(tip)}\nClick to show only rows that are equal in “${esc(c.name)}”">${pct(eq, found)} =</button>`
            : `<span class="dfd-muted">0% =</span>`
        }
        ${
          mm
            ? `<button class="dfd-delta ${on && !state.diffEqual ? "on" : ""}" data-diffcol="${i}"
                  title="${esc(tip)}\nClick to show only rows that differ in “${esc(c.name)}”">${pct(mm, found)} ≠</button>`
            : `<span class="dfd-muted">0% ≠</span>`
        }
      </div>`;
    }
    return `<th class="${c.numeric ? "num" : ""}">
      <div class="dfd-th-top">
        <button class="dfd-sort" data-sort="${i}" title="Sort">${esc(c.name)}<span class="dfd-arrow">${arrow}</span></button>
        <button class="dfd-colbtn ${state.filters.has(i) ? "on" : ""} ${state.colMenu === i ? "open" : ""}" data-colmenu="${i}"
          aria-label="Sort and filter ${esc(c.name)}" aria-haspopup="dialog" title="Sort and filter">${FUNNEL}</button>
      </div>
      <div class="dfd-colmeta"><span class="dfd-dtype">${esc(c.dtype)}</span>${tag}</div>${stats}</th>`;
  }

  /**
   * Body cell of column `i`; differing cells show both values.
   * @param {Row} row
   * @param {number} i
   */
  function td(row, i) {
    const c = M.columns[i];
    const cls = [c.numeric ? "num" : "", c.kind === "key" ? "key" : ""].join(" ");
    // a column can only differ in rows that exist on both sides
    if (row.l && row.r && row.d.includes(i)) {
      const l = row.l[i];
      const r = row.r[i];
      const d = delta(l, r);
      const inv = onlyInvisible(l, r);
      const title = `${inv ? "Differs only in invisible characters\n" : ""}${M.L}: ${literal(l)}\n${M.R}: ${literal(r)}${d ? `\nΔ ${d}` : ""}`;
      return `<td class="${cls} diff ${inv ? "inv-only" : ""}" title="${esc(title)}">
        <div class="dfd-val"><span class="dfd-v">${cellHtml(l, row, i)}</span>${chip("l")}</div>
        <div class="dfd-val"><span class="dfd-v">${cellHtml(r, row, i)}</span>${chip("r")}</div></td>`;
    }
    const v = cellValue(row, i);
    return v === undefined ? `<td class="${cls} absent">–</td>` : `<td class="${cls}">${cellHtml(v, row, i)}</td>`;
  }

  function renderTable() {
    if (state.flyover) {
      // a new page may hold a fresh copy of the row – or no longer hold it
      const id = state.flyover.row.id;
      const row = view.rows.find((r) => r.id === id);
      if (row) state.flyover.row = row;
      else closeFlyover();
    }
    const cols = visibleCols();
    const rows = view.rows;
    const pages = Math.max(1, Math.ceil(view.filtered / state.pageSize));
    const start = state.page * state.pageSize;
    const allSel = rows.length > 0 && rows.every((r) => selected.has(r.id));
    const stArrow = state.sort && state.sort.col === "status" ? (state.sort.dir > 0 ? "↑" : "↓") : "";
    const activeId = state.detail ? state.detail.row.id : null;

    let body;
    if (rows.length) {
      body = rows
        .map(
          (r) => `
        <tr data-id="${r.id}" class="st-${r.s} ${selected.has(r.id) ? "sel" : ""} ${activeId === r.id ? "active" : ""}">
          <td class="cb"><input type="checkbox" data-sel="${r.id}" ${selected.has(r.id) ? "checked" : ""} aria-label="Select row"></td>
          <td class="status" title="${esc(`${label(r.s)}: ${hint(r.s)}`)}"><span class="dfd-sr">${esc(label(r.s))}</span><span class="dfd-st">${badge(r.s)}${dupSlot(r)}</span></td>
          ${cols.map((i) => td(r, i)).join("")}
        </tr>`,
        )
        .join("");
    } else if (loading) {
      body = `<tr><td class="dfd-empty" colspan="${cols.length + 2}">Loading…</td></tr>`;
    } else {
      body = `<tr><td class="dfd-empty" colspan="${cols.length + 2}">No rows match these filters.
        <button class="dfd-link" data-act="reset">Reset filters</button></td></tr>`;
    }

    $(".dfd-scroll").innerHTML = `
      <table class="${rows.length === pageRows() ? "full" : ""}">
        <thead><tr>
          <th class="cb"><input type="checkbox" data-act="page-all" ${allSel ? "checked" : ""} aria-label="Select all rows on this page"></th>
          <th class="status">
            <div class="dfd-th-top">
              <button class="dfd-sort" data-sort="status" title="Sort">Status<span class="dfd-arrow">${stArrow}</span></button>
              <button class="dfd-colbtn ${state.statuses.size !== STATUS_ORDER.length || state.dupOnly ? "on" : ""} ${state.colMenu === "status" ? "open" : ""}" data-colmenu="status"
                aria-label="Sort and filter by status" aria-haspopup="dialog" title="Sort and filter">${FUNNEL}</button>
            </div>
          </th>
          ${cols.map(th).join("")}
        </tr></thead>
        <tbody>${body}</tbody>
      </table>`;

    const end = Math.min(view.filtered, start + rows.length);
    const of = view.filtered !== M.summary.total ? ` (filtered from ${n(M.summary.total)})` : "";
    $(".dfd-footer").innerHTML = `
      <span>${view.filtered ? `${n(start + 1)}–${n(end)}` : 0} of ${n(view.filtered)}${of}</span>
      <span class="dfd-spacer"></span>
      <label>Rows per page
        <select data-act="page-size">${PAGE_SIZES.map((x) => `<option ${x === state.pageSize ? "selected" : ""}>${x}</option>`).join("")}</select>
      </label>
      <span class="dfd-pager">
        <button data-page="first" ${state.page === 0 ? "disabled" : ""} aria-label="First page">«</button>
        <button data-page="prev" ${state.page === 0 ? "disabled" : ""} aria-label="Previous page">‹</button>
        <span>Page ${n(state.page + 1)} of ${n(pages)}</span>
        <button data-page="next" ${state.page >= pages - 1 ? "disabled" : ""} aria-label="Next page">›</button>
        <button data-page="last" ${state.page >= pages - 1 ? "disabled" : ""} aria-label="Last page">»</button>
      </span>`;
    renderToolbarDyn();
    fixTableHeight();
  }

  /**
   * Rows the table has room for: a full page, or all rows of a smaller diff. The
   * total never changes while filtering, so neither does the height.
   */
  const pageRows = () => Math.max(1, Math.min(state.pageSize, M.summary.total));

  /**
   * Height for `pageRows()` rows, whatever the view holds: header + rows × row
   * height (`--dfd-row-h`) + borders and a horizontal scrollbar.
   */
  function fixTableHeight() {
    const scroll = $(".dfd-scroll");
    const head = scroll.querySelector("thead");
    const rowH = Number.parseFloat(getComputedStyle(root).getPropertyValue("--dfd-row-h")) || 48;
    const headH = head ? head.getBoundingClientRect().height : 0;
    if (!headH) return; // not laid out yet (hidden or detached)
    const chrome = scroll.offsetHeight - scroll.clientHeight;
    // round up: the header can be a fractional height, and a scrollbar must never appear
    const height = `${Math.ceil(headH + pageRows() * rowH + chrome)}px`;
    if (scroll.style.height !== height) scroll.style.height = height;
  }
  // A resize can add or remove the horizontal scrollbar (or lay out a widget that
  // was hidden): measure again in the next frame, which also keeps the observer
  // from reporting a loop when the new height resizes the box once more.
  const resizeObserver = new ResizeObserver(() => requestAnimationFrame(fixTableHeight));

  // ---------------------------------------------------------- detail view

  function renderDetail() {
    const wrap = $(".dfd-detail-wrap");
    if (!state.detail) {
      wrap.innerHTML = "";
      return;
    }
    const { row, pos } = state.detail;
    const key = keyOf(row)
      .map(([k, v]) => `<code>${esc(k)} = ${F(v)}</code>`)
      .join(" ");
    const lines = M.columns
      .map((c, i) => {
        const isDiff = row.d.includes(i);
        const lv = c.kind === "right_only" || !row.l ? undefined : row.l[i];
        const rv = c.kind === "left_only" || !row.r ? undefined : row.r[i];
        /** @param {Value|undefined} v */
        const show = (v) => (v === undefined ? '<span class="dfd-absent">–</span>' : F(v));
        const d = isDiff ? delta(lv, rv) : null;
        const mark =
          c.kind === "key"
            ? "key"
            : isDiff
              ? d
                ? `Δ ${esc(d)}`
                : onlyInvisible(lv, rv)
                  ? "≠ invisible characters only"
                  : "≠"
              : c.kind === "compared" && row.l && row.r
                ? "="
                : "";
        let [lh, rh] = [show(lv), show(rv)];
        let below = "";
        if (state.textDiff && isDiff && typeof lv === "string" && typeof rv === "string" && lv && rv) {
          if (linesOf(lv) || linesOf(rv)) {
            // texts: first lines here, the unified diff in a row of its own below
            const ud = diffHtml(lv, rv);
            if (ud) {
              /** @param {string} v */
              const first = (v) =>
                linesOf(v) ? `${firstLine(v, state.showInvisible)} <span class="dfd-muted">⋯</span>` : F(v);
              [lh, rh] = [first(lv), first(rv)];
              below = `<tr class="dfd-udrow"><td colspan="4">${ud}</td></tr>`;
            }
          } else {
            const w = wordDiff(lv, rv)?.marks;
            if (w)
              [lh, rh] = [diffLineHtml(lv, w[0], state.showInvisible), diffLineHtml(rv, w[1], state.showInvisible)];
          }
        }
        return `<tr class="${isDiff ? "diff" : ""}"><th>${esc(c.name)}</th>
        <td class="${c.numeric ? "num" : ""}">${lh}</td><td class="${c.numeric ? "num" : ""}">${rh}</td>
        <td class="mark">${mark}</td></tr>${below}`;
      })
      .join("");
    wrap.innerHTML = `
      <section class="dfd-detail st-${row.s}" aria-label="Row details">
        <header>
          <span class="dfd-pill">${badge(row.s)}${esc(label(row.s))}</span>
          ${row.k ? `<span class="dfd-pill">${dupChip(row)}${esc(dupText(row.k))}</span>` : ""}
          <span class="dfd-detail-key">${key}</span>
          <span class="dfd-spacer"></span>
          <span class="dfd-pager">
            <button data-detail="prev" ${pos <= 0 ? "disabled" : ""} aria-label="Previous row">‹</button>
            <span>${pos >= 0 ? `${n(pos + 1)} / ${n(view.filtered)}` : "outside filter"}</span>
            <button data-detail="next" ${pos < 0 || pos >= view.filtered - 1 ? "disabled" : ""} aria-label="Next row">›</button>
          </span>
          <button class="dfd-close" data-act="close-detail" aria-label="Close details">×</button>
        </header>
        <div class="dfd-detail-scroll">
          <table><thead><tr><th>Column</th><th><span class="dfd-sidename">${chip("l")}${esc(M.L)} (left)</span></th><th><span class="dfd-sidename">${chip("r")}${esc(M.R)} (right)</span></th><th></th></tr></thead>
          <tbody>${lines}</tbody></table>
        </div>
      </section>`;
  }

  /**
   * Move the detail panel by one row, fetching the neighbouring page if needed.
   * @param {1|-1} dirn
   */
  function stepDetail(dirn) {
    if (!state.detail || state.detail.pos < 0) return;
    const target = state.detail.pos + dirn;
    if (target < 0 || target >= view.filtered) return;
    const pageOfTarget = Math.floor(target / state.pageSize);
    if (pageOfTarget === state.page) {
      const row = view.rows[target - state.page * state.pageSize];
      state.detail = { row, pos: target };
      renderTable();
      renderDetail();
    } else {
      state.page = pageOfTarget;
      pendingDetail = dirn > 0 ? "first" : "last";
      query();
    }
  }

  // -------------------------------------------------------------- flyover

  /**
   * Render the full text of the cell in `state.flyover`. A difference shows both
   * sides next to each other, with the lines that differ highlighted.
   */
  function renderFlyover() {
    const wrap = $(".dfd-flyover-wrap");
    const fly = state.flyover;
    if (!fly) {
      wrap.innerHTML = "";
      return;
    }
    const { row, col } = fly;
    const c = M.columns[col];
    const both = !!(row.l && row.r && row.d.includes(col));
    /** @type {["l"|"r"|null, Value|undefined][]} */
    const sides = both
      ? [
          ["l", row.l?.[col]],
          ["r", row.r?.[col]],
        ]
      : [[null, cellValue(row, col)]];
    const all = sides.map(([, v]) => linesOf(v));
    /**
     * Full text of one side, one numbered line per line.
     * @param {Value|undefined} v
     * @param {number} k Index of the side.
     */
    const text = (v, k) => {
      const lines = all[k];
      if (!lines)
        return `<div class="dfd-fly-text"><div class="dfd-fly-line ${both ? "diff" : ""}"><span class="dfd-ln"></span><span>${F(v)}</span></div></div>`;
      const other = both ? all[1 - k] : null;
      const breaks = breaksOf(/** @type {string} */ (v));
      const html = lines
        .map((line, j) => {
          const differs = both && (!other || other[j] !== line);
          const brk = state.showInvisible && j < breaks.length ? Array.from(breaks[j]).map(invMark).join("") : "";
          return `<div class="dfd-fly-line ${differs ? "diff" : ""}"><span class="dfd-ln">${j + 1}</span><span>${fmtLine(line, state.showInvisible)}${brk}</span></div>`;
        })
        .join("");
      return `<div class="dfd-fly-text">${html}</div>`;
    };
    const [lv, rv] = [row.l?.[col], row.r?.[col]];
    const unified = both && typeof lv === "string" && typeof rv === "string" ? diffHtml(lv, rv) : null;
    const body =
      unified ??
      sides
        .map(
          ([side, v], k) => `<section>
          ${side ? `<div class="dfd-fly-head dfd-sidename">${chip(side)}${esc(side === "l" ? M.L : M.R)}</div>` : ""}
          ${text(v, k)}</section>`,
        )
        .join("");
    wrap.innerHTML = `
      <div class="dfd-flyover ${both ? "two" : ""}" role="dialog" aria-label="${unified ? "Differences" : "Full text"} of ${esc(c.name)}">
        <div class="dfd-cm-title"><b>${esc(c.name)}</b><span class="dfd-dtype">${esc(c.dtype)}</span>
          <span class="dfd-spacer"></span>
          <button class="dfd-close" data-act="close-flyover" aria-label="Close">×</button></div>
        ${unified ? body : `<div class="dfd-fly-sides">${body}</div>`}
      </div>`;
    const box = /** @type {HTMLElement} */ (wrap.firstElementChild);
    const btn = root.querySelector(`.dfd-scroll [data-more="${row.id}:${col}"]`);
    if (btn) {
      const rr = root.getBoundingClientRect();
      const br = btn.getBoundingClientRect();
      box.style.top = `${br.bottom - rr.top + 4}px`;
      box.style.left = `${Math.max(0, Math.min(br.left - rr.left, rr.width - box.offsetWidth))}px`;
    }
    box.querySelector("button")?.focus({ preventScroll: true });
  }

  /** Mark the button whose flyover is open. */
  function syncMoreButtons() {
    const open = state.flyover ? `${state.flyover.row.id}:${state.flyover.col}` : null;
    for (const b of root.querySelectorAll(".dfd-more")) {
      if (b instanceof HTMLElement) b.classList.toggle("open", b.dataset.more === open);
    }
  }

  /**
   * Toggle the flyover of a cell, given as `"<row id>:<column>"`.
   * @param {string} ref
   */
  function openFlyover(ref) {
    const [id, col] = ref.split(":").map(Number);
    const row = view.rows.find((r) => r.id === id);
    const same = state.flyover?.row.id === id && state.flyover.col === col;
    closeColMenu();
    state.flyover = same || !row ? null : { row, col };
    renderFlyover();
    syncMoreButtons();
  }

  function closeFlyover() {
    if (state.flyover === null) return;
    state.flyover = null;
    renderFlyover();
    syncMoreButtons();
  }

  // ---------------------------------------------------------- column menu

  /**
   * Render the sort/filter popover of the column in `state.colMenu`.
   * @param {Element} [anchor] Element to position the popover under.
   */
  function renderColMenu(anchor) {
    const wrap = $(".dfd-colmenu-wrap");
    const i = state.colMenu;
    if (i === null) {
      wrap.innerHTML = "";
      return;
    }
    if (i === "status") {
      wrap.innerHTML = statusMenuHtml();
      placeMenu(anchor);
      $opt(".dfd-colmenu input")?.focus({ preventScroll: true });
      return;
    }
    const c = M.columns[i];
    const t = c.ftype;
    const f = state.filters.get(i) ?? { op: OPS[t][0][0], a: "", b: "" };
    const inType = t === "number" ? 'type="number" step="any"' : t === "datetime" ? 'type="date"' : 'type="text"';
    const sorted = state.sort && state.sort.col === i ? state.sort.dir : 0;
    wrap.innerHTML = `
      <div class="dfd-colmenu" role="dialog" aria-label="Sort and filter ${esc(c.name)}">
        <div class="dfd-cm-title"><b>${esc(c.name)}</b><span class="dfd-dtype">${esc(c.dtype)}</span></div>
        <div class="dfd-cm-sort">
          <button data-cm="asc" class="${sorted === 1 ? "on" : ""}">↑ Sort ascending</button>
          <button data-cm="desc" class="${sorted === -1 ? "on" : ""}">↓ Sort descending</button>
          ${sorted ? '<button data-cm="unsort">Clear sort</button>' : ""}
        </div>
        <div class="dfd-cm-filter ${NO_VALUE.has(f.op) ? "novalue" : ""} ${f.op === "between" ? "two" : ""}">
          <label class="dfd-cm-label">Filter
            <select data-f="op">${OPS[t].map(([k, lbl]) => `<option value="${k}" ${k === f.op ? "selected" : ""}>${esc(lbl)}</option>`).join("")}</select>
          </label>
          <div class="dfd-cm-values">
            <input data-f="a" ${inType} value="${esc(f.a)}" placeholder="${t === "number" ? "min" : t === "datetime" ? "" : "value"}" aria-label="Value">
            <span class="dfd-cm-and">and</span>
            <input data-f="b" ${inType} value="${esc(f.b)}" placeholder="${t === "number" ? "max" : ""}" aria-label="Upper value">
          </div>
          ${c.kind === "compared" ? `<p class="dfd-cm-hint">A differing row matches if the ${esc(M.L)} <i>or</i> the ${esc(M.R)} value matches.</p>` : ""}
        </div>
        <div class="dfd-cm-actions">
          <button data-cm="clear" ${state.filters.has(i) ? "" : "disabled"}>Clear filter</button>
          <button data-cm="apply" class="primary">Apply</button>
        </div>
      </div>`;
    const menu = placeMenu(anchor);
    const first = menu?.querySelector(NO_VALUE.has(f.op) ? "select" : 'input[data-f="a"]');
    if (first instanceof HTMLElement) first.focus({ preventScroll: true });
  }

  /**
   * Position the open popover under `anchor` (default: its header button).
   * @param {Element} [anchor]
   * @returns {HTMLElement|null} The popover.
   */
  function placeMenu(anchor) {
    const menu = $opt(".dfd-colmenu");
    const btn = anchor ?? root.querySelector(`.dfd-scroll [data-colmenu="${state.colMenu}"]`);
    if (menu && btn) {
      const rr = root.getBoundingClientRect();
      const br = btn.getBoundingClientRect();
      const w = menu.offsetWidth || 280;
      menu.style.top = `${br.bottom - rr.top + 4}px`;
      menu.style.left = `${Math.max(0, Math.min(br.left - rr.left, rr.width - w))}px`;
    }
    return menu;
  }

  /** Mark the header button whose menu is open. */
  function syncColButtons() {
    for (const b of root.querySelectorAll(".dfd-colbtn")) {
      if (b instanceof HTMLElement) b.classList.toggle("open", b.dataset.colmenu === String(state.colMenu));
    }
  }

  /**
   * Toggle the column menu of column `i` (or of the status column).
   * @param {number|"status"} i
   * @param {Element} anchor
   */
  function openColMenu(i, anchor) {
    closeFlyover();
    state.colMenu = state.colMenu === i ? null : i;
    renderColMenu(anchor);
    syncColButtons();
  }

  function closeColMenu() {
    if (state.colMenu === null) return;
    state.colMenu = null;
    renderColMenu();
    syncColButtons();
  }

  /** Read the popover inputs into `state.filters` (an empty value removes the filter). */
  function applyColMenu() {
    const i = state.colMenu;
    const menu = $opt(".dfd-colmenu");
    if (typeof i !== "number" || !menu) return;
    /** @param {"op"|"a"|"b"} k */
    const get = (k) => {
      const input = menu.querySelector(`[data-f="${k}"]`);
      return input instanceof HTMLInputElement || input instanceof HTMLSelectElement ? input.value.trim() : "";
    };
    const f = { op: get("op"), a: get("a"), b: get("b") };
    const empty = !NO_VALUE.has(f.op) && (f.op === "between" ? !f.a && !f.b : !f.a);
    if (empty) state.filters.delete(i);
    else state.filters.set(i, f);
    closeColMenu();
    changed();
  }

  // ------------------------------------------------------------ actions

  /**
   * Any change to the query: back to page 1 and fetch.
   * @param {{overview?: boolean}} [options] Also re-render the status overview.
   */
  function changed({ overview = false } = {}) {
    state.page = 0;
    if (overview) {
      renderOverview();
      syncStatusMenu();
    }
    renderToolbarDyn();
    query();
  }

  /** @param {Iterable<Status>} statuses */
  const setStatuses = (statuses) => {
    state.statuses = new Set(statuses);
    changed({ overview: true });
  };

  /** Push the checked row ids to Python (the only reactive trait). */
  function syncSelection() {
    model.set(
      "selected_ids",
      [...selected].sort((a, b) => a - b),
    );
    model.save_changes();
  }

  function exportCsv() {
    const btn = /** @type {HTMLButtonElement} */ ($('[data-act="export"]'));
    btn.disabled = true;
    btn.textContent = "Exporting…";
    // own request id space: an export must not invalidate a pending page request
    model.send({ type: "export", req: "export", ...queryPayload() });
  }

  /**
   * Closest ancestor (or self) of `t` matching `selector`, as an HTMLElement.
   * @param {Element} t
   * @param {string} selector
   * @returns {HTMLElement|null}
   */
  const closest = (t, selector) => {
    const found = t.closest(selector);
    return found instanceof HTMLElement ? found : null;
  };

  /** @param {string|undefined} value */
  const isStatus = (value) => /** @type {readonly string[]} */ (STATUS_ORDER).includes(value ?? "");

  // --------------------------------------------------------------- events

  /**
   * One delegated click handler for the whole widget.
   * @param {MouseEvent} e
   */
  function onClick(e) {
    if (!(e.target instanceof Element)) return;
    const t = e.target;
    const sel = closest(t, "[data-sel]");
    if (sel instanceof HTMLInputElement) {
      const id = Number(sel.dataset.sel);
      if (sel.checked) selected.add(id);
      else selected.delete(id);
      syncSelection();
      return renderTable();
    }
    const preset = closest(t, "[data-preset]");
    if (preset) {
      e.preventDefault();
      const p = preset.dataset.preset;
      return setStatuses(
        p === "all"
          ? STATUS_ORDER
          : p === "diff"
            ? ["mismatch", "missing_left", "missing_right"]
            : ["missing_left", "missing_right"],
      );
    }
    const only = closest(t, "[data-only]");
    if (only) {
      e.preventDefault();
      const s = only.dataset.only;
      return isStatus(s) ? setStatuses([/** @type {Status} */ (s)]) : undefined;
    }
    const cm = closest(t, "[data-cm]");
    if (cm && state.colMenu !== null) {
      const i = state.colMenu;
      const a = cm.dataset.cm;
      if (a === "apply") return applyColMenu();
      if (a === "clear" && i !== "status") {
        state.filters.delete(i);
        closeColMenu();
        return changed();
      }
      state.sort = a === "unsort" ? null : { col: i, dir: a === "asc" ? 1 : -1 };
      closeColMenu();
      return changed();
    }
    const cmb = closest(t, "[data-colmenu]");
    if (cmb) return openColMenu(cmb.dataset.colmenu === "status" ? "status" : Number(cmb.dataset.colmenu), cmb);
    const rm = closest(t, "[data-rmfilter]");
    if (rm) {
      state.filters.delete(Number(rm.dataset.rmfilter));
      return changed();
    }
    if (t.closest(".dfd-colmenu")) return;
    const more = closest(t, "[data-more]");
    if (more) return openFlyover(more.dataset.more ?? "");

    const act = closest(t, "[data-act]")?.dataset.act;
    switch (act) {
      case "page-all": {
        const checked = t instanceof HTMLInputElement && t.checked;
        for (const r of view.rows) {
          if (checked) selected.add(r.id);
          else selected.delete(r.id);
        }
        syncSelection();
        return renderTable();
      }
      case "clear-status":
        return setStatuses(STATUS_ORDER);
      case "dup-only":
      case "clear-dup":
        state.dupOnly = act === "dup-only" && !state.dupOnly;
        return changed({ overview: true });
      case "clear-filters":
        state.filters.clear();
        return changed();
      case "clear-colfilter":
        state.diffColumn = null;
        state.diffEqual = false;
        return changed();
      case "clear-sel":
        selected.clear();
        syncSelection();
        return renderTable();
      case "close-flyover":
        return closeFlyover();
      case "close-detail":
        state.detail = null;
        renderTable();
        return renderDetail();
      case "export":
        return exportCsv();
      case "reset":
        Object.assign(state, {
          statuses: new Set(STATUS_ORDER),
          dupOnly: false,
          search: "",
          diffColumn: null,
          diffEqual: false,
        });
        state.filters.clear();
        searchInput.value = "";
        return changed({ overview: true });
    }
    if (act) return;

    const dc = closest(t, "[data-diffcol]");
    if (dc) {
      const i = Number(dc.dataset.diffcol);
      const eq = dc.dataset.eq !== undefined;
      // clicking the active filter turns it off, the other symbol switches to it
      const same = state.diffColumn === i && state.diffEqual === eq;
      state.diffColumn = same ? null : i;
      state.diffEqual = !same && eq;
      return changed();
    }
    const so = closest(t, "[data-sort]");
    if (so) {
      /** @type {Sort["col"]} */
      const col = so.dataset.sort === "status" ? "status" : Number(so.dataset.sort);
      if (!state.sort || state.sort.col !== col) state.sort = { col, dir: 1 };
      else if (state.sort.dir === 1) state.sort.dir = -1;
      else state.sort = null;
      return changed();
    }
    const pg = closest(t, "[data-page]");
    if (pg) {
      const pages = Math.max(1, Math.ceil(view.filtered / state.pageSize));
      /** @type {Record<string, number>} */
      const targets = { first: 0, prev: state.page - 1, next: state.page + 1, last: pages - 1 };
      state.page = targets[pg.dataset.page ?? ""] ?? state.page;
      return query();
    }
    const dn = closest(t, "[data-detail]");
    if (dn) return stepDetail(dn.dataset.detail === "next" ? 1 : -1);
    const tr = closest(t, "tbody tr[data-id]");
    if (tr && $(".dfd-scroll").contains(tr)) {
      const id = Number(tr.dataset.id);
      const idx = view.rows.findIndex((r) => r.id === id);
      state.detail =
        state.detail?.row.id === id || idx < 0 ? null : { row: view.rows[idx], pos: state.page * state.pageSize + idx };
      renderTable();
      renderDetail();
      $opt(".dfd-detail")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  /**
   * Checkboxes and selects (status filter, toggles, page size, filter operator).
   * @param {Event} e
   */
  function onChange(e) {
    const t = e.target;
    if (!(t instanceof HTMLInputElement || t instanceof HTMLSelectElement)) return;
    const status = t.dataset.status;
    if (t instanceof HTMLInputElement && isStatus(status)) {
      const s = new Set(state.statuses);
      if (t.checked) s.add(/** @type {Status} */ (status));
      else s.delete(/** @type {Status} */ (status));
      return setStatuses(s);
    }
    if (t.dataset.f === "op") {
      const box = t.closest(".dfd-cm-filter");
      box?.classList.toggle("novalue", NO_VALUE.has(t.value));
      box?.classList.toggle("two", t.value === "between");
      return;
    }
    const checked = t instanceof HTMLInputElement && t.checked;
    if (t.dataset.dupOnly !== undefined) {
      state.dupOnly = checked;
      return changed({ overview: true });
    }
    switch (t.dataset.act) {
      case "only-diff":
        state.onlyDiffCols = checked;
        return renderTable();
      case "show-inv":
        state.showInvisible = checked;
        renderTable();
        renderFlyover();
        return renderDetail();
      case "text-diff":
        state.textDiff = checked;
        renderFlyover();
        return renderDetail();
      case "page-size":
        state.pageSize = Number(t.value);
        return changed();
    }
  }

  /**
   * Close popovers on clicks outside of them.
   * @param {MouseEvent} e
   */
  function onDocClick(e) {
    const path = e.composedPath();
    if (
      state.flyover !== null &&
      !path.includes($(".dfd-flyover-wrap")) &&
      !path.some((x) => x instanceof HTMLElement && x.dataset.more !== undefined)
    )
      closeFlyover();
    if (
      state.colMenu !== null &&
      !path.includes($(".dfd-colmenu-wrap")) &&
      !path.some((x) => x instanceof HTMLElement && x.dataset.colmenu !== undefined)
    )
      closeColMenu();
  }

  /**
   * Escape closes popovers. Listened for on the document: re-rendering a menu
   * can drop the focus to `<body>`, where a widget-level listener would miss it.
   * @param {KeyboardEvent} e
   */
  function onDocKey(e) {
    if (e.key !== "Escape") return;
    closeColMenu();
    closeFlyover();
  }

  /**
   * Enter applies the column filter.
   * @param {KeyboardEvent} e
   */
  function onKey(e) {
    const t = e.target;
    if (e.key === "Enter" && t instanceof HTMLElement && t.closest(".dfd-colmenu") && t.tagName !== "BUTTON") {
      e.preventDefault();
      applyColMenu();
    }
  }

  /** @type {ReturnType<typeof setTimeout>|undefined} */
  let timer;
  /**
   * Debounced search.
   * @param {Event} e
   */
  function onInput(e) {
    if (e.target !== searchInput) return;
    clearTimeout(timer);
    timer = setTimeout(() => {
      state.search = searchInput.value;
      changed();
    }, SEARCH_DEBOUNCE_MS);
  }

  /** New comparison from Python (e.g. widget re-used): reset and refetch. */
  function onMeta() {
    M = meta();
    Object.assign(state, {
      dupOnly: false,
      diffColumn: null,
      diffEqual: false,
      detail: null,
      colMenu: null,
      flyover: null,
    });
    state.filters.clear();
    renderColMenu();
    renderFlyover();
    changed({ overview: true });
  }
  /** Text diff switched from Python: follow it like the toggle. */
  function onTextDiffFromPython() {
    state.textDiff = model.get("text_diff") ?? true;
    /** @type {HTMLInputElement} */ ($('[data-act="text-diff"]')).checked = state.textDiff;
    renderFlyover();
    renderDetail();
  }
  function onSelectionFromPython() {
    selected = new Set(model.get("selected_ids") || []);
    renderTable();
  }
  const onScroll = () => {
    closeColMenu();
    closeFlyover();
  };

  root.addEventListener("click", onClick);
  root.addEventListener("change", onChange);
  root.addEventListener("input", onInput);
  root.addEventListener("keydown", onKey);
  $(".dfd-scroll").addEventListener("scroll", onScroll, { passive: true });
  resizeObserver.observe($(".dfd-scroll"));
  // web fonts that load after the first render change the header height
  document.fonts?.ready.then(() => fixTableHeight());
  document.addEventListener("click", onDocClick);
  document.addEventListener("keydown", onDocKey);
  model.on("msg:custom", onMessage);
  for (const name of ["columns", "summary", "left_short", "right_short", "left_name", "right_name"]) {
    model.on(`change:${name}`, onMeta);
  }
  model.on("change:selected_ids", onSelectionFromPython);
  model.on("change:text_diff", onTextDiffFromPython);

  renderOverview();
  renderTable();
  root.style.setProperty("--dfd-bg", detectBg(el) || "Canvas");
  query();

  return () => {
    clearTimeout(timer);
    resizeObserver.disconnect();
    document.removeEventListener("click", onDocClick);
    document.removeEventListener("keydown", onDocKey);
    model.off("msg:custom", onMessage);
    for (const name of ["columns", "summary", "left_short", "right_short", "left_name", "right_name"]) {
      model.off(`change:${name}`, onMeta);
    }
    model.off("change:selected_ids", onSelectionFromPython);
    model.off("change:text_diff", onTextDiffFromPython);
  };
}

export default { render };
