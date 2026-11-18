"""The data behind one Plotly figure as a table — "export this graph" without a per-graph recipe.

Asked for by the advisor on 2026-11-17: every graph of the app, the phase ones included, should
be exportable on its own, as a picture and as data, without going through the Export tab.
The picture is made in the browser (see ``app.py``); this module turns the figure itself into a
CSV, so the file holds exactly what is drawn — the smoothed line if the line is smoothed, the
plateau means if the staircase is flattened, the pair of wavelengths chosen on the Phase tab.

Layout of the table

* Curves (scatter, bar): one column per curve, named ``<curve> [<y axis title>]``, next to the x
  column they share, named after the x axis. Curves with different x values (a spectrum and its
  peak markers, a scatter and its fitted line, two panels) come as groups side by side — ``x1: …``
  followed by its ``y1: …`` columns, then ``x2: …`` — padded with empty cells to the longest.
  A curve given on part of another's x values (a fringe that lost lock on some frames) joins that
  curve's group, with empty cells where it has no point.
  Marker labels and error bars follow their curve as ``… label`` and ``… ±``.
* A map or a surface (heatmap, surface, contour): the matrix as drawn, first column the y values,
  the header the x values (the layout of the spectra files themselves). Lines drawn over a map
  are not repeated — they are the tracked fringes of the Export tab.
* 3-D curves: ``x``, ``y``, ``z`` per curve.

Animation frames are ignored: a movie has its own export.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

MATRIX_TYPES = ("heatmap", "heatmapgl", "surface", "contour")
XY_TYPES = ("scatter", "scattergl", "bar")


def _clean(text) -> str:
    """Plain text of a Plotly title or trace name (tags and line breaks removed)."""
    if text is None:
        return ""
    text = re.sub(r"<br\s*/?>", " ", str(text), flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Igor Pro 6 and Excel read a CSV in the system code page: "|φ(1570) − φ(1545)| (rad)" arrives
# as mojibake. The file is therefore plain ASCII; these are the symbols the figures use.
ASCII = {"φ": "phi", "λ": "lambda", "Δ": "delta ", "δ": "delta ", "π": "pi", "σ": "sigma", "µ": "u", "μ": "u",
         "ε": "eps", "°": "deg", "±": "+/-", "−": "-", "–": "-", "—": "-", "·": "-", "×": "x", "→": "->",
         "≈": "~", "≤": "<=", "≥": ">=", "²": "^2", "³": "^3", "…": "...", "▶": ">"}


def ascii_text(text) -> str:
    """Plain-ASCII spelling of a header or label: "λc +31.0 pm/°C" → "lambdac +31.0 pm/degC"."""
    out = "".join(ASCII.get(ch, ch) for ch in str(text))
    out = unicodedata.normalize("NFKD", out).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", out).strip()


def safe_name(text: str, fallback: str = "figure") -> str:
    """A file-name piece: letters, digits, ``-``, ``_`` and ``.`` only."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_.")
    return slug or fallback


def _layout_axis(fig, ref: str):
    """The layout axis object for a reference like "x", "x2" or "y3" (None when absent)."""
    return getattr(fig.layout, f"{ref[0]}axis{ref[1:]}", None)


def _title_of(layout_axis) -> str:
    title = getattr(layout_axis, "title", None) if layout_axis is not None else None
    return _clean(getattr(title, "text", None))


def _axis_title(fig, trace, axis: str) -> str:
    """Title of the x or y axis a trace is drawn on ("" when it has none). Stacked panels that
    share an axis (``matches``) carry the title on one of them only: it is looked up there."""
    ref = getattr(trace, f"{axis}axis", None) or axis            # "x", "x2", "y3", …
    own = _layout_axis(fig, ref)
    text = _title_of(own)
    if text or own is None:
        return text
    partner = getattr(own, "matches", None)
    if partner:
        text = _title_of(_layout_axis(fig, partner))
        if text:
            return text
    for key in fig.layout:                                       # an axis that matches this one
        if key.startswith(f"{axis}axis"):
            other = fig.layout[key]
            if getattr(other, "matches", None) in (ref, partner) and _title_of(other):
                return _title_of(other)
    return ""


def _scene_titles(fig, trace) -> tuple[str, str, str]:
    scene = getattr(fig.layout, getattr(trace, "scene", None) or "scene", None)
    out = []
    for axis in "xyz":
        ax = getattr(scene, f"{axis}axis", None) if scene is not None else None
        title = getattr(ax, "title", None) if ax is not None else None
        out.append(_clean(getattr(title, "text", None)) or axis)
    return tuple(out)


def _values(arr) -> np.ndarray | None:
    """A trace attribute as a 1-D array (numbers where possible); None when empty."""
    if arr is None:
        return None
    out = np.asarray(arr)
    if out.ndim == 0 or out.size == 0:
        return None
    out = out.ravel()
    if out.dtype == object:
        try:
            out = out.astype(float)
        except (TypeError, ValueError):
            pass
    return out


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
        return bool(np.array_equal(a.astype(float), b.astype(float), equal_nan=True))
    return bool(np.array_equal(a, b))


def _increasing(x: np.ndarray) -> bool:
    return x.dtype.kind in "fiu" and len(x) > 1 and bool(np.all(np.diff(x.astype(float)) > 0))


def _onto(values: np.ndarray, x_from: np.ndarray, x_to: np.ndarray) -> np.ndarray:
    """``values`` given at ``x_from`` placed at the same x in ``x_to`` (empty elsewhere)."""
    return pd.Series(values, index=x_from).reindex(x_to).to_numpy()


def _join(group: dict, x: np.ndarray, cols: list) -> bool:
    """Add curves to a group when their x values are the group's, a part of them (a fringe that
    lost lock on some frames) or contain them; False when they do not fit."""
    gx = group["x"]
    if _same(gx, x):
        group["cols"] += cols
        return True
    if not (_increasing(gx) and _increasing(x)):
        return False
    if len(x) <= len(gx) and bool(np.isin(x, gx).all()):
        group["cols"] += [(header, _onto(values, x, gx)) for header, values in cols]
        return True
    if len(x) > len(gx) and bool(np.isin(gx, x).all()):
        group["cols"] = [(header, _onto(values, gx, x)) for header, values in group["cols"]] + cols
        group["x"] = x
        return True
    return False


def _unique(names: list[str]) -> list[str]:
    """Make column headers unique: a repeated name gets " (2)", " (3)", …"""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


def _matrix_table(fig, trace) -> pd.DataFrame:
    z = np.asarray(trace.z, dtype=float)
    x = _values(trace.x)
    y = _values(trace.y)
    x = np.arange(z.shape[1]) if x is None else x
    y = np.arange(z.shape[0]) if y is None else y
    if trace.type == "surface":
        x_title, y_title, _ = _scene_titles(fig, trace)
    else:
        x_title, y_title = _axis_title(fig, trace, "x") or "x", _axis_title(fig, trace, "y") or "y"
    table = pd.DataFrame(z, columns=[f"{v:.8g}" if isinstance(v, (float, np.floating)) else str(v) for v in x])
    table.insert(0, f"{y_title} \\ {x_title}", y)
    return table


def figure_table(fig) -> pd.DataFrame:
    """The data drawn in a Plotly figure as one table (see the module docstring for its layout)."""
    traces = list(fig.data)
    for trace in traces:
        if trace.type in MATRIX_TYPES and trace.z is not None and np.asarray(trace.z).ndim == 2:
            return _matrix_table(fig, trace)

    groups: list[dict] = []         # {"x": array, "title": str, "cols": [(header, array), …]}
    for number, trace in enumerate(traces, 1):
        name = _clean(trace.name) or f"trace {number}"
        if trace.type == "scatter3d":
            x, y, z = _values(trace.x), _values(trace.y), _values(trace.z)
            if x is None or y is None or z is None:
                continue
            tx, ty, tz = _scene_titles(fig, trace)
            groups.append(dict(x=x, title=f"{name} [{tx}]", solo=True,
                               cols=[(f"{name} [{ty}]", y), (f"{name} [{tz}]", z)]))
            continue
        if trace.type not in XY_TYPES:
            continue
        x, y = _values(trace.x), _values(trace.y)
        if y is None:
            continue
        if x is None:
            x = np.arange(len(y))
        if len(x) != len(y):
            continue
        y_title = _axis_title(fig, trace, "y")
        header = name if not y_title or y_title == name else f"{name} [{y_title}]"
        cols = [(header, y)]
        text = _values(trace.text) if not isinstance(trace.text, str) else None
        if text is not None and len(text) == len(y):
            cols.append((f"{name} label", text))
        error = getattr(trace, "error_y", None)
        err = _values(getattr(error, "array", None)) if error is not None else None
        if err is not None and len(err) == len(y):
            cols.append((f"{name} ±", err))
        x_title = _axis_title(fig, trace, "x") or "x"
        for group in groups:
            if not group.get("solo") and group["title"] == x_title and _join(group, x, cols):
                break
        else:
            groups.append(dict(x=x, title=x_title, cols=cols))

    if not groups:
        return pd.DataFrame()
    many = len(groups) > 1
    headers, columns = [], []
    for g, group in enumerate(groups, 1):
        solo = group.get("solo")
        headers.append(group["title"] if solo or not many else f"x{g}: {group['title']}")
        columns.append(group["x"])
        for header, values in group["cols"]:
            headers.append(header if solo or not many else f"y{g}: {header}")
            columns.append(values)
    length = max(len(c) for c in columns)
    table = pd.DataFrame({h: pd.Series(c).reindex(range(length)) for h, c in zip(_unique(headers), columns)})
    return table


def figure_csv(fig) -> str:
    """``figure_table`` as CSV text: empty cells for missing values, 10 significant digits, and
    plain ASCII headers and labels (``ascii_text``) so Igor and Excel read it as it is."""
    table = figure_table(fig)
    table.columns = _unique([ascii_text(c) for c in table.columns])
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda v: ascii_text(v) if isinstance(v, str) else v)
    return table.to_csv(index=False, float_format="%.10g")
