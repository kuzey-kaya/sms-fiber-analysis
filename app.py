"""FringeLab -- interactive explorer for fiber-sensor spectra (draft).

Load any spectra CSV the loader understands, adjust the detection / tracking
settings in the sidebar and see the spectral map, the detected peaks, the
tracked fringes and the calibration update immediately.

Start it with the launcher for your system (run_app.bat / run_app.command) or
    python launch.py
or, for development,
    streamlit run app.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sms_analysis import (curvature_test, detect_extrema, fit_sensitivity, load_sms_csv,
                          track_all_extrema)
from sms_analysis.io import SpectraSet

APP_NAME = "FringeLab"
ROOT = Path(__file__).resolve().parent
EXAMPLE_CSV = ROOT / "data" / "data.csv"

RED, BLUE = "#ff0000", "#0000ff"        # Igor Pro trace colours
PEAK_COLOR, DIP_COLOR = "#C4402F", "#2E5FA3"
LINEAR_COLOR, QUAD_COLOR = "#8c8c8c", "#2E5FA3"   # calibration models
MAX_MAP_ROWS, MAX_MAP_COLS = 1200, 2000  # display resolution of the heatmap
MAX_3D_ROWS, MAX_3D_COLS = 150, 300

ICON_PNG = ROOT / "assets" / "fringelab_icon.png"      # scripts/make_icon.py draws it

st.set_page_config(page_title=APP_NAME, page_icon=str(ICON_PNG) if ICON_PNG.exists() else "〰️", layout="wide",
                   menu_items={"About": f"{APP_NAME} — fringe analysis for SMS fiber sensors."})


# --------------------------------------------------------------------------
# look & feel
# --------------------------------------------------------------------------
STYLE = """
<style>
.block-container {padding-top: 3.4rem; padding-bottom: 2.5rem;}
.stAppDeployButton {display: none;}
.fl-hero {display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; margin-bottom:0;}
.fl-hero h1 {font-size:2.4rem; font-weight:700; letter-spacing:-0.02em; margin:0; padding:0; line-height:1.1;}
.fl-hero .fl-tag {font-size:0.95rem; opacity:0.62;}
/* the fringe strip: the first spectrum of the loaded run, drawn as bands */
.fl-strip {height:18px; border-radius:3px; margin:12px 0 18px 0; width:100%;}
.fl-strip-note {font-size:0.74rem; opacity:0.55; margin:-12px 0 16px 0;}
/* key figures: light tiles, wrap on narrow screens */
.fl-figs {display:flex; flex-wrap:wrap; gap:10px; margin:0 0 14px 0;}
.fl-fig {flex:1 1 150px; min-width:140px; padding:10px 14px; border-radius:8px;
         border:1px solid rgba(128,128,128,0.3); background:rgba(128,128,128,0.06);}
.fl-fig .v {font-size:1.5rem; font-weight:650; line-height:1.3; font-variant-numeric: tabular-nums;}
.fl-fig .l {font-size:0.78rem; opacity:0.7;}
.fl-fig .n {font-size:0.74rem; opacity:0.55;}
.stTabs [data-baseweb="tab"] {font-weight:600; padding-left:4px; padding-right:4px;}
.fl-brand {font-size:1.5rem; font-weight:700; letter-spacing:-0.02em; margin-bottom:0; line-height:1.2;}
.fl-brand small {font-size:0.8rem; font-weight:400; opacity:0.65; display:block; margin-top:4px;}
.fl-brand .fl-strip {height:8px; margin:8px 0 2px 0;}
.fl-foot {font-size:0.78rem; opacity:0.55; margin-top:28px; padding-top:10px;
          border-top:1px solid rgba(128,128,128,0.25);}
</style>
"""
STRIP_DARK, STRIP_LIGHT = (0x14, 0x2A, 0x4E), (0xDD, 0xE7, 0xF5)   # band colours, same in both themes


def fringe_strip(wavelengths: np.ndarray, spectrum: np.ndarray, stops: int = 240) -> str:
    """One spectrum as a horizontal band pattern (CSS gradient), like one row of the map."""
    y = np.asarray(spectrum, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return ""
    lo, hi = np.percentile(y[ok], [2, 98])
    z = np.clip((np.nan_to_num(y, nan=lo) - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    idx = np.linspace(0, len(z) - 1, min(stops, len(z))).astype(int)
    parts = []
    for i, k in enumerate(idx):
        c = [int(round(a + (b - a) * z[k])) for a, b in zip(STRIP_DARK, STRIP_LIGHT)]
        parts.append(f"rgb({c[0]},{c[1]},{c[2]}) {100 * i / (len(idx) - 1):.2f}%")
    title = f"first spectrum, {wavelengths.min():.0f}–{wavelengths.max():.0f} nm: bright and dark bands are the fringes"
    return (f'<div class="fl-strip" title="{title}" style="background:linear-gradient(90deg,'
            + ",".join(parts) + ')"></div>')


REPO_URL = "https://github.com/kuzey-kaya/sms-fiber-analysis"


def inject_style() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)


def brand_icon() -> str:
    """The icon inlined for the sidebar brand line (empty if the PNG is missing)."""
    if not ICON_PNG.exists():
        return ""
    data = base64.b64encode(ICON_PNG.read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{data}" alt="" '
            'style="height:30px;width:30px;vertical-align:-7px;margin-right:9px;border-radius:7px">')


def hero(tagline: str, strip: str = "", strip_note: str = "") -> None:
    """Title line, then the loaded run's first spectrum as a band of fringes."""
    html = f'<div class="fl-hero"><h1>{APP_NAME}</h1><span class="fl-tag">{tagline}</span></div>'
    if strip:
        html += strip
        if strip_note:
            html += f'<div class="fl-strip-note">{strip_note}</div>'
    else:
        html += '<div style="height:14px"></div>'
    st.markdown(html, unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="fl-foot">{APP_NAME} · SMS fiber-sensor fringe analysis · '
                f'<a href="{REPO_URL}" target="_blank">source &amp; handbook on GitHub</a> · '
                'figures: hover for values, drag to zoom, camera icon to save as PNG</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# file-format guide
# --------------------------------------------------------------------------
FORMAT_RULES = """
**Base layout** — one row per spectrum, in acquisition order:

| column | header | content |
|---|---|---|
| 1 | `Temperature (°C)` — or any condition: `Strain`, `Load (g)`, … | value of the condition when the spectrum was taken |
| 2 | `Time (s)` | seconds since the start of the run |
| 3 … | the **wavelength in nm** (`1475.0`, `1475.1`, …) | transmittance (or power) at that wavelength |

Headers in rows 3… must be plain numbers between 200 and 5000 (nm). The first
two columns are found by their header names (`temp`, `°C`, `time`, `sec`, …);
if they have no names, the first two non-wavelength columns are taken as
condition and time, in that order.

**Also accepted**

- `;` or tab separators and decimal commas (`0,769`).
- An index column (`Unnamed: 0`) or other extra columns — ignored.
- No time column — the spectrum number is used instead.
- No condition column — maps, peaks and tracking work; calibration is skipped.
- Transposed files: wavelengths down the first column, one spectrum per further
  column (headers may be the times). This layout carries no condition values.

Values are read as plain numbers; keep units out of the cells (`0.769`, not `0.769 dB`).
"""


def template_csv(example: SpectraSet | None) -> str:
    """A small CSV in the base layout, taken from the example run when present."""
    if example is not None and len(example) >= 3 and example.has_temperature:
        wl = example.wavelengths
        cols = np.arange(0, len(wl), max(1, len(wl) // 10))[:11]
        rows = np.linspace(0, len(example) - 1, 5).astype(int)
        df = pd.DataFrame(example.spectra[np.ix_(rows, cols)].round(6), columns=[f"{wl[c]:g}" for c in cols])
        df.insert(0, "Time (s)", example.time[rows].round(1))
        df.insert(0, "Temperature (°C)", example.temperature[rows].round(2))
    else:
        wl = np.arange(1500.0, 1500.6, 0.1)
        df = pd.DataFrame(np.full((3, len(wl)), 0.5), columns=[f"{w:g}" for w in wl])
        df.insert(0, "Time (s)", [0.0, 23.5, 47.0])
        df.insert(0, "Temperature (°C)", [25.0, 25.5, 26.0])
    return df.to_csv(index=False)


def format_guide(example: SpectraSet | None, key: str) -> None:
    """Render the expected-file-format guide with a downloadable template."""
    st.markdown(FORMAT_RULES)
    csv = template_csv(example)
    st.caption("First rows of a file in the base layout:")
    st.dataframe(pd.read_csv(io.StringIO(csv)), hide_index=True, width="stretch")
    st.download_button("Download this template (CSV)", csv, "fringelab_template.csv", "text/csv",
                       key=f"tpl_{key}")


def stat_strip(items: list[tuple[str, str, str]]) -> None:
    """Key numbers in a wrapping row of tiles (label, value, note).

    Plain HTML with inherited colours so it follows the light/dark theme and
    wraps on narrow screens instead of truncating like st.metric does.
    """
    figs = "".join(
        f'<div class="fl-fig"><div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="n">{note}&nbsp;</div></div>'
        for label, value, note in items)
    st.markdown(f'<div class="fl-figs">{figs}</div>', unsafe_allow_html=True)


def read_report(data: SpectraSet, cond_name: str, cond_unit: str) -> pd.DataFrame:
    """How the loader interpreted the file, as a two-column table."""
    wl = data.wavelengths
    step = np.median(np.diff(wl)) if len(wl) > 1 else np.nan
    dt = np.median(np.diff(data.time)) if len(data) > 1 else np.nan
    rows = [
        ("Layout", data.layout + (" (rows = wavelengths, columns = spectra)" if data.layout == "transposed" else "")),
        ("Spectra (rows)", f"{len(data)}"),
        ("Wavelength columns", f"{len(wl)} · {wl.min():.2f}–{wl.max():.2f} nm · step {step:.3f} nm"),
        ("Time", f"{data.time.min():.1f}–{data.time.max():.1f} s · median step {dt:.1f} s"),
        (cond_name, (f"{np.nanmin(data.temperature):.2f}–{np.nanmax(data.temperature):.2f} {cond_unit}"
                     if data.has_temperature else "not found — calibration skipped")),
        ("Transmittance", f"{np.nanmin(data.spectra):.3f}–{np.nanmax(data.spectra):.3f} · "
                          f"{int(np.isnan(data.spectra).sum())} empty cells"),
    ]
    return pd.DataFrame(rows, columns=["item", "as read"])


# --------------------------------------------------------------------------
# cached computation
# --------------------------------------------------------------------------
def _col(text: str):
    """Sidebar text -> loader column override (blank = auto, digits = index)."""
    text = text.strip()
    if not text:
        return None
    return int(text) if text.isdigit() else text


@st.cache_data(show_spinner="Reading the file…")
def load_data(file_bytes: bytes | None, suffix: str, cond_col, time_col,
              example_stamp: str = "") -> SpectraSet:
    # example_stamp only keys the cache: a replaced data/data.csv is re-read.
    if file_bytes is None:
        return load_sms_csv(EXAMPLE_CSV, temp_col=cond_col, time_col=time_col)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(file_bytes)
        return load_sms_csv(path, temp_col=cond_col, time_col=time_col)


@st.cache_data(show_spinner="Tracking every fringe…")
def track_all(_data: SpectraSet, data_key: str, prominence: float, distance: int,
              search_window: float, smooth_window: int, max_step: float | None):
    # _data is not hashed by Streamlit; data_key identifies it instead.
    return track_all_extrema(_data, prominence=prominence, distance=distance,
                             search_window=search_window, smooth_window=smooth_window,
                             max_step=max_step)


def fringe_table(tracks: dict[str, pd.DataFrame], has_cond: bool) -> pd.DataFrame:
    """One row per tracked fringe: where it started/ended, lock, sensitivity."""
    rows = []
    for label, tr in tracks.items():
        kind, w0 = label.split("@")
        ok = tr.dropna(subset=["wavelength"])
        row = dict(fringe=label, kind=kind, start_nm=float(w0),
                   end_nm=ok["wavelength"].iloc[-1] if len(ok) else np.nan,
                   shift_nm=(ok["wavelength"].iloc[-1] - ok["wavelength"].iloc[0]) if len(ok) else np.nan,
                   locked_pct=100.0 * len(ok) / max(len(tr), 1),
                   sensitivity_pm=np.nan, stderr_pm=np.nan, r_squared=np.nan,
                   rmse_linear_pm=np.nan, rmse_quadratic_pm=np.nan,
                   sens_at_min_pm=np.nan, sens_at_max_pm=np.nan)
        if has_cond:
            try:
                fit = fit_sensitivity(tr)
                curv = curvature_test(tr)
                s_lo, s_hi = curv.sensitivity_at_ends_pm
                row.update(sensitivity_pm=fit.slope * 1e3, stderr_pm=fit.stderr * 1e3,
                           r_squared=fit.r_squared,
                           rmse_linear_pm=curv.linear.rmse * 1e3,
                           rmse_quadratic_pm=curv.quadratic.rmse * 1e3,
                           sens_at_min_pm=s_lo, sens_at_max_pm=s_hi)
            except ValueError:
                pass
        rows.append(row)
    cols = ["fringe", "kind", "start_nm", "end_nm", "shift_nm", "locked_pct",
            "sensitivity_pm", "stderr_pm", "r_squared",
            "rmse_linear_pm", "rmse_quadratic_pm", "sens_at_min_pm", "sens_at_max_pm"]
    return pd.DataFrame(rows, columns=cols).sort_values("start_nm").reset_index(drop=True)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _stride(n: int, limit: int) -> int:
    return max(1, int(np.ceil(n / limit)))


def _y_axis(data: SpectraSet, y_mode: str, cond_title: str):
    """Per-spectrum y value and its axis title for the map-like figures."""
    if y_mode == "Time":
        return data.time / 60.0, "Time (min)"
    if y_mode == "Spectrum number":
        return np.arange(len(data), dtype=float), "Spectrum number"
    return data.temperature, cond_title


def map_figure(data, z, y_mode, cond_title, colorscale, zmid, clip_pct, z_title, overlay):
    y, y_title = _y_axis(data, y_mode, cond_title)
    if y_mode == "Condition":
        # Several spectra can share one condition value (e.g. a stepped
        # experiment): show their mean so the y axis stays single-valued.
        good = np.isfinite(y)
        uniq, inv = np.unique(y[good], return_inverse=True)
        counts = np.bincount(inv)
        z_plot = np.array([np.bincount(inv, weights=col) for col in z[good].T]).T / counts[:, None]
        y_plot = uniq
    else:
        z_plot, y_plot = z, y

    sr, sc = _stride(z_plot.shape[0], MAX_MAP_ROWS), _stride(z_plot.shape[1], MAX_MAP_COLS)
    z_plot, y_plot, x_plot = z_plot[::sr, ::sc], y_plot[::sr], data.wavelengths[::sc]

    lo, hi = np.nanpercentile(z_plot, [clip_pct, 100 - clip_pct])
    if zmid is not None:  # symmetric range for the difference view
        lim = max(abs(lo), abs(hi))
        lo, hi = -lim, lim
    fig = go.Figure(go.Heatmap(
        x=x_plot, y=y_plot, z=z_plot, colorscale=colorscale, zmin=lo, zmax=hi,
        colorbar=dict(title=z_title),
        hovertemplate="λ %{x:.2f} nm<br>" + y_title + " %{y:.2f}<br>" + z_title + " %{z:.4f}<extra></extra>",
    ))
    for label, tr in overlay.items():
        fig.add_trace(go.Scatter(
            x=tr["wavelength"], y=y, mode="lines", name=label, showlegend=False,
            line=dict(color="white", width=1.2), opacity=0.85, connectgaps=False,
            hovertemplate=label + "<br>λ %{x:.3f} nm<extra></extra>",
        ))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Wavelength (nm)", yaxis_title=y_title)
    fig.update_xaxes(range=[data.wavelengths.min(), data.wavelengths.max()])
    return fig


def spectrum_figure(data, row, ext, overlay_rows, cond_name, cond_unit):
    fig = go.Figure()
    for r in overlay_rows:
        name = f"#{r}"
        if data.has_temperature:
            name += f" · {data.temperature[r]:.1f} {cond_unit}"
        fig.add_trace(go.Scatter(x=data.wavelengths, y=data.spectra[r], mode="lines",
                                 name=name, line=dict(width=1), opacity=0.55))
    fig.add_trace(go.Scatter(x=data.wavelengths, y=data.spectra[row], mode="lines",
                             name=f"spectrum #{row}", line=dict(color="#444444", width=1.6)))
    fig.add_trace(go.Scatter(x=ext.peak_wavelengths, y=ext.peak_values, mode="markers",
                             name="peaks (maxima)",
                             marker=dict(symbol="circle", size=9, color=PEAK_COLOR,
                                         line=dict(color="black", width=0.6))))
    fig.add_trace(go.Scatter(x=ext.dip_wavelengths, y=ext.dip_values, mode="markers",
                             name="dips (minima)",
                             marker=dict(symbol="triangle-down", size=9, color=DIP_COLOR,
                                         line=dict(color="black", width=0.6))))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Wavelength (nm)", yaxis_title="Transmittance",
                      hovermode="x unified")
    return fig


def shifts_figure(tracks, x_mode, cond_title):
    """Wavelength shift of every fringe relative to where it started."""
    fig = go.Figure()
    for label, tr in tracks.items():
        ok = tr.dropna(subset=["wavelength"])
        if ok.empty:
            continue
        x = ok["time"] if x_mode == "Time" else ok["temperature"]
        fig.add_trace(go.Scatter(x=x, y=ok["wavelength"] - ok["wavelength"].iloc[0],
                                 mode="lines", name=label, line=dict(width=1.3)))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Time (s)" if x_mode == "Time" else cond_title,
                      yaxis_title="Wavelength shift from start (nm)")
    return fig


def igor_figure(track, has_cond, cond_title):
    """Tracked wavelength (red, left) and condition (blue, right) vs time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=track["time"], y=track["wavelength"], mode="lines",
                             name="wavelength", line=dict(color=RED, width=1.6),
                             connectgaps=False), secondary_y=False)
    if has_cond:
        fig.add_trace(go.Scatter(x=track["time"], y=track["temperature"], mode="lines",
                                 name=cond_title, line=dict(color=BLUE, width=1.6)),
                      secondary_y=True)
        fig.update_yaxes(title_text=cond_title, color=BLUE, showgrid=False, secondary_y=True)
    fig.update_yaxes(title_text="Peak Wavelength (nm)", color=RED, secondary_y=False)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Time, seconds", showlegend=False)
    return fig


def calibration_figure(track, curv, model, cond_title, unit):
    """Tracked wavelength vs condition with the chosen fit(s) and their residuals."""
    lin, quad = curv.linear, curv.quadratic
    ok = quad.residuals(track)
    t = np.linspace(quad.t_min, quad.t_max, 100)
    s_lo, s_hi = curv.sensitivity_at_ends_pm
    models = []
    if model in ("Linear", "Both"):
        models.append((lin, LINEAR_COLOR, "dash",
                       f"linear {lin.coefficients[0] * 1e3:+.1f} pm/{unit} · RMSE {lin.rmse * 1e3:.0f} pm"))
    if model in ("Quadratic", "Both"):
        models.append((quad, QUAD_COLOR, "solid",
                       f"quadratic {s_lo:+.0f} → {s_hi:+.0f} pm/{unit} · RMSE {quad.rmse * 1e3:.0f} pm"))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=ok["temperature"], y=ok["wavelength"], mode="markers",
                             name="tracked", marker=dict(size=4, color=PEAK_COLOR, opacity=0.5),
                             hovertemplate="%{x:.2f} → %{y:.3f} nm<extra></extra>"), row=1, col=1)
    for fit, color, dash, name in models:
        fig.add_trace(go.Scatter(x=t, y=fit.predict(t), mode="lines", name=name,
                                 line=dict(color=color, width=2, dash=dash),
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ok["temperature"], y=(ok["wavelength"] - fit.predict(ok["temperature"])) * 1e3,
                                 mode="markers", showlegend=False, name=name.split(" ")[0],
                                 marker=dict(size=3.5, color=color, opacity=0.6),
                                 hovertemplate="%{x:.2f} → %{y:+.0f} pm<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0, line_color="#999999", line_width=1, row=2, col=1)
    fig.update_yaxes(title_text="Wavelength (nm)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (pm)", row=2, col=1)
    fig.update_xaxes(title_text=cond_title, row=2, col=1)
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.16, x=0))
    return fig


def sensitivity_figure(table, unit):
    fig = go.Figure()
    for kind, symbol, color in (("peak", "circle", PEAK_COLOR), ("dip", "triangle-down", DIP_COLOR)):
        sel = table[table["kind"] == kind]
        fig.add_trace(go.Scatter(
            x=sel["start_nm"], y=sel["sensitivity_pm"], mode="markers", name=f"{kind}s",
            text=sel["fringe"], error_y=dict(type="data", array=sel["stderr_pm"], color="#999999"),
            marker=dict(symbol=symbol, size=10, color=color, line=dict(color="black", width=0.6)),
            hovertemplate="%{text}<br>%{y:+.1f} pm/" + unit + "<extra></extra>"))
    fig.add_hline(y=0, line_color="#999999", line_width=1)
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Fringe wavelength at start (nm)",
                      yaxis_title=f"Sensitivity (pm/{unit})")
    return fig


def surface_figure(data, y_mode, cond_title, colorscale):
    y, y_title = _y_axis(data, y_mode, cond_title)
    sr, sc = _stride(len(data), MAX_3D_ROWS), _stride(len(data.wavelengths), MAX_3D_COLS)
    fig = go.Figure(go.Surface(x=data.wavelengths[::sc], y=y[::sr], z=data.spectra[::sr, ::sc],
                               colorscale=colorscale, colorbar=dict(title="Transmittance")))
    fig.update_layout(height=680, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title="Wavelength (nm)", yaxis_title=y_title,
                                 zaxis_title="Transmittance",
                                 aspectratio=dict(x=1.6, y=1.2, z=0.6)))
    return fig


# --------------------------------------------------------------------------
# sidebar: data + settings
# --------------------------------------------------------------------------
inject_style()
brand_slot = st.sidebar.empty()
brand_slot.markdown(f'<div class="fl-brand">{brand_icon()}{APP_NAME}<small>Explore fiber-sensor spectra: map, '
                    'peaks, fringe tracking, calibration.</small></div>', unsafe_allow_html=True)

st.sidebar.header("1 · Data")
sources = ["Upload a CSV file"]
if EXAMPLE_CSV.exists():
    sources.insert(0, "Example run (data/data.csv)")
source = st.sidebar.radio("Source", sources, label_visibility="collapsed")

file_bytes, suffix, file_name = None, ".csv", EXAMPLE_CSV.name
example_stamp = f"{EXAMPLE_CSV}:{EXAMPLE_CSV.stat().st_mtime_ns}" if EXAMPLE_CSV.exists() else ""
if source == "Upload a CSV file":
    upload = st.sidebar.file_uploader("Spectra file", type=["csv", "txt", "tsv", "dat"],
                                      help="CSV / TXT / TSV / DAT, up to Streamlit's upload limit (200 MB).")
    if upload is None:
        example = None
        if EXAMPLE_CSV.exists():
            try:
                example = load_data(None, ".csv", None, None, example_stamp)
            except Exception:  # noqa: BLE001 - the guide works without the example
                example = None
        hero("fringe analysis for SMS fiber sensors",
             fringe_strip(example.wavelengths, example.spectra[0]) if example is not None else "",
             "The example run's first spectrum: every bright or dark band is one interference fringe." if example is not None else "")
        intro, what = st.columns([1.1, 1])
        intro.markdown("**Upload a spectra file in the sidebar to begin.**\n\n"
                       "A run is a CSV of transmission spectra recorded while a condition — "
                       "temperature, strain, load — changes. The file layout below is what "
                       f"{APP_NAME} expects; a template is available to download.")
        what.markdown("**What you get**\n"
                      "- the whole run as a wavelength × time map\n"
                      "- every peak and dip of every spectrum\n"
                      "- each interference fringe tracked through the run\n"
                      "- linear and curved calibration, sensitivity per fringe\n"
                      "- CSV export of all of it")
        format_guide(example, key="welcome")
        st.stop()
    file_bytes, suffix, file_name = upload.getvalue(), Path(upload.name).suffix or ".csv", upload.name

with st.sidebar.expander("File format guide"):
    st.markdown("One row per spectrum: condition, time, then one column per wavelength "
                "(header = nm). Full rules and a template are on the start page and under "
                "*How the file was read* above the tabs.")

with st.sidebar.expander("Condition and columns"):
    cond_name = st.text_input("Condition name", "Temperature",
                              help="What was varied during the run, e.g. Temperature, Strain, Load.")
    cond_unit = st.text_input("Condition unit", "°C")
    cond_col_txt = st.text_input("Condition column", "", placeholder="auto",
                                 help="Header name or 0-based column number. Blank = detect automatically.")
    time_col_txt = st.text_input("Time column", "", placeholder="auto",
                                 help="Header name or 0-based column number. Blank = detect automatically.")
cond_title = f"{cond_name} ({cond_unit})" if cond_unit else cond_name

try:
    full = load_data(file_bytes, suffix, _col(cond_col_txt), _col(time_col_txt), example_stamp)
except Exception as exc:  # noqa: BLE001 - show any loader problem to the user
    hero("fringe analysis for SMS fiber sensors")
    st.error(f"Could not read **{file_name}**: {exc}")
    st.markdown("Check the file against the expected layout:")
    format_guide(None, key="error")
    st.stop()

wl_lo, wl_hi = float(full.wavelengths.min()), float(full.wavelengths.max())
wl_range = st.sidebar.slider("Wavelength range (nm)", wl_lo, wl_hi, (wl_lo, wl_hi))
keep = (full.wavelengths >= wl_range[0]) & (full.wavelengths <= wl_range[1])
if keep.sum() < 20:
    st.error("The selected wavelength range is too narrow.")
    st.stop()
data = replace(full, wavelengths=full.wavelengths[keep], spectra=full.spectra[:, keep])
has_cond = data.has_temperature

with st.sidebar.expander("2 · Peak detection", expanded=False):
    prominence = st.number_input(
        "Prominence (transmittance)", 0.0005, 1.0, 0.02, 0.005, format="%.4f",
        help="How much a peak/dip must stand out from its surroundings. Lower = more, weaker extrema.")
    distance = st.slider("Minimum separation (samples)", 1, 100, 5)
    smooth_window = st.slider("Smoothing window (samples)", 5, 51, 11, step=2,
                              help="Savitzky–Golay window applied before detection.")

with st.sidebar.expander("3 · Fringe tracking", expanded=False):
    search_window = st.slider(
        "Search half-window (nm)", 0.1, 10.0, 1.5, 0.1,
        help="How far from its last position a fringe is searched. Keep below the fringe spacing.")
    reject = st.checkbox("Reject jumps (fringe-hop guard)", True)
    max_step = st.slider("Largest allowed jump per spectrum (nm)", 0.05, 5.0, 0.5, 0.05,
                         disabled=not reject) if reject else None
    min_lock = st.slider("Count a fringe as tracked when locked ≥ (%)", 10, 100, 90, 5)
st.sidebar.caption("Defaults are the values used for the published analysis; open a section to change them.")

data_key = hashlib.md5(
    (file_bytes or example_stamp.encode()) + repr((cond_col_txt, time_col_txt, wl_range)).encode()).hexdigest()
tracks = track_all(data, data_key, prominence, distance, search_window, smooth_window, max_step)
table = fringe_table(tracks, has_cond)
tracked = table[table["locked_pct"] >= min_lock]

# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
strip = fringe_strip(data.wavelengths, data.spectra[0])
hero(f"{file_name} — {len(data)} spectra, {len(data.wavelengths)} wavelengths, layout {data.layout}", strip,
     f"First spectrum, {data.wavelengths.min():.0f}–{data.wavelengths.max():.0f} nm, as a band: "
     "bright and dark stripes are the fringes the tracker follows.")
brand_slot.markdown(f'<div class="fl-brand">{brand_icon()}{APP_NAME}{strip}<small>Explore fiber-sensor spectra: map, '
                    'peaks, fringe tracking, calibration.</small></div>', unsafe_allow_html=True)
stat_strip([
    ("Spectra", f"{len(data)}", f"one every {np.median(np.diff(data.time)):.0f} s" if len(data) > 1 else ""),
    ("Wavelength (nm)", f"{data.wavelengths.min():.0f}–{data.wavelengths.max():.0f}",
     f"{len(data.wavelengths)} points · {np.median(np.diff(data.wavelengths)):.3g} nm step"),
    ("Duration", f"{(data.time.max() - data.time.min()) / 60:.0f} min", ""),
    (cond_name, (f"{data.temperature[0]:.1f} → {data.temperature[-1]:.1f} {cond_unit}" if has_cond else "not in file"),
     f"{np.nanmin(data.temperature):.1f}–{np.nanmax(data.temperature):.1f} {cond_unit}" if has_cond else ""),
    ("Fringes tracked", f"{len(tracked)} / {len(table)}", f"locked ≥ {min_lock} % of spectra"),
])
with st.expander("How the file was read · file format guide"):
    st.dataframe(read_report(full, cond_name, cond_unit), hide_index=True, width="stretch")
    st.markdown("If a column was taken for the wrong thing, name it under *Condition and columns* "
                "in the sidebar (header name or 0-based column number).")
    format_guide(full if full.has_temperature else None, key="loaded")
if not has_cond:
    st.warning(f"No {cond_name.lower()} column was found, so calibration (sensitivity) is skipped. "
               "Maps, peaks and tracking still work. If the file has one, name it under "
               "*Condition and columns* in the sidebar.")

y_modes = ["Time", "Spectrum number"] + (["Condition"] if has_cond else [])
y_label = {"Time": "Time", "Spectrum number": "Spectrum number", "Condition": cond_name}.get

tab_map, tab_spec, tab_track, tab_sens, tab_3d, tab_export = st.tabs(
    [":material/gradient: Spectral map", ":material/show_chart: Spectrum & peaks",
     ":material/timeline: Fringe tracking", ":material/thermostat: Sensitivity",
     ":material/view_in_ar: 3-D view", ":material/download: Export"])

# ---- spectral map ---------------------------------------------------------
with tab_map:
    c = st.columns([1.2, 1.6, 1, 1, 1.2])
    y_mode = c[0].selectbox("Vertical axis", y_modes, format_func=y_label, key="map_y")
    z_mode = c[1].selectbox("Show", ["Transmittance", "Change from first spectrum",
                                     "Each spectrum scaled 0–1"])
    cmap = c[2].selectbox("Colours", ["Viridis", "Plasma", "Inferno", "Cividis", "Turbo", "Greys"])
    clip = c[3].slider("Contrast clip (%)", 0.0, 10.0, 0.0, 0.5,
                       help="Ignore this share of the darkest and brightest values when scaling colours.")
    show_tracks = c[4].checkbox("Overlay tracked fringes", True)

    zmid = None
    if z_mode == "Change from first spectrum":
        z, cmap, zmid, z_title = data.spectra - data.spectra[0], "RdBu_r", 0.0, "Δ transmittance"
    elif z_mode == "Each spectrum scaled 0–1":
        lo = np.nanmin(data.spectra, axis=1, keepdims=True)
        span = np.nanmax(data.spectra, axis=1, keepdims=True) - lo
        z, z_title = (data.spectra - lo) / np.where(span == 0, 1, span), "Scaled transmittance"
    else:
        z, z_title = data.spectra, "Transmittance"

    overlay = {k: tracks[k] for k in tracked["fringe"]} if show_tracks else {}
    st.plotly_chart(map_figure(data, z, y_mode, cond_title, cmap, zmid, clip, z_title, overlay),
                    width="stretch")
    st.caption("Every row is one spectrum; bright and dark bands are interference fringes. "
               "White lines are the fringes followed by the tracker. Drag to zoom, double-click to reset.")

# ---- single spectrum ------------------------------------------------------
with tab_spec:
    c = st.columns([3, 1])
    row = c[0].slider("Spectrum number", 0, len(data) - 1, 0) if len(data) > 1 else 0
    n_overlay = c[1].slider("Also overlay evenly spaced spectra", 0, 8, 0)
    ext = detect_extrema(data.wavelengths, data.spectra[row], prominence=prominence,
                         distance=distance, smooth_window=smooth_window)
    overlay_rows = ([int(r) for r in np.linspace(0, len(data) - 1, n_overlay)] if n_overlay else [])
    st.plotly_chart(spectrum_figure(data, row, ext, overlay_rows, cond_name, cond_unit),
                    width="stretch")
    info = f"t = {data.time[row]:.0f} s"
    if has_cond:
        info += f" · {cond_name} = {data.temperature[row]:.2f} {cond_unit}"
    st.caption(f"{info} · {len(ext.peak_wavelengths)} peaks and {len(ext.dip_wavelengths)} dips "
               "with the current detection settings.")
    with st.expander("Detected extrema of this spectrum"):
        st.dataframe(pd.DataFrame({
            "kind": ["peak"] * len(ext.peak_wavelengths) + ["dip"] * len(ext.dip_wavelengths),
            "wavelength_nm": np.r_[ext.peak_wavelengths, ext.dip_wavelengths],
            "transmittance": np.r_[ext.peak_values, ext.dip_values],
        }).sort_values("wavelength_nm"), hide_index=True, width="stretch")

# ---- tracking -------------------------------------------------------------
with tab_track:
    if table.empty:
        st.info("No peaks or dips were found in the first spectrum. Lower the prominence in the sidebar.")
    else:
        x_mode = st.radio("Horizontal axis", ["Time"] + (["Condition"] if has_cond else []),
                          format_func=y_label, horizontal=True, key="shift_x")
        st.plotly_chart(shifts_figure({k: tracks[k] for k in tracked["fringe"]}, x_mode, cond_title),
                        width="stretch")
        st.caption("Each line is one fringe from the first spectrum, shown as its shift from where it started.")

        st.subheader("One fringe in detail")
        ranked = tracked.dropna(subset=["sensitivity_pm"])
        ranked = ranked[ranked["r_squared"] >= 0.9]
        default = (ranked.loc[ranked["sensitivity_pm"].abs().idxmax(), "fringe"] if len(ranked)
                   else (tracked["fringe"].iloc[0] if len(tracked) else table["fringe"].iloc[0]))
        labels = list(table["fringe"])
        label = st.selectbox(
            "Fringe (type @ wavelength in the first spectrum)", labels, index=labels.index(default),
            format_func=lambda k: f"{k} nm · locked {table.set_index('fringe').loc[k, 'locked_pct']:.0f}%")
        track = tracks[label]
        left, right = st.columns(2)
        left.plotly_chart(igor_figure(track, has_cond, cond_title), width="stretch")
        left.caption("Igor-style view: tracked wavelength (red, left axis) and the condition "
                     "(blue, right axis) against time.")
        if has_cond:
            try:
                fit = fit_sensitivity(track)
                curv = curvature_test(track)
                model = right.radio("Calibration model", ["Both", "Linear", "Quadratic"],
                                    horizontal=True, key="cal_model",
                                    help="Linear: one sensitivity for the whole run. Quadratic: the "
                                         "sensitivity changes along the run, as it does close to the "
                                         "critical wavelength. The lower panel shows measured − fitted.")
                right.plotly_chart(calibration_figure(track, curv, model, cond_title, cond_unit),
                                   width="stretch")
                s_lo, s_hi = curv.sensitivity_at_ends_pm
                q = curv.quadratic
                right.caption(
                    f"Linear: {fit.slope * 1e3:+.1f} ± {fit.stderr * 1e3:.1f} pm/{cond_unit}, "
                    f"R² = {fit.r_squared:.4f}, RMSE {curv.linear.rmse * 1e3:.0f} pm, {fit.n_points} points. "
                    f"Quadratic: RMSE {q.rmse * 1e3:.0f} pm "
                    f"({curv.rmse_improvement:+.0%} vs linear); local sensitivity "
                    f"{s_lo:+.0f} pm/{cond_unit} at {q.t_min:.1f} {cond_unit} and "
                    f"{s_hi:+.0f} pm/{cond_unit} at {q.t_max:.1f} {cond_unit}; "
                    f"curvature {curv.curvature_pm_per_C2:+.2f} pm/{cond_unit}².")
            except ValueError:
                right.info("Too few locked points to fit this fringe.")
        else:
            right.info(f"Calibration needs a {cond_name.lower()} column.")

# ---- sensitivity ----------------------------------------------------------
with tab_sens:
    min_r2 = st.slider("Minimum R² to include", 0.0, 1.0, 0.9, 0.01)
    shown = table[table["r_squared"] >= min_r2] if has_cond else table
    if has_cond and len(shown):
        st.plotly_chart(sensitivity_figure(shown, cond_unit), width="stretch")
    elif has_cond:
        st.info("No fringe passes the R² threshold.")
    else:
        st.info(f"Sensitivity needs a {cond_name.lower()} column; the table lists the tracked fringes only.")
    if has_cond:
        t_lo, t_hi = np.nanmin(data.temperature), np.nanmax(data.temperature)
        st.caption(f"Linear columns: one sensitivity for the whole run. Curved columns: a quadratic "
                   f"calibration, its RMSE against the linear one, and the local sensitivity at "
                   f"{t_lo:.1f} and {t_hi:.1f} {cond_unit}. A clearly lower quadratic RMSE means the "
                   f"sensitivity changed along the run (fringes close to the critical wavelength).")
    st.dataframe(
        shown.rename(columns={"sensitivity_pm": f"sensitivity (pm/{cond_unit})",
                              "stderr_pm": f"± (pm/{cond_unit})", "locked_pct": "locked (%)",
                              "start_nm": "start (nm)", "end_nm": "end (nm)", "shift_nm": "shift (nm)",
                              "r_squared": "R²", "rmse_linear_pm": "RMSE linear (pm)",
                              "rmse_quadratic_pm": "RMSE quadratic (pm)",
                              "sens_at_min_pm": f"S at min {cond_unit} (pm)",
                              "sens_at_max_pm": f"S at max {cond_unit} (pm)"}),
        hide_index=True, width="stretch",
        column_config={"start (nm)": st.column_config.NumberColumn(format="%.2f"),
                       "end (nm)": st.column_config.NumberColumn(format="%.2f"),
                       "shift (nm)": st.column_config.NumberColumn(format="%+.3f"),
                       "locked (%)": st.column_config.NumberColumn(format="%.0f"),
                       f"sensitivity (pm/{cond_unit})": st.column_config.NumberColumn(format="%+.1f"),
                       f"± (pm/{cond_unit})": st.column_config.NumberColumn(format="%.1f"),
                       "R²": st.column_config.NumberColumn(format="%.4f"),
                       "RMSE linear (pm)": st.column_config.NumberColumn(format="%.0f"),
                       "RMSE quadratic (pm)": st.column_config.NumberColumn(format="%.0f"),
                       f"S at min {cond_unit} (pm)": st.column_config.NumberColumn(format="%+.0f"),
                       f"S at max {cond_unit} (pm)": st.column_config.NumberColumn(format="%+.0f")})

# ---- 3-D ------------------------------------------------------------------
with tab_3d:
    c = st.columns([1, 1, 3])
    y3 = c[0].selectbox("Depth axis", y_modes, index=len(y_modes) - 1, format_func=y_label, key="y3")
    cmap3 = c[1].selectbox("Colours", ["Viridis", "Plasma", "Inferno", "Cividis", "Turbo"], key="c3")
    if c[2].toggle("Draw the 3-D surface", False,
                   help="Off by default because the surface takes a moment to draw."):
        st.plotly_chart(surface_figure(data, y3, cond_title, cmap3), width="stretch")
        st.caption("Transmittance as height. Drag to rotate, scroll to zoom.")

# ---- export ---------------------------------------------------------------
with tab_export:
    st.write("Download the results for the current file and settings. "
             "Any figure can be saved as PNG with the camera icon that appears above it.")
    wide = pd.DataFrame({"time": data.time, "condition": data.temperature})
    for k, tr in tracks.items():
        wide[k] = tr["wavelength"].to_numpy()
    long = pd.concat([tr.assign(fringe=k) for k, tr in tracks.items()], ignore_index=True) \
        if tracks else pd.DataFrame()
    settings = dict(file=file_name, layout=data.layout, condition=cond_name, unit=cond_unit,
                    wavelength_range_nm=list(wl_range), prominence=prominence, distance=distance,
                    smooth_window=smooth_window, search_window_nm=search_window,
                    max_step_nm=max_step, min_lock_pct=min_lock)
    stem = Path(file_name).stem
    c = st.columns(4)
    c[0].download_button("Tracked fringes (wide CSV)", wide.to_csv(index=False),
                         f"{stem}_tracked_wide.csv", "text/csv")
    c[1].download_button("Tracked fringes (long CSV)", long.to_csv(index=False),
                         f"{stem}_tracked_long.csv", "text/csv")
    c[2].download_button("Fringe / sensitivity table", table.to_csv(index=False),
                         f"{stem}_fringes.csv", "text/csv")
    c[3].download_button("Settings (JSON)", json.dumps(settings, indent=2),
                         f"{stem}_settings.json", "application/json")

footer()
