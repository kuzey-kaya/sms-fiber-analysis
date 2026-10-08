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
import time as _time
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
from sms_analysis.movie import frame_rows, render_movie, render_phase_movie
from sms_analysis.phase import (PAPER_LAMBDA_A, PAPER_LAMBDA_B, phase_fits, phase_sensitivity,
                                phase_series, scan_pairs, tracked_phase_change)

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


@st.cache_data(show_spinner="Fitting the phase profile of every spectrum…")
def phase_fits_cached(_data: SpectraSet, data_key: str, prominence: float, distance: int,
                      smooth_window: int):
    return phase_fits(_data, prominence=prominence, distance=distance, smooth_window=smooth_window)


def phase_profile_figure(data, fit):
    """First spectrum with numbered extrema, and their phases with the cubic fit."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Extrema numbered outward from λc",
                                                        "Δφ(λ±n) = −(n−1)π and the cubic fit"))
    fig.add_trace(go.Scatter(x=data.wavelengths, y=data.spectra[0], mode="lines", name="first spectrum",
                             line=dict(color=PEAK_COLOR, width=1.4), hoverinfo="skip"), row=1, col=1)
    y0 = np.interp(fit.wavelengths, data.wavelengths, data.spectra[0])
    labels = fit.labels()
    near = [lab if (lab == "λc" or int(lab[2:]) <= 3) else "" for lab in labels]   # avoid crowding
    fig.add_trace(go.Scatter(
        x=fit.wavelengths, y=y0, mode="markers+text", name="extrema", text=near, customdata=labels,
        textposition=["top center" if k == "peak" else "bottom center" for k in fit.kinds],
        textfont=dict(size=10),
        marker=dict(symbol=["circle" if k == "peak" else "triangle-down" for k in fit.kinds], size=8,
                    color=DIP_COLOR, line=dict(color="black", width=0.5)),
        hovertemplate="%{customdata}: %{x:.2f} nm<extra></extra>"), row=1, col=1)
    ok = np.isfinite(fit.phases)
    xs = np.linspace(data.wavelengths.min(), data.wavelengths.max(), 300)
    fig.add_trace(go.Scatter(x=fit.wavelengths[ok], y=fit.phases[ok], mode="markers", name="assigned phase",
                             marker=dict(symbol="circle-open", size=8, color=PEAK_COLOR),
                             text=[l for l, k in zip(labels, ok) if k],
                             hovertemplate="%{text}: %{y:.2f} rad<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=fit.phase(xs), mode="lines", name=f"cubic, RMS {fit.rms:.3f} rad",
                             line=dict(color=QUAD_COLOR, width=2), hoverinfo="skip"), row=1, col=2)
    fig.add_vline(x=fit.lambda_c, line_dash="dash", line_color="#999999", row=1, col=2)
    fig.update_xaxes(title_text="Wavelength (nm)")
    fig.update_yaxes(title_text="Transmittance", row=1, col=1)
    fig.update_yaxes(title_text="Δφ (rad)", row=1, col=2)
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.2, x=0))
    return fig


def phase_readout_figure(series, sens, la, lb, has_cond, cond_title):
    """Readout vs time (Igor colours), vs the condition with its fit, and λc vs the condition."""
    # time on its own full-width row (it has two y-axes), the two condition plots below it
    if has_cond:
        fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2, "secondary_y": True}, None], [{}, {}]],
                            subplot_titles=("Readout and condition vs time", "Readout vs condition",
                                            "λc vs condition"),
                            vertical_spacing=0.2, horizontal_spacing=0.12)
    else:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]],
                            subplot_titles=("Readout vs time",))
    fig.add_trace(go.Scatter(x=series["time"], y=series["phase_diff"], mode="lines", name="readout",
                             line=dict(color=RED, width=1.4)), row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"|φ({lb:g}) − φ({la:g})| (rad)", color=RED, row=1, col=1, secondary_y=False)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    if has_cond:
        fig.add_trace(go.Scatter(x=series["time"], y=series["temperature"], mode="lines", name=cond_title,
                                 line=dict(color=BLUE, width=1.4)), row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text=cond_title, color=BLUE, showgrid=False, row=1, col=1, secondary_y=True)
        ok = series.dropna(subset=["phase_diff", "temperature"])
        t = np.linspace(ok["temperature"].min(), ok["temperature"].max(), 50)
        fig.add_trace(go.Scatter(x=ok["temperature"], y=ok["phase_diff"], mode="markers", name="frames",
                                 marker=dict(size=4, color=PEAK_COLOR, opacity=0.5)), row=2, col=1)
        if sens is not None:
            fig.add_trace(go.Scatter(x=t, y=sens.slope * t + sens.intercept, mode="lines",
                                     name=f"{sens.slope * 1e3:+.1f} mrad per unit, R² {sens.r_squared:.4f}",
                                     line=dict(color="#333333", width=2)), row=2, col=1)
        okc = ok.dropna(subset=["lambda_c"])
        k = np.polyfit(okc["temperature"], okc["lambda_c"], 1)
        fig.add_trace(go.Scatter(x=okc["temperature"], y=okc["lambda_c"], mode="markers", name="λc",
                                 marker=dict(size=4, color=DIP_COLOR, opacity=0.5)), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=np.polyval(k, t), mode="lines", name=f"λc {k[0] * 1e3:+.1f} pm per unit",
                                 line=dict(color="#333333", width=2, dash="dot")), row=2, col=2)
        fig.update_xaxes(title_text=cond_title, row=2, col=1)
        fig.update_xaxes(title_text=cond_title, row=2, col=2)
        fig.update_yaxes(title_text="Readout (rad)", row=2, col=1)
        fig.update_yaxes(title_text="λc (nm)", row=2, col=2)
    fig.update_layout(height=760 if has_cond else 420, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.12, x=0))
    return fig


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


MOVIE_MAX_FRAMES, MOVIE_MAX_COLS = 150, 500


@st.cache_data(show_spinner="Finding the extrema of every frame…")
def movie_extrema(_data: SpectraSet, data_key: str, rows: tuple, prominence: float, distance: int,
                  smooth_window: int):
    out = []
    for r in rows:
        ext = detect_extrema(_data.wavelengths, _data.spectra[r], prominence=prominence,
                             distance=distance, smooth_window=smooth_window)
        out.append((ext.peak_wavelengths, ext.peak_values, ext.dip_wavelengths, ext.dip_values))
    return out


@st.cache_data(show_spinner=False)
def map_image(_data: SpectraSet, max_rows: int = 400, max_cols: int = 600) -> str:
    """The spectral map as a viridis PNG data-URI (background of the movie's map panel)."""
    import base64 as _b64
    from io import BytesIO
    from matplotlib import colormaps
    from PIL import Image
    sr = max(1, int(np.ceil(len(_data) / max_rows)))
    sc = max(1, int(np.ceil(len(_data.wavelengths) / max_cols)))
    z = _data.spectra[::sr, ::sc][::-1]                # first spectrum at the bottom
    lo, hi = np.nanmin(z), np.nanmax(z)
    rgba = (colormaps["viridis"]((np.nan_to_num(z, nan=lo) - lo) / (hi - lo if hi > lo else 1.0)) * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode()


FRINGE_PALETTE = ["#2E5FA3", "#C4402F", "#4B9B78", "#E69F00", "#7B5AA6", "#1B9E9E", "#B5651D",
                  "#6C757D", "#D62E8A", "#8DA300", "#3A86FF", "#FF6B35"]


def _delta_at(wl, spec, spec0, x):
    """Value of (spectrum - first spectrum) at wavelengths x (for extrema in delta mode)."""
    return np.interp(x, wl, spec - spec0) if len(x) else np.array([])


def movie_figure(data, rows, extrema, track, track_label, cond_name, cond_unit, fps, *,
                 mode="Transmittance", follow_nm=None, trail=8, mark_fringe=True,
                 panel="One fringe", all_tracks=None, big_readout=True):
    """Plotly animation of the run.

    Trace order (frames update by index):
      0 first spectrum (static)      1 current spectrum     2 trail of past extrema
      3 peaks    4 dips    5 tracked-fringe marker    6 map cursor
      7/8 lower-right panel: fringe λ + condition, or current points + cursor of all fringes
      9… static faint shift curves (panel = "All fringes")
    """
    wl = data.wavelengths
    sc = max(1, int(np.ceil(len(wl) / MOVIE_MAX_COLS)))
    t_min = data.time / 60.0
    has_cond = data.has_temperature
    delta = mode == "Change from first spectrum"
    s0 = data.spectra[0]
    panel_title = {"One fringe": f"Tracked {track_label}", "All fringes": "All fringes: shift from start"}.get(panel, "")
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {"secondary_y": True}]],
                        column_widths=[0.58, 0.42], vertical_spacing=0.16, horizontal_spacing=0.1,
                        subplot_titles=("", "Spectral map", panel_title))

    # tracked wavelength per row, forward-filled for the camera and the marker
    if track is not None:
        trk_wl = track["wavelength"].ffill().bfill().to_numpy(float)
    else:
        trk_wl = np.full(len(data), np.nan)

    shifts = {}
    if panel == "All fringes" and all_tracks:
        for lab, tr in all_tracks.items():
            w = tr["wavelength"].to_numpy(float)
            first = w[np.isfinite(w)]
            shifts[lab] = w - (first[0] if len(first) else np.nan)

    def spec_y(r):
        y = data.spectra[r]
        return (y - s0) if delta else y

    def frame_traces(k):
        r = rows[k]
        pw, pv, dw, dv = extrema[k]
        if delta:
            pv, dv = _delta_at(wl, data.spectra[r], s0, pw), _delta_at(wl, data.spectra[r], s0, dw)
        # trail: extrema of the previous `trail` frames, fading out
        tx, ty, tsym, top = [], [], [], []
        for j in range(max(0, k - trail), k):
            rj = rows[j]
            qw, qv, ew, ev = extrema[j]
            if delta:
                qv, ev = _delta_at(wl, data.spectra[rj], s0, qw), _delta_at(wl, data.spectra[rj], s0, ew)
            op = 0.08 + 0.5 * (j - (k - trail) + 1) / (trail + 1)
            tx += list(qw) + list(ew); ty += list(qv) + list(ev)
            tsym += ["circle"] * len(qw) + ["triangle-down"] * len(ew); top += [op] * (len(qw) + len(ew))
        w_now = trk_wl[r] if mark_fringe and np.isfinite(trk_wl[r]) else None
        ylo, yhi = y_range
        traces = [
            go.Scatter(x=wl[::sc], y=spec_y(r)[::sc], mode="lines", line=dict(color="#444444", width=1.5),
                       name="current spectrum", hovertemplate="λ %{x:.2f} nm · %{y:.4f}<extra></extra>"),
            go.Scatter(x=tx, y=ty, mode="markers", name="previous frames", showlegend=False, hoverinfo="skip",
                       marker=dict(symbol=tsym, size=6, color="#8a8a8a", opacity=top)),
            go.Scatter(x=pw, y=pv, mode="markers", name="peaks",
                       marker=dict(symbol="circle", size=8, color=PEAK_COLOR, line=dict(color="black", width=0.5))),
            go.Scatter(x=dw, y=dv, mode="markers", name="dips",
                       marker=dict(symbol="triangle-down", size=8, color=DIP_COLOR, line=dict(color="black", width=0.5))),
            go.Scatter(x=[w_now, w_now] if w_now is not None else [], y=[ylo, yhi] if w_now is not None else [],
                       mode="lines", name="tracked fringe", line=dict(color=RED, width=1.2, dash="dot"),
                       hovertemplate="tracked fringe · %{x:.3f} nm<extra></extra>" if w_now is not None else None),
            go.Scatter(x=[wl.min(), wl.max()], y=[t_min[r], t_min[r]], mode="lines", name="now",
                       line=dict(color="white", width=2), showlegend=False, hoverinfo="skip"),
        ]
        if panel == "All fringes" and shifts:
            labs = list(shifts)
            traces += [
                go.Scatter(x=[data.time[r]] * len(labs), y=[shifts[l][r] for l in labs], mode="markers",
                           showlegend=False, text=labs, hovertemplate="%{text}<br>%{y:+.3f} nm<extra></extra>",
                           marker=dict(size=7, color=[FRINGE_PALETTE[i % len(FRINGE_PALETTE)] for i in range(len(labs))],
                                       line=dict(color="black", width=0.4))),
                go.Scatter(x=[data.time[r], data.time[r]], y=[shift_lo, shift_hi], mode="lines", showlegend=False,
                           line=dict(color="#888888", width=1), hoverinfo="skip"),
            ]
        else:
            seg = track.iloc[: r + 1] if (track is not None and panel == "One fringe") else None
            traces += [
                go.Scatter(x=seg["time"] if seg is not None else [], y=seg["wavelength"] if seg is not None else [],
                           mode="lines", line=dict(color=RED, width=1.6), name="wavelength", showlegend=False,
                           connectgaps=False),
                go.Scatter(x=seg["time"] if (seg is not None and has_cond) else [],
                           y=seg["temperature"] if (seg is not None and has_cond) else [],
                           mode="lines", line=dict(color=BLUE, width=1.6), name=cond_name, showlegend=False),
            ]
        return traces

    # axis ranges first (frame traces need them)
    if delta:
        d_all = data.spectra[::max(1, len(data) // 100)] - s0
        lim = float(np.nanpercentile(np.abs(d_all), 99.5)) or 0.1
        y_range = (-1.08 * lim, 1.30 * lim)
    else:
        lo, hi = np.nanmin(data.spectra), np.nanmax(data.spectra)
        y_range = (lo - 0.04 * (hi - lo), hi + 0.18 * (hi - lo))
    if shifts:
        allv = np.concatenate([v[np.isfinite(v)] for v in shifts.values()] or [np.array([0.0])])
        shift_lo, shift_hi = float(allv.min()) - 0.3, float(allv.max()) + 0.3
    else:
        shift_lo = shift_hi = 0.0

    base = frame_traces(0)
    if delta:
        fig.add_trace(go.Scatter(x=[wl.min(), wl.max()], y=[0, 0], mode="lines", name="first spectrum (= 0)",
                                 line=dict(color="#aaaaaa", width=1), hoverinfo="skip"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=wl[::sc], y=s0[::sc], mode="lines", name="first spectrum",
                                 line=dict(color="#aaaaaa", width=1), opacity=0.6, hoverinfo="skip"), row=1, col=1)
    for tr in base[:5]:
        fig.add_trace(tr, row=1, col=1)
    # The map is a static picture behind the cursor (a Heatmap trace is dropped by
    # Plotly's animation redraw and makes every frame heavy).
    fig.add_layout_image(source=map_image(data), xref="x2", yref="y2", x=wl.min(), y=t_min.max(),
                         sizex=wl.max() - wl.min(), sizey=t_min.max() - t_min.min(),
                         sizing="stretch", layer="below", xanchor="left", yanchor="top")
    fig.add_trace(base[5], row=1, col=2)
    fig.add_trace(base[6], row=2, col=2, secondary_y=False)
    fig.add_trace(base[7], row=2, col=2, secondary_y=(panel != "All fringes"))
    animated = [1, 2, 3, 4, 5, 6, 7, 8]
    if shifts:
        for i, (lab, sh) in enumerate(shifts.items()):
            fig.add_trace(go.Scatter(x=data.time, y=sh, mode="lines", name=lab, showlegend=False, hoverinfo="skip",
                                     line=dict(color=FRINGE_PALETTE[i % len(FRINGE_PALETTE)], width=1), opacity=0.35,
                                     connectgaps=False), row=2, col=2, secondary_y=False)

    def stamp(r):
        text = f"spectrum {r + 1}/{len(data)} · t = {data.time[r]:.0f} s"
        if has_cond:
            text += f" · {cond_name} = {data.temperature[r]:.2f} {cond_unit}"
        return text

    base_annotations = list(fig.layout.annotations)

    def frame_layout(r):
        lay = dict(title_text=stamp(r))
        if big_readout and has_cond:
            lay["annotations"] = base_annotations + [dict(
                xref="x domain", yref="y domain", x=0.99, y=0.99, xanchor="right", yanchor="top",
                text=f"<b>{data.temperature[r]:.2f} {cond_unit}</b>  <span style='font-size:11px'>"
                     f"{t_min[r]:.1f} min</span>",
                showarrow=False, font=dict(size=22, color=BLUE), align="right")]
        if follow_nm and np.isfinite(trk_wl[r]):
            lay["xaxis"] = dict(range=[trk_wl[r] - follow_nm, trk_wl[r] + follow_nm], title_text="Wavelength (nm)")
        return go.Layout(**lay)

    frames = [go.Frame(name=str(k), data=frame_traces(k), traces=animated, layout=frame_layout(rows[k]))
              for k in range(len(rows))]
    fig.frames = frames
    step = dict(duration=int(1000 / fps), redraw=True)
    label_every = max(1, len(rows) // 12)
    fig.update_layout(
        height=700, margin=dict(l=10, r=10, t=100, b=90),
        title=dict(text=stamp(rows[0]), x=0, xanchor="left", y=0.985, yanchor="top"),
        legend=dict(orientation="h", y=-0.32, yanchor="top", x=0, xanchor="left", font=dict(size=11)),
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=1.0, xanchor="left", yanchor="bottom",
                          direction="right", pad=dict(b=4),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=step, fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.0, y=-0.14, len=1.0, pad=dict(t=0), ticklen=4,
                      currentvalue=dict(visible=False),
                      steps=[dict(method="animate", label=(f"{t_min[r]:.0f}" if k % label_every == 0 else ""),
                                  value=str(k),
                                  args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                             for k, r in enumerate(rows)])],
    )
    if big_readout and has_cond:
        fig.update_layout(annotations=frame_layout(rows[0]).annotations)
    x0 = [trk_wl[rows[0]] - follow_nm, trk_wl[rows[0]] + follow_nm] if (follow_nm and np.isfinite(trk_wl[rows[0]])) \
        else [wl.min(), wl.max()]
    fig.update_xaxes(title_text="Wavelength (nm)", range=x0, row=1, col=1)
    fig.update_yaxes(title_text="Δ transmittance" if delta else "Transmittance", range=list(y_range), row=1, col=1)
    fig.update_xaxes(range=[wl.min(), wl.max()], row=1, col=2)
    fig.update_yaxes(title_text="Time (min)", range=[t_min.min(), t_min.max()], row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", range=[data.time.min(), data.time.max()], row=2, col=2)
    if panel == "All fringes":
        fig.update_yaxes(title_text="Shift from start (nm)", range=[shift_lo, shift_hi], row=2, col=2, secondary_y=False)
    elif track is not None:
        ok = track.dropna(subset=["wavelength"])
        pad = 0.05 * (ok["wavelength"].max() - ok["wavelength"].min() + 1e-9)
        fig.update_yaxes(title_text="λ (nm)", color=RED, range=[ok["wavelength"].min() - pad, ok["wavelength"].max() + pad],
                         row=2, col=2, secondary_y=False)
        if has_cond:
            fig.update_yaxes(title_text=f"{cond_name} ({cond_unit})", color=BLUE, showgrid=False,
                             range=[np.nanmin(data.temperature), np.nanmax(data.temperature)],
                             row=2, col=2, secondary_y=True)
    return fig


def phase_movie_figure(data, rows, fits, la, lb, fps, cond_name, cond_unit):
    """Plotly animation of the SM1E.2 readout.

    Traces (frames update 1, 2, 3, 5, 6, 7, 8, 9): 0 spectrum ghost (static) · 1 spectrum ·
    2 extrema markers · 3 λ±n labels · 4 first-frame cubic (static) · 5 assigned phases ·
    6 cubic · 7 λc line · 8 readout · 9 condition.  The labels are a separate text trace
    with a fixed number of slots: Plotly does not reliably animate a text array whose
    length changes between frames (the extrema count goes 24 → 25 in the example run).
    """
    n_slots = 7                                    # λc and λ±1…λ±3
    wl = data.wavelengths
    sc = max(1, int(np.ceil(len(wl) / MOVIE_MAX_COLS)))
    xs = np.linspace(wl.min(), wl.max(), 200)
    has_cond = data.has_temperature
    series = phase_series(data, la, lb, fits=fits)
    lo, hi = np.nanmin(data.spectra), np.nanmax(data.spectra)
    allp = np.concatenate([fits[r].phase(xs) for r in rows])
    p_lo, p_hi = float(allp.min() - 2), float(max(allp.max(), 0) + 3)
    fit0 = fits[rows[0]]

    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {"secondary_y": True}]],
                        column_widths=[0.54, 0.46], vertical_spacing=0.18, horizontal_spacing=0.15,
                        subplot_titles=("", "Phase from the numbered extrema", "Phase-difference readout"))

    def frame_traces(k):
        r = rows[k]
        f = fits[r]
        y = np.interp(f.wavelengths, wl, data.spectra[r])
        labs = f.labels()
        idx = [i for i, lab in enumerate(labs) if lab == "λc" or int(lab[2:]) <= 3][:n_slots]
        off = 0.045 * (hi - lo)
        lx = [float(f.wavelengths[i]) for i in idx] + [None] * (n_slots - len(idx))
        ly = [float(y[i] + (off if f.kinds[i] == "peak" else -off)) for i in idx] + [None] * (n_slots - len(idx))
        lt = [labs[i] for i in idx] + [""] * (n_slots - len(idx))
        okp = np.isfinite(f.phases)
        seg = series.iloc[: r + 1]
        return [
            go.Scatter(x=wl[::sc], y=data.spectra[r][::sc], mode="lines", name="spectrum",
                       line=dict(color=PEAK_COLOR, width=1.5), hoverinfo="skip"),
            go.Scatter(x=f.wavelengths, y=y, mode="markers", name="extrema", customdata=labs,
                       marker=dict(symbol=["circle" if k == "peak" else "triangle-down" for k in f.kinds],
                                   size=8, color=DIP_COLOR, line=dict(color="black", width=0.5)),
                       hovertemplate="%{customdata}: %{x:.2f} nm<extra></extra>"),
            go.Scatter(x=lx, y=ly, mode="text", text=lt, textposition="middle center", showlegend=False,
                       textfont=dict(size=11), hoverinfo="skip", cliponaxis=False),
            go.Scatter(x=f.wavelengths[okp], y=f.phases[okp], mode="markers", name="Δφ(λ±n) = −(n−1)π",
                       marker=dict(symbol="circle-open", size=7, color=PEAK_COLOR), hoverinfo="skip"),
            go.Scatter(x=xs, y=f.phase(xs), mode="lines", name="cubic Δφ(λ)",
                       line=dict(color=QUAD_COLOR, width=2), hoverinfo="skip"),
            go.Scatter(x=[f.lambda_c, f.lambda_c], y=[p_lo, p_hi], mode="lines", name="λc",
                       line=dict(color="#999999", width=1, dash="dash"), hoverinfo="skip"),
            go.Scatter(x=seg["time"], y=seg["phase_diff"], mode="lines", name="readout", showlegend=False,
                       line=dict(color=RED, width=1.6)),
            go.Scatter(x=seg["time"] if has_cond else [], y=seg["temperature"] if has_cond else [],
                       mode="lines", name=cond_name, showlegend=False, line=dict(color=BLUE, width=1.6)),
        ]

    base = frame_traces(0)
    fig.add_trace(go.Scatter(x=wl[::sc], y=data.spectra[rows[0]][::sc], mode="lines", name="first spectrum",
                             line=dict(color="#aaaaaa", width=1), opacity=0.5, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(base[0], row=1, col=1)
    fig.add_trace(base[1], row=1, col=1)
    fig.add_trace(base[2], row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=fit0.phase(xs), mode="lines", name="first-frame cubic",
                             line=dict(color="#bbbbbb", width=1.2), hoverinfo="skip"), row=1, col=2)
    fig.add_trace(base[3], row=1, col=2)
    fig.add_trace(base[4], row=1, col=2)
    fig.add_trace(base[5], row=1, col=2)
    fig.add_trace(base[6], row=2, col=2, secondary_y=False)
    fig.add_trace(base[7], row=2, col=2, secondary_y=True)
    for w in (la, lb):
        fig.add_vline(x=w, line_dash="dot", line_color="#777777", row=1, col=1)
    animated = [1, 2, 3, 5, 6, 7, 8, 9]

    def stamp(r):
        text = f"spectrum {r + 1}/{len(data)} · t = {data.time[r] / 60:.0f} min · λc = {fits[r].lambda_c:.2f} nm<br>readout {series['phase_diff'][r]:.2f} rad"
        if has_cond:
            text += f" · {cond_name} = {data.temperature[r]:.2f} {cond_unit}"
        return text

    fig.frames = [go.Frame(name=str(k), data=frame_traces(k), traces=animated,
                           layout=go.Layout(title_text=stamp(rows[k]))) for k in range(len(rows))]
    t_min = data.time / 60.0
    label_every = max(1, len(rows) // 12)
    step = dict(duration=int(1000 / fps), redraw=True)
    fig.update_layout(
        height=700, margin=dict(l=10, r=10, t=100, b=90),
        title=dict(text=stamp(rows[0]), x=0, xanchor="left", y=0.985, yanchor="top"),
        legend=dict(orientation="h", y=-0.32, yanchor="top", x=0, font=dict(size=11)),
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=1.0, xanchor="left", yanchor="bottom",
                          direction="right", pad=dict(b=4),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=step, fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.0, y=-0.14, len=1.0, pad=dict(t=0), ticklen=4,
                      currentvalue=dict(visible=False),
                      steps=[dict(method="animate", label=(f"{t_min[r]:.0f}" if k % label_every == 0 else ""),
                                  value=str(k),
                                  args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                             for k, r in enumerate(rows)])],
    )
    fig.update_xaxes(title_text="Wavelength (nm)", range=[wl.min(), wl.max()], row=1, col=1)
    fig.update_yaxes(title_text="Transmittance", range=[lo - 0.08 * (hi - lo), hi + 0.12 * (hi - lo)], row=1, col=1)
    fig.update_xaxes(title_text="Wavelength (nm)", range=[wl.min(), wl.max()], row=1, col=2)
    fig.update_yaxes(title_text="Δφ (rad)", range=[p_lo, p_hi], row=1, col=2)
    ok = series.dropna(subset=["phase_diff"])
    pad = 0.08 * (ok["phase_diff"].max() - ok["phase_diff"].min() + 1e-9)
    fig.update_xaxes(title_text="Time (s)", range=[data.time.min(), data.time.max()], row=2, col=2)
    fig.update_yaxes(title_text="rad", color=RED, range=[ok["phase_diff"].min() - pad, ok["phase_diff"].max() + pad],
                     row=2, col=2, secondary_y=False)
    if has_cond:
        fig.update_yaxes(title_text=f"{cond_name} ({cond_unit})", color=BLUE, showgrid=False,
                         range=[np.nanmin(data.temperature), np.nanmax(data.temperature)],
                         row=2, col=2, secondary_y=True)
    return fig


@st.cache_data(show_spinner=False)
def phase_movie_gif(_data: SpectraSet, _fits, data_key: str, la: float, lb: float, every: int, fps: int,
                    cond_name: str, cond_unit: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = render_phase_movie(_data, Path(tmp) / "phase.gif", fits=_fits, lambda_a=la, lambda_b=lb,
                                 every=every, fps=fps, cond_name=cond_name, cond_unit=cond_unit, dpi=80)
        return out.read_bytes()


@st.cache_data(show_spinner=False)
def movie_gif(_data: SpectraSet, _track, data_key: str, track_label: str, every: int, fps: int,
              prominence: float, distance: int, smooth_window: int, cond_name: str, cond_unit: str,
              delta: bool = False, follow_nm: float | None = None, trail: int = 8, mark_fringe: bool = True) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = render_movie(_data, Path(tmp) / "movie.gif", track=_track, track_label=track_label,
                           every=every, fps=fps, prominence=prominence, distance=distance,
                           smooth_window=smooth_window, cond_name=cond_name, cond_unit=cond_unit, dpi=80,
                           delta=delta, follow_nm=follow_nm, trail=trail, mark_fringe=mark_fringe)
        return out.read_bytes()


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

tab_map, tab_spec, tab_track, tab_sens, tab_phase, tab_3d, tab_movie, tab_export = st.tabs(
    [":material/gradient: Spectral map", ":material/show_chart: Spectrum & peaks",
     ":material/timeline: Fringe tracking", ":material/thermostat: Sensitivity",
     ":material/waves: Phase", ":material/view_in_ar: 3-D view", ":material/movie: Movie",
     ":material/download: Export"])

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

# ---- phase unwrapping (SM1E.2) ---------------------------------------------
with tab_phase:
    st.markdown("**Phase-unwrapping readout** (Salik et al., Optica Sensing Congress 2025, SM1E.2). "
                "The extrema on either side of the critical wavelength are numbered λ±1, λ±2, …; "
                "λ+n and λ−n share a phase and successive extrema are π apart, so "
                "Δφ(λ±n) = −(n−1)π. A cubic through these points gives Δφ(λ) for every wavelength, and "
                "its maximum is λc. The readout is the phase difference between two wavelengths, which "
                "does not depend on how the extrema are labelled — that is what lets it run past one "
                "fringe spacing without hopping.")
    fits_p = phase_fits_cached(data, data_key, prominence, distance, smooth_window)
    fit0_p = next((f for f in fits_p if f is not None), None)
    if fit0_p is None:
        st.info("No spectrum has at least two extrema on each side of a critical wavelength inside the "
                "selected band, so the phase profile cannot be built. Widen the wavelength range or lower "
                "the prominence.")
    else:
        wl_min, wl_max = float(data.wavelengths.min()), float(data.wavelengths.max())
        c = st.columns([1.2, 1.2, 2])
        la = c[0].number_input("Wavelength a (nm)", wl_min, wl_max,
                               float(np.clip(PAPER_LAMBDA_A, wl_min, wl_max)), 0.5, key="phase_a")
        lb = c[1].number_input("Wavelength b (nm)", wl_min, wl_max,
                               float(np.clip(PAPER_LAMBDA_B, wl_min, wl_max)), 0.5, key="phase_b")
        c[2].caption("Defaults are the pair used in the paper (1545 and 1570 nm). The pair scan below "
                     "shows which pair responds most to this run's condition.")
        ser = phase_series(data, la, lb, fits=fits_p)
        n_fit = int(ser["phase_diff"].notna().sum())
        sens_p = None
        if has_cond and n_fit >= 3:
            try:
                sens_p = phase_sensitivity(ser)
            except ValueError:
                sens_p = None
        lam = ser["lambda_c"].dropna()
        tiles = [("λc (first → last)", f"{lam.iloc[0]:.1f} → {lam.iloc[-1]:.1f} nm" if len(lam) else "—",
                  f"cubic fit RMS {ser['rms'].median():.3f} rad (median)"),
                 ("Spectra fitted", f"{n_fit} / {len(ser)}", f"{len(fit0_p.wavelengths)} extrema in the first")]
        if sens_p is not None:
            tiles += [(f"Readout per {cond_unit}", f"{sens_p.slope * 1e3:+.1f} mrad",
                       f"± {sens_p.stderr * 1e3:.1f} · R² {sens_p.r_squared:.4f}"),
                      ("Span over the run", f"{sens_p.span_rad:.2f} rad", f"= {sens_p.span_fringes:.2f} fringe")]
        stat_strip(tiles)
        st.plotly_chart(phase_profile_figure(data, fit0_p), width="stretch")
        st.plotly_chart(phase_readout_figure(ser, sens_p, la, lb, has_cond, cond_title), width="stretch")
        if has_cond:
            with st.expander("Which wavelength pair works best for this run? · what the readout sees"):
                grid = np.arange(np.ceil(wl_min / 5) * 5 + 5, wl_max - 4, 5.0)
                scan = scan_pairs(fits_p, data.temperature, grid)
                if len(scan):
                    st.dataframe(scan.sort_values("resolution").head(10).rename(columns={
                        "lambda_a": "a (nm)", "lambda_b": "b (nm)", "slope": f"rad/{cond_unit}",
                        "r_squared": "R²", "resid_rad": "scatter (rad)", "resolution": f"resolution ({cond_unit})",
                        "span_rad": "span (rad)"}), hide_index=True, width="stretch",
                        column_config={f"rad/{cond_unit}": st.column_config.NumberColumn(format="%+.4f"),
                                       "R²": st.column_config.NumberColumn(format="%.4f"),
                                       "scatter (rad)": st.column_config.NumberColumn(format="%.3f"),
                                       f"resolution ({cond_unit})": st.column_config.NumberColumn(format="%.2f"),
                                       "span (rad)": st.column_config.NumberColumn(format="%.2f")})
                    st.caption("Resolution = scatter about the line ÷ slope; curvature counts as scatter, so it "
                               "is an upper bound.")
                common = tracked_phase_change(fit0_p, {k: tracks[k] for k in tracked["fringe"]})
                if len(common):
                    st.markdown(
                        f"Peak tracking implies that every fringe's phase changed by "
                        f"**{common.phase_change_rad.mean():+.2f} rad** on average over the run "
                        f"({common.phase_change_rad.mean() / np.pi:+.2f}π), with a spread of "
                        f"{common.phase_change_rad.min():.2f}…{common.phase_change_rad.max():.2f} rad across the band. "
                        "The average is common to all wavelengths and cancels in a phase difference; only the "
                        "spread reaches this readout. A strain run, where the phase profile changes shape, is "
                        "where the paper demonstrates the method.")
        st.download_button("Phase readout per spectrum (CSV)", ser.to_csv(index=False),
                           f"{Path(file_name).stem}_phase_series.csv", "text/csv")

# ---- 3-D ------------------------------------------------------------------
with tab_3d:
    c = st.columns([1, 1, 3])
    y3 = c[0].selectbox("Depth axis", y_modes, index=len(y_modes) - 1, format_func=y_label, key="y3")
    cmap3 = c[1].selectbox("Colours", ["Viridis", "Plasma", "Inferno", "Cividis", "Turbo"], key="c3")
    if c[2].toggle("Draw the 3-D surface", False,
                   help="Off by default because the surface takes a moment to draw."):
        st.plotly_chart(surface_figure(data, y3, cond_title, cmap3), width="stretch")
        st.caption("Transmittance as height. Drag to rotate, scroll to zoom.")

# ---- movie ----------------------------------------------------------------
with tab_movie:
    movie_kind = st.radio("Movie", ["Spectra and fringes", "Phase readout (SM1E.2)"], horizontal=True,
                          key="movie_kind", label_visibility="collapsed")
    if movie_kind == "Phase readout (SM1E.2)":
        st.markdown("The phase-unwrapping readout as a movie: each frame numbers the extrema λ±n around λc, "
                    "fits the cubic Δφ(λ) (the first frame's cubic stays as a grey ghost) and draws the "
                    "readout up to the current time. Watch the extremum at λc change type when a new pair "
                    "of extrema is born there — the readout does not jump. The wavelength pair is the one "
                    "chosen in the Phase tab.")
        la_m = float(st.session_state.get("phase_a", PAPER_LAMBDA_A))
        lb_m = float(st.session_state.get("phase_b", PAPER_LAMBDA_B))
        fits_m = phase_fits_cached(data, data_key, prominence, distance, smooth_window)
        if not any(f is not None for f in fits_m):
            st.info("No spectrum can be fitted with the phase model in the selected band.")
        else:
            c = st.columns([1, 1, 1.2])
            every_p = c[0].slider("Every N-th spectrum", 1, max(2, len(data) // 20),
                                  max(1, int(np.ceil(len(data) / MOVIE_MAX_FRAMES))), key="pm_every",
                                  help="Fewer frames load faster; the last spectrum is always included.")
            fps_p = c[1].slider("Frames per second", 2, 30, 12, key="pm_fps")
            rows_p = [int(r) for r in frame_rows(len(data), every_p, MOVIE_MAX_FRAMES) if fits_m[int(r)] is not None]
            st.plotly_chart(phase_movie_figure(data, rows_p, fits_m, la_m, lb_m, fps_p, cond_name, cond_unit),
                            width="stretch")
            st.caption(f"{len(rows_p)} frames · readout |φ({lb_m:g} nm) − φ({la_m:g} nm)|. Press ▶ Play, or drag "
                       "the slider (time in minutes).")
            with c[2]:
                st.write("")
                if st.button("Render as GIF for slides", key="pm_gif_btn",
                             help="Draws every frame with matplotlib; takes a minute."):
                    t0 = _time.time()
                    with st.spinner("Rendering the GIF…"):
                        gif = phase_movie_gif(data, fits_m, data_key, la_m, lb_m, every_p, fps_p, cond_name, cond_unit)
                    st.session_state["phase_movie_gif"] = (gif, f"{Path(file_name).stem}_phase_movie.gif")
                    st.caption(f"{len(gif) / 1e6:.1f} MB in {_time.time() - t0:.0f} s")
                if "phase_movie_gif" in st.session_state:
                    gif, name = st.session_state["phase_movie_gif"]
                    st.download_button("Download GIF", gif, name, "image/gif", key="pm_gif_dl")
    else:
        st.markdown("The run as a movie: the spectrum frame by frame with its peaks and dips, the "
                    "map with a cursor at the current time, and the tracked fringes drawn as time advances "
                    "— the Igor Pro movie, in the browser.")
        c = st.columns([1.3, 1, 1, 1.2])
        labels = list(table["fringe"])
        if labels:
            ranked = tracked.dropna(subset=["sensitivity_pm"]) if has_cond else tracked
            default_m = (ranked.loc[ranked["sensitivity_pm"].abs().idxmax(), "fringe"]
                         if has_cond and len(ranked) else (tracked["fringe"].iloc[0] if len(tracked) else labels[0]))
            movie_label = c[0].selectbox("Fringe to follow", labels, index=labels.index(default_m), key="movie_fringe",
                                         help="Marked on the spectrum, used by the camera and by the lower-right panel.")
            movie_track = tracks[movie_label]
        else:
            movie_label, movie_track = "", None
        every_m = c[1].slider("Every N-th spectrum", 1, max(2, len(data) // 20), max(1, int(np.ceil(len(data) / MOVIE_MAX_FRAMES))),
                              help="Fewer frames load faster; the last spectrum is always included.")
        fps_m = c[2].slider("Frames per second", 2, 30, 12)

        with st.expander("Movie options", expanded=False):
            o = st.columns([1.2, 1.2, 1, 1])
            mode_m = o[0].radio("Show", ["Transmittance", "Change from first spectrum"], key="movie_mode",
                                help="Δ mode subtracts the first spectrum: only what moved is left.")
            panel_m = o[1].radio("Lower-right panel", ["One fringe", "All fringes", "None"], key="movie_panel",
                                 help="One fringe: Igor-style wavelength + condition. All fringes: every tracked "
                                      "fringe's shift from its start, with the current points marked.")
            camera = o[2].radio("Camera", ["Whole spectrum", "Follow the fringe"], key="movie_cam")
            follow_m = o[2].slider("Window (± nm)", 1.0, 20.0, 6.0, 0.5, key="movie_win",
                                   disabled=camera != "Follow the fringe")
            trail_m = o[3].slider("Trail (previous frames)", 0, 20, 8, key="movie_trail",
                                  help="Past peak/dip positions fade out behind the current ones.")
            mark_m = o[3].checkbox("Mark the followed fringe", True, key="movie_mark")
            big_m = o[3].checkbox(f"Big {cond_name.lower()} readout", True, key="movie_big", disabled=not has_cond)
        follow_nm_m = follow_m if (camera == "Follow the fringe" and movie_track is not None) else None

        rows_m = tuple(int(r) for r in frame_rows(len(data), every_m, MOVIE_MAX_FRAMES))
        ext_m = movie_extrema(data, data_key, rows_m, prominence, distance, smooth_window)
        all_m = {k: tracks[k] for k in tracked["fringe"]} if panel_m == "All fringes" else None
        st.plotly_chart(movie_figure(data, rows_m, ext_m, movie_track, movie_label, cond_name, cond_unit, fps_m,
                                     mode=mode_m, follow_nm=follow_nm_m, trail=trail_m, mark_fringe=mark_m,
                                     panel=panel_m, all_tracks=all_m, big_readout=big_m and has_cond),
                        width="stretch")
        st.caption(f"{len(rows_m)} frames. Press ▶ Play, or drag the slider (time in minutes). "
                   "Peaks and dips are detected with the sidebar settings on every frame; grey markers are "
                   "their positions in the previous frames.")
        with c[3]:
            st.write("")
            if st.button("Render as GIF for slides", help="Draws every frame with matplotlib using the options "
                                                           "above (panel: one fringe); takes a minute."):
                t0 = _time.time()
                with st.spinner("Rendering the GIF…"):
                    gif = movie_gif(data, movie_track, data_key, movie_label, every_m, fps_m,
                                    prominence, distance, smooth_window, cond_name, cond_unit,
                                    delta=(mode_m == "Change from first spectrum"), follow_nm=follow_nm_m,
                                    trail=trail_m, mark_fringe=mark_m)
                st.session_state["movie_gif"] = (gif, f"{Path(file_name).stem}_movie.gif")
                st.caption(f"{len(gif) / 1e6:.1f} MB in {_time.time() - t0:.0f} s")
            if "movie_gif" in st.session_state:
                gif, name = st.session_state["movie_gif"]
                st.download_button("Download GIF", gif, name, "image/gif")

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
