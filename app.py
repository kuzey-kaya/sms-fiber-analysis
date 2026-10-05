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

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sms_analysis import detect_extrema, fit_sensitivity, load_sms_csv, track_all_extrema
from sms_analysis.io import SpectraSet

APP_NAME = "FringeLab"
ROOT = Path(__file__).resolve().parent
EXAMPLE_CSV = ROOT / "data" / "data.csv"

RED, BLUE = "#ff0000", "#0000ff"        # Igor Pro trace colours
PEAK_COLOR, DIP_COLOR = "#C4402F", "#2E5FA3"
MAX_MAP_ROWS, MAX_MAP_COLS = 1200, 2000  # display resolution of the heatmap
MAX_3D_ROWS, MAX_3D_COLS = 150, 300

st.set_page_config(page_title=APP_NAME, page_icon="〰️", layout="wide")


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
                   sensitivity_pm=np.nan, stderr_pm=np.nan, r_squared=np.nan)
        if has_cond:
            try:
                fit = fit_sensitivity(tr)
                row.update(sensitivity_pm=fit.slope * 1e3, stderr_pm=fit.stderr * 1e3,
                           r_squared=fit.r_squared)
            except ValueError:
                pass
        rows.append(row)
    cols = ["fringe", "kind", "start_nm", "end_nm", "shift_nm", "locked_pct",
            "sensitivity_pm", "stderr_pm", "r_squared"]
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


def calibration_figure(track, fit, cond_title, unit):
    ok = track.dropna(subset=["wavelength", "temperature"])
    t = np.linspace(ok["temperature"].min(), ok["temperature"].max(), 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ok["temperature"], y=ok["wavelength"], mode="markers",
                             name="tracked", marker=dict(size=4, color=PEAK_COLOR, opacity=0.5)))
    fig.add_trace(go.Scatter(
        x=t, y=fit.slope * t + fit.intercept, mode="lines", line=dict(color="#333333", width=2),
        name=f"fit: {fit.slope * 1e3:+.1f} pm/{unit} (R² = {fit.r_squared:.4f})"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=cond_title, yaxis_title="Wavelength (nm)",
                      legend=dict(orientation="h", y=1.12))
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
st.sidebar.title(APP_NAME)
st.sidebar.caption("Explore fiber-sensor spectra: map, peaks, fringe tracking, calibration.")

st.sidebar.header("1 · Data")
sources = ["Upload a CSV file"]
if EXAMPLE_CSV.exists():
    sources.insert(0, "Example run (data/data.csv)")
source = st.sidebar.radio("Source", sources, label_visibility="collapsed")

file_bytes, suffix, file_name = None, ".csv", EXAMPLE_CSV.name
example_stamp = f"{EXAMPLE_CSV}:{EXAMPLE_CSV.stat().st_mtime_ns}" if EXAMPLE_CSV.exists() else ""
if source == "Upload a CSV file":
    upload = st.sidebar.file_uploader("Spectra file", type=["csv", "txt", "tsv", "dat"])
    if upload is None:
        st.title(APP_NAME)
        st.info(
            "Upload a spectra file in the sidebar to begin.\n\n"
            "Expected layout: one row per spectrum, one column per wavelength with the "
            "wavelength in nm as the column header, plus optional time and condition "
            "(temperature, strain, …) columns. Transposed files, `;`/tab separators and "
            "decimal commas are detected automatically."
        )
        st.stop()
    file_bytes, suffix, file_name = upload.getvalue(), Path(upload.name).suffix or ".csv", upload.name

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
    st.title(APP_NAME)
    st.error(f"Could not read **{file_name}**: {exc}")
    st.stop()

wl_lo, wl_hi = float(full.wavelengths.min()), float(full.wavelengths.max())
wl_range = st.sidebar.slider("Wavelength range (nm)", wl_lo, wl_hi, (wl_lo, wl_hi))
keep = (full.wavelengths >= wl_range[0]) & (full.wavelengths <= wl_range[1])
if keep.sum() < 20:
    st.error("The selected wavelength range is too narrow.")
    st.stop()
data = replace(full, wavelengths=full.wavelengths[keep], spectra=full.spectra[:, keep])
has_cond = data.has_temperature

st.sidebar.header("2 · Peak detection")
prominence = st.sidebar.number_input(
    "Prominence (transmittance)", 0.0005, 1.0, 0.02, 0.005, format="%.4f",
    help="How much a peak/dip must stand out from its surroundings. Lower = more, weaker extrema.")
distance = st.sidebar.slider("Minimum separation (samples)", 1, 100, 5)
smooth_window = st.sidebar.slider("Smoothing window (samples)", 5, 51, 11, step=2,
                                  help="Savitzky–Golay window applied before detection.")

st.sidebar.header("3 · Fringe tracking")
search_window = st.sidebar.slider(
    "Search half-window (nm)", 0.1, 10.0, 1.5, 0.1,
    help="How far from its last position a fringe is searched. Keep below the fringe spacing.")
reject = st.sidebar.checkbox("Reject jumps (fringe-hop guard)", True)
max_step = st.sidebar.slider("Largest allowed jump per spectrum (nm)", 0.05, 5.0, 0.5, 0.05,
                             disabled=not reject) if reject else None
min_lock = st.sidebar.slider("Count a fringe as tracked when locked ≥ (%)", 10, 100, 90, 5)

data_key = hashlib.md5(
    (file_bytes or example_stamp.encode()) + repr((cond_col_txt, time_col_txt, wl_range)).encode()).hexdigest()
tracks = track_all(data, data_key, prominence, distance, search_window, smooth_window, max_step)
table = fringe_table(tracks, has_cond)
tracked = table[table["locked_pct"] >= min_lock]

# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
st.title(APP_NAME)
st.caption(f"**{file_name}** · layout: {data.layout}")
m = st.columns(5)
m[0].metric("Spectra", f"{len(data)}")
m[1].metric("Wavelength", f"{data.wavelengths.min():.1f}–{data.wavelengths.max():.1f} nm")
m[2].metric("Duration", f"{(data.time.max() - data.time.min()) / 60:.0f} min")
m[3].metric(cond_name, f"{np.nanmin(data.temperature):.1f} → {np.nanmax(data.temperature):.1f} {cond_unit}"
            if has_cond else "not in file")
m[4].metric("Fringes tracked", f"{len(tracked)} / {len(table)}")
if not has_cond:
    st.warning(f"No {cond_name.lower()} column was found, so calibration (sensitivity) is skipped. "
               "Maps, peaks and tracking still work. If the file has one, name it under "
               "*Condition and columns* in the sidebar.")

y_modes = ["Time", "Spectrum number"] + (["Condition"] if has_cond else [])
y_label = {"Time": "Time", "Spectrum number": "Spectrum number", "Condition": cond_name}.get

tab_map, tab_spec, tab_track, tab_sens, tab_3d, tab_export = st.tabs(
    ["Spectral map", "Spectrum & peaks", "Fringe tracking", "Sensitivity", "3-D view", "Export"])

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
        if has_cond:
            try:
                fit = fit_sensitivity(track)
                right.plotly_chart(calibration_figure(track, fit, cond_title, cond_unit),
                                   width="stretch")
                right.caption(f"Sensitivity {fit.slope * 1e3:+.1f} ± {fit.stderr * 1e3:.1f} pm/{cond_unit}, "
                              f"R² = {fit.r_squared:.4f}, {fit.n_points} points.")
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
    st.dataframe(
        shown.rename(columns={"sensitivity_pm": f"sensitivity (pm/{cond_unit})",
                              "stderr_pm": f"± (pm/{cond_unit})", "locked_pct": "locked (%)",
                              "start_nm": "start (nm)", "end_nm": "end (nm)", "shift_nm": "shift (nm)",
                              "r_squared": "R²"}),
        hide_index=True, width="stretch",
        column_config={"start (nm)": st.column_config.NumberColumn(format="%.2f"),
                       "end (nm)": st.column_config.NumberColumn(format="%.2f"),
                       "shift (nm)": st.column_config.NumberColumn(format="%+.3f"),
                       "locked (%)": st.column_config.NumberColumn(format="%.0f"),
                       f"sensitivity (pm/{cond_unit})": st.column_config.NumberColumn(format="%+.1f"),
                       f"± (pm/{cond_unit})": st.column_config.NumberColumn(format="%.1f"),
                       "R²": st.column_config.NumberColumn(format="%.4f")})

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
