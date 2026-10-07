"""Animate a run: the spectrum frame by frame with its extrema, the spectral
map with a time cursor, and one tracked fringe drawn as time advances.

This is the Python counterpart of Igor Pro's movie export (stepping a graph
through the rows of a wave and saving each frame).  MP4 needs ffmpeg on the
PATH; without it a GIF is written with Pillow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd

from .igor_style import IGOR_BLUE, IGOR_RED
from .io import SpectraSet
from .peaks import detect_extrema
from .plotting import COLOR_TEMPERATURE, COLOR_WAVELENGTH


def frame_rows(n_rows: int, every: int = 5, max_frames: int | None = None) -> np.ndarray:
    """Row indices to render: every ``every``-th spectrum, always ending on the last."""
    if max_frames:
        every = max(every, int(np.ceil(n_rows / max_frames)))
    rows = np.arange(0, n_rows, max(1, every))
    if rows[-1] != n_rows - 1:
        rows = np.r_[rows, n_rows - 1]
    return rows


def render_movie(
    data: SpectraSet,
    out: str | Path,
    track: pd.DataFrame | None = None,
    track_label: str = "",
    every: int = 5,
    fps: int = 12,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
    cond_name: str = "Temperature",
    cond_unit: str = "°C",
    dpi: int = 90,
    progress=None,
    delta: bool = False,
    follow_nm: float | None = None,
    trail: int = 8,
    mark_fringe: bool = True,
) -> Path:
    """Write the animation to ``out`` (.mp4 with ffmpeg, otherwise .gif).

    ``track`` (output of ``track_feature``) adds the lower-right panel: the
    tracked wavelength and the condition drawn up to the current frame, in the
    Igor colours.  ``delta`` shows each spectrum minus the first one;
    ``follow_nm`` makes the spectrum panel a window of ± that many nm around
    the tracked fringe; ``trail`` leaves the extrema of that many previous
    frames fading out; ``mark_fringe`` draws a dotted line at the tracked
    wavelength.  ``progress`` is an optional callback ``f(i, n)``.
    """
    out = Path(out)
    rows = frame_rows(len(data), every)
    wl = data.wavelengths
    t_min = data.time / 60.0
    has_cond = data.has_temperature

    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.42,
                          top=0.90, bottom=0.09, left=0.07, right=0.93)
    ax_spec = fig.add_subplot(gs[:, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_trk = fig.add_subplot(gs[1, 1])

    # --- spectrum panel ---------------------------------------------------
    s0 = data.spectra[0]

    def spec_y(r):
        return data.spectra[r] - s0 if delta else data.spectra[r]

    def ext_of(r):
        ext = detect_extrema(wl, data.spectra[r], prominence=prominence, distance=distance,
                             smooth_window=smooth_window)
        pw, pv, dw, dv = ext.peak_wavelengths, ext.peak_values, ext.dip_wavelengths, ext.dip_values
        if delta:
            d = data.spectra[r] - s0
            pv, dv = np.interp(pw, wl, d), np.interp(dw, wl, d)
        return pw, pv, dw, dv

    ext_cache: dict[int, tuple] = {}

    if delta:
        ghost = ax_spec.axhline(0, color="#999999", lw=0.8, alpha=0.6, label="first spectrum (= 0)")
    else:
        ghost, = ax_spec.plot(wl, s0, color="#999999", lw=0.8, alpha=0.5, label="first spectrum")
    line, = ax_spec.plot(wl, spec_y(0), color="#333333", lw=1.4, label="current spectrum")
    trail_sc = ax_spec.scatter([], [], s=14, color="#8a8a8a", linewidths=0, zorder=2)
    peaks, = ax_spec.plot([], [], "o", ms=5, color=COLOR_WAVELENGTH, mec="black", mew=0.4, label="peaks", zorder=3)
    dips, = ax_spec.plot([], [], "v", ms=5, color=COLOR_TEMPERATURE, mec="black", mew=0.4, label="dips", zorder=3)
    marker = ax_spec.axvline(np.nan, color=IGOR_RED, lw=1.0, ls=":", label="tracked fringe") \
        if (mark_fringe and track is not None) else None
    trk_wl = (track["wavelength"].ffill().bfill().to_numpy(float) if track is not None
              else np.full(len(data), np.nan))
    ax_spec.set_xlim(wl.min(), wl.max())
    if delta:
        d_all = data.spectra[::max(1, len(data) // 100)] - s0
        lim = float(np.nanpercentile(np.abs(d_all), 99.5)) or 0.1
        ax_spec.set_ylim(-1.08 * lim, 1.12 * lim)
        ax_spec.set_ylabel("Transmittance − first spectrum")
    else:
        lo, hi = np.nanmin(data.spectra), np.nanmax(data.spectra)
        ax_spec.set_ylim(lo - 0.04 * (hi - lo), hi + 0.10 * (hi - lo))
        ax_spec.set_ylabel("Transmittance")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    readout = ax_spec.text(0.02, 0.97, "", transform=ax_spec.transAxes, va="top", ha="left",
                           fontsize=15, fontweight="bold", color=IGOR_BLUE) if has_cond else None
    stamp = fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=12, fontweight="bold")

    # --- map panel with a time cursor --------------------------------------
    sr = max(1, len(data) // 400)
    sc = max(1, len(wl) // 600)
    ax_map.pcolormesh(wl[::sc], t_min[::sr], data.spectra[::sr, ::sc], cmap="viridis",
                      shading="auto", rasterized=True)
    cursor = ax_map.axhline(t_min[0], color="white", lw=1.6)
    ax_map.set_xlabel("Wavelength (nm)")
    ax_map.set_ylabel("Time (min)")
    ax_map.set_title("Spectral map", fontsize=10)
    ax_map.grid(False)

    # --- tracked fringe panel (Igor colours) --------------------------------
    trk_line = cond_line = None
    ax_cond = None
    if track is not None:
        ok = track.dropna(subset=["wavelength"])
        trk_line, = ax_trk.plot([], [], color=IGOR_RED, lw=1.4)
        ax_trk.set_xlim(data.time.min(), data.time.max())
        pad = 0.05 * (ok["wavelength"].max() - ok["wavelength"].min() + 1e-9)
        ax_trk.set_ylim(ok["wavelength"].min() - pad, ok["wavelength"].max() + pad)
        ax_trk.set_xlabel("Time, seconds")
        ax_trk.xaxis.set_major_locator(MaxNLocator(5))
        ax_trk.set_ylabel("Peak Wavelength (nm)", color=IGOR_RED)
        ax_trk.tick_params(axis="y", colors=IGOR_RED)
        ax_trk.set_title(f"Tracked {track_label}".strip(), fontsize=10)
        if has_cond:
            ax_cond = ax_trk.twinx()
            cond_line, = ax_cond.plot([], [], color=IGOR_BLUE, lw=1.4)
            ax_cond.set_ylim(np.nanmin(data.temperature), np.nanmax(data.temperature))
            ax_cond.set_ylabel(f"{cond_name}, {cond_unit}", color=IGOR_BLUE, rotation=270, labelpad=14)
            ax_cond.tick_params(axis="y", colors=IGOR_BLUE)
            ax_cond.grid(False)
    else:
        ax_trk.set_axis_off()

    def draw(k: int):
        r = int(rows[k])
        line.set_ydata(spec_y(r))
        if k not in ext_cache:
            ext_cache[k] = ext_of(r)
        pw, pv, dw, dv = ext_cache[k]
        peaks.set_data(pw, pv)
        dips.set_data(dw, dv)
        # trail of the previous frames' extrema, fading out
        tx, ty, ta = [], [], []
        for j in range(max(0, k - trail), k):
            if j not in ext_cache:
                ext_cache[j] = ext_of(int(rows[j]))
            qw, qv, ew, ev = ext_cache[j]
            op = 0.08 + 0.5 * (j - (k - trail) + 1) / (trail + 1)
            tx += list(qw) + list(ew); ty += list(qv) + list(ev); ta += [op] * (len(qw) + len(ew))
        trail_sc.set_offsets(np.c_[tx, ty] if tx else np.empty((0, 2)))
        trail_sc.set_alpha(None)
        trail_sc.set_facecolor([(0.54, 0.54, 0.54, a) for a in ta] if ta else [])
        if marker is not None and np.isfinite(trk_wl[r]):
            marker.set_xdata([trk_wl[r], trk_wl[r]])
        if follow_nm and np.isfinite(trk_wl[r]):
            ax_spec.set_xlim(trk_wl[r] - follow_nm, trk_wl[r] + follow_nm)
        if readout is not None:
            readout.set_text(f"{data.temperature[r]:.2f} {cond_unit}")
        text = f"spectrum {r + 1}/{len(data)} · t = {data.time[r]:.0f} s"
        if has_cond:
            text += f" · {cond_name} = {data.temperature[r]:.2f} {cond_unit}"
        stamp.set_text(text)
        cursor.set_ydata([t_min[r], t_min[r]])
        if trk_line is not None:
            seg = track.iloc[: r + 1]
            trk_line.set_data(seg["time"], seg["wavelength"])
            if cond_line is not None:
                cond_line.set_data(seg["time"], seg["temperature"])
        if progress:
            progress(k + 1, len(rows))
        return line, peaks, dips, stamp, cursor

    anim = animation.FuncAnimation(fig, draw, frames=len(rows), interval=1000 / fps, blit=False)
    if out.suffix.lower() == ".mp4" and animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    else:
        if out.suffix.lower() != ".gif":
            out = out.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    return out


def render_phase_movie(
    data: SpectraSet,
    out: str | Path,
    fits: list | None = None,
    lambda_a: float = 1545.0,
    lambda_b: float = 1570.0,
    every: int = 5,
    fps: int = 12,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
    cond_name: str = "Temperature",
    cond_unit: str = "°C",
    dpi: int = 90,
    progress=None,
) -> Path:
    """Animate the SM1E.2 phase-unwrapping readout (see :mod:`sms_analysis.phase`).

    Each frame: the spectrum with its extrema numbered λ±n around λc and the
    two readout wavelengths marked; the assigned phases Δφ(λ±n) = −(n−1)π with
    the cubic Δφ(λ) and λc (the first frame's cubic stays as a grey ghost, so
    the change of shape is visible); and the readout |φ(λb) − φ(λa)| with the
    condition drawn up to the current time.  ``fits`` (from
    :func:`sms_analysis.phase.phase_fits`) are computed when not given.
    MP4 with ffmpeg on the PATH, otherwise GIF.
    """
    from .phase import phase_fits, phase_series

    out = Path(out)
    if fits is None:
        fits = phase_fits(data, prominence=prominence, distance=distance, smooth_window=smooth_window)
    series = phase_series(data, lambda_a, lambda_b, fits=fits)
    rows = [int(r) for r in frame_rows(len(data), every) if fits[int(r)] is not None]
    if not rows:
        raise ValueError("No spectrum could be fitted with the phase model")
    wl = data.wavelengths
    xs = np.linspace(wl.min(), wl.max(), 300)
    has_cond = data.has_temperature
    fit0 = fits[rows[0]]

    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], wspace=0.3, hspace=0.45,
                          top=0.90, bottom=0.09, left=0.07, right=0.93)
    ax_s = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])

    # spectrum with numbered extrema
    line, = ax_s.plot(wl, data.spectra[rows[0]], color=COLOR_WAVELENGTH, lw=1.3)
    ext_pts = ax_s.scatter([], [], s=22, color=COLOR_TEMPERATURE, zorder=3)
    for w in (lambda_a, lambda_b):
        ax_s.axvline(w, color="#666666", ls=":", lw=1)
        ax_s.text(w, 1.0, f"{w:g}", transform=ax_s.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=8)
    labels_txt = [ax_s.text(0, 0, "", fontsize=8, ha="center") for _ in range(7)]
    lo, hi = np.nanmin(data.spectra), np.nanmax(data.spectra)
    ax_s.set_xlim(wl.min(), wl.max())
    ax_s.set_ylim(lo - 0.06 * (hi - lo), hi + 0.10 * (hi - lo))
    ax_s.set_xlabel("Wavelength (nm)")
    ax_s.set_ylabel("Transmittance")

    # phase profile, first frame as a ghost
    ax_p.plot(xs, fit0.phase(xs), color="#bbbbbb", lw=1.2, label="first frame")
    curve, = ax_p.plot(xs, fit0.phase(xs), color=COLOR_TEMPERATURE, lw=1.6, label="cubic Δφ(λ)")
    pts, = ax_p.plot([], [], "o", mfc="none", color=COLOR_WAVELENGTH, ms=4)
    lc_line = ax_p.axvline(fit0.lambda_c, color="#999999", ls="--", lw=1)
    allp = np.concatenate([fits[r].phase(xs) for r in rows])
    ax_p.set_xlim(wl.min(), wl.max())
    ax_p.set_ylim(allp.min() - 2, max(allp.max(), 0) + 3)
    ax_p.set_xlabel("Wavelength (nm)")
    ax_p.set_ylabel("Δφ (rad)")
    ax_p.set_title("Phase from the numbered extrema", fontsize=10)
    ax_p.legend(frameon=False, fontsize=8, loc="lower center")

    # readout growing in time (Igor colours)
    rd, = ax_r.plot([], [], color=IGOR_RED, lw=1.4)
    ok = series.dropna(subset=["phase_diff"])
    pad = 0.08 * (ok["phase_diff"].max() - ok["phase_diff"].min() + 1e-9)
    ax_r.set_xlim(data.time.min(), data.time.max())
    ax_r.set_ylim(ok["phase_diff"].min() - pad, ok["phase_diff"].max() + pad)
    ax_r.set_xlabel("Time, seconds")
    ax_r.set_ylabel(f"|φ({lambda_b:g})−φ({lambda_a:g})| (rad)", color=IGOR_RED, fontsize=9)
    ax_r.tick_params(axis="y", colors=IGOR_RED)
    ax_r.xaxis.set_major_locator(MaxNLocator(5))
    ax_r.set_title("Phase-difference readout", fontsize=10)
    cond_line = None
    if has_cond:
        ax_c = ax_r.twinx()
        cond_line, = ax_c.plot([], [], color=IGOR_BLUE, lw=1.4)
        ax_c.set_ylim(np.nanmin(data.temperature), np.nanmax(data.temperature))
        ax_c.set_ylabel(f"{cond_name}, {cond_unit}", color=IGOR_BLUE, rotation=270, labelpad=14)
        ax_c.tick_params(axis="y", colors=IGOR_BLUE)
        ax_c.grid(False)
    stamp = fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=12, fontweight="bold")

    def draw(k: int):
        r = rows[k]
        f = fits[r]
        line.set_ydata(data.spectra[r])
        y = np.interp(f.wavelengths, wl, data.spectra[r])
        ext_pts.set_offsets(np.c_[f.wavelengths, y])
        labs = f.labels()
        near = [i for i, lab in enumerate(labs) if lab == "λc" or int(lab[2:]) <= 3]
        for t, i in zip(labels_txt, near + [None] * (len(labels_txt) - len(near))):
            if i is None:
                t.set_text("")
                continue
            up = f.kinds[i] == "peak"
            t.set_position((f.wavelengths[i], y[i] + (0.035 if up else -0.06) * (hi - lo)))
            t.set_text(labs[i])
        curve.set_ydata(f.phase(xs))
        okp = np.isfinite(f.phases)
        pts.set_data(f.wavelengths[okp], f.phases[okp])
        lc_line.set_xdata([f.lambda_c, f.lambda_c])
        seg = series.iloc[: r + 1]
        rd.set_data(seg["time"], seg["phase_diff"])
        if cond_line is not None:
            cond_line.set_data(seg["time"], seg["temperature"])
        text = (f"spectrum {r + 1}/{len(data)} · λc = {f.lambda_c:.2f} nm · "
                f"readout {series['phase_diff'][r]:.2f} rad")
        if has_cond:
            text += f" · {cond_name} = {data.temperature[r]:.2f} {cond_unit}"
        stamp.set_text(text)
        if progress:
            progress(k + 1, len(rows))
        return line, ext_pts, curve, pts, lc_line, rd, stamp

    anim = animation.FuncAnimation(fig, draw, frames=len(rows), interval=1000 / fps, blit=False)
    if out.suffix.lower() == ".mp4" and animation.FFMpegWriter.isAvailable():
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    else:
        if out.suffix.lower() != ".gif":
            out = out.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    return out
