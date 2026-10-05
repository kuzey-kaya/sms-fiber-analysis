#!/usr/bin/env python3
"""3-D views of the run: wavelength x time x temperature, colored by transmittance.

The cooldown ties time and temperature together, so the spectral map can be
"draped" over the cooling curve: a surface whose height is the measured
temperature and whose color is the transmittance.  Reading it: the surface's
shape IS the temperature-time correlation, the color bands ARE the fringes,
and the white lines are the tracked fringes riding on the surface.

Produces:
  fig18_3d_spectral_surface.png   - the draped surface (slide version)
  fig19_3d_fringe_landscape.png   - transmittance as height over
                                    wavelength x temperature: the fringes as
                                    ridges and valleys, tilting across lambda_c
  fig18_3d_interactive.html       - rotatable version (needs `pip3 install plotly`;
                                    skipped with a note if plotly is missing)

Usage: python scripts/spectral_3d.py [csv] --outdir figures      (default csv: data/data.csv)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sms_analysis import load_sms_csv, track_all_extrema  # noqa: E402

LAMBDA_C = 1525.0
plt.rcParams.update({"font.size": 11})


def _grid(data, step_wl=2, step_t=2):
    """Downsample for rendering (the full 730 x 1001 grid is not needed)."""
    wl = data.wavelengths[::step_wl]
    t = data.time[::step_t] / 60.0
    temp = data.temperature[::step_t]
    Z = data.spectra[::step_t, ::step_wl]
    return wl, t, temp, Z


def draped_surface(data, tracks, out):
    wl, t_min, temp, Z = _grid(data)
    X, Y = np.meshgrid(wl, t_min)
    T = np.repeat(temp[:, None], len(wl), axis=1)
    norm = Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))

    z_floor = temp.min() - 1.0
    fig = plt.figure(figsize=(13, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False  # paint in the order below

    # Floor: the spectral map (wavelength x time), color = transmittance.
    ax.plot_surface(X, Y, np.full_like(X, z_floor), facecolors=colors,
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False, zorder=1)

    # Back wall: the cooling curve, temperature vs time, at the long-wavelength edge.
    ax.plot(np.full(len(t_min), wl.max()), t_min, temp,
            color="#12213A", lw=2.5, zorder=3)
    ax.text(wl.max() + 2, t_min[0], temp[0] + 1.5,
            "cooling curve T(t)", color="#12213A", fontsize=10)

    # Each tracked fringe as a 3-D curve (wavelength, time, temperature),
    # with its shadow on the floor.
    for label, tr in tracks.items():
        ok = tr.dropna(subset=["wavelength"])
        if len(ok) < 0.9 * len(data):
            continue
        ax.plot(ok["wavelength"], ok["time"] / 60.0, np.full(len(ok), z_floor + 0.15),
                color="white", lw=0.9, alpha=0.9, zorder=2)
        c = "#D9A441" if ok["wavelength"].iloc[0] > LAMBDA_C else "#1F8A85"
        ax.plot(ok["wavelength"], ok["time"] / 60.0, ok["temperature"],
                color=c, lw=1.6, alpha=0.95, zorder=4)

    # Critical wavelength: a translucent curtain from the floor to the cooling curve.
    ax.plot(np.full(len(t_min), LAMBDA_C), t_min, temp,
            color="#ff5555", lw=1.8, ls="--", zorder=5)
    ax.plot(np.full(len(t_min), LAMBDA_C), t_min, np.full(len(t_min), z_floor + 0.15),
            color="#ff5555", lw=1.2, ls="--", zorder=2)

    ax.set_xlabel("Wavelength (nm)", labelpad=16)
    ax.set_ylabel("Time (min)", labelpad=16)
    ax.set_zlabel("Temperature (°C)", labelpad=10)
    ax.set_zlim(z_floor, temp.max() + 1)
    ax.view_init(elev=30, azim=-52)
    ax.set_box_aspect((2.2, 1.4, 1.0))
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)

    m = cm.ScalarMappable(norm=norm, cmap="viridis")
    cb = fig.colorbar(m, ax=ax, shrink=0.5, pad=0.12, location="left")
    cb.set_label("Transmittance (floor map)")
    ax.set_title("Wavelength × time × temperature: spectral map on the floor, tracked fringes rising with temperature\n"
                 "teal = fringes below λc, gold = above λc, red dashed = critical wavelength, dark = cooling curve",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fringe_landscape(data, tracks, out):
    wl, _, temp, Z = _grid(data)
    order = np.argsort(temp)
    temp, Z = temp[order], Z[order]
    X, Y = np.meshgrid(wl, temp)
    norm = Normalize(Z.min(), Z.max())

    fig = plt.figure(figsize=(13, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=cm.viridis(norm(Z)), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    for label, tr in tracks.items():
        ok = tr.dropna(subset=["wavelength"])
        if len(ok) < 0.9 * len(data):
            continue
        ax.plot(ok["wavelength"], ok["temperature"], ok["value"] + 0.02,
                color="white", lw=1.0, alpha=0.9)
    ax.set_xlabel("Wavelength (nm)", labelpad=14)
    ax.set_ylabel("Temperature (°C)", labelpad=14)
    ax.set_zlabel("Transmittance", labelpad=8)
    ax.view_init(elev=38, azim=-62)
    ax.set_box_aspect((2.2, 1.4, 0.7))
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_title("Fringe landscape: transmittance as height over wavelength × temperature\n"
                 "ridges = peaks, valleys = dips; they tilt in opposite directions across λc",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def interactive_html(data, tracks, out):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed -> skipping the interactive HTML "
              "(pip3 install plotly, then re-run)")
        return
    wl, t_min, temp, Z = _grid(data, step_wl=2, step_t=2)
    T = np.repeat(temp[:, None], len(wl), axis=1)
    fig = go.Figure()
    fig.add_surface(x=wl, y=t_min, z=T, surfacecolor=Z, colorscale="Viridis",
                    colorbar=dict(title="Transmittance"), showscale=True)
    for label, tr in tracks.items():
        ok = tr.dropna(subset=["wavelength"])
        if len(ok) < 0.9 * len(data):
            continue
        fig.add_scatter3d(x=ok["wavelength"], y=ok["time"] / 60.0, z=ok["temperature"] + 0.3,
                          mode="lines", line=dict(color="white", width=3), name=label)
    fig.update_layout(
        title="Wavelength × time × temperature (drag to rotate)",
        scene=dict(xaxis_title="Wavelength (nm)", yaxis_title="Time (min)",
                   zaxis_title="Temperature (°C)"),
        width=1200, height=750, showlegend=False,
    )
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"interactive: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="?", default="data/data.csv",
                   help="spectra CSV (default: data/data.csv)")
    p.add_argument("--outdir", default="figures")
    a = p.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    data = load_sms_csv(a.csv)
    if not data.has_temperature:
        # No temperature column: use time (minutes) as the third axis instead.
        data.temperature = data.time / 60.0
        print("no temperature column -> using time (min) as the vertical axis of the 3-D views")
    tracks = track_all_extrema(data)
    draped_surface(data, tracks, outdir / "fig18_3d_spectral_surface.png")
    fringe_landscape(data, tracks, outdir / "fig19_3d_fringe_landscape.png")
    interactive_html(data, tracks, outdir / "fig18_3d_interactive.html")
    print(f"written: fig18, fig19 in {outdir}/")


if __name__ == "__main__":
    main()
