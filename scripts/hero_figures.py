#!/usr/bin/env python3
"""Presentation versions of the spectral map ("the whole experiment in one image").

Produces:
  fig15_spectral_map_annotated.png  - heatmap + every tracked fringe overlaid,
                                      critical wavelength marked, temperature
                                      axis on the right
  fig16_spectral_map_vs_temperature.png - same map re-indexed by temperature:
                                      fringe trajectories become straight lines
  fig17_spectral_map_zooms.png      - three zooms: below lambda_c, the
                                      lambda_c region, above lambda_c

Usage: python scripts/hero_figures.py [csv] --outdir figures     (default csv: data/data.csv)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sms_analysis import load_sms_csv, track_all_extrema, sensitivity_map  # noqa: E402

LAMBDA_C = 1525.0
CMAP = "viridis"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13})


def _temperature_axis(ax, data):
    """Right-hand axis that reads temperature at the time ticks."""
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ticks = ax.get_yticks()
    ticks = ticks[(ticks >= ax.get_ylim()[0]) & (ticks <= ax.get_ylim()[1])]
    t_min = data.time / 60.0
    labels = [f"{np.interp(t, t_min, data.temperature):.0f} °C" for t in ticks]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels)
    ax2.set_ylabel("Temperature")
    ax2.grid(False)
    return ax2


def annotated_map(data, tracks, out):
    fig, ax = plt.subplots(figsize=(12, 6.4))
    t_min = data.time / 60.0
    mesh = ax.pcolormesh(data.wavelengths, t_min, data.spectra, cmap=CMAP,
                         shading="auto", rasterized=True)
    cb = plt.colorbar(mesh, ax=ax, pad=0.09)
    cb.set_label("Transmittance")

    for label, tr in tracks.items():
        ok = tr.dropna(subset=["wavelength"])
        if len(ok) < 0.9 * len(data):
            continue
        ax.plot(ok["wavelength"], ok["time"] / 60.0, color="white", lw=0.9, alpha=0.85)

    ax.axvline(LAMBDA_C, color="#ff5555", ls="--", lw=1.4)
    ax.text(LAMBDA_C + 1.5, t_min.max() * 0.55, "critical wavelength\nλc ≈ 1525 nm",
            color="#ff5555", va="center", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(1500, 250), xytext=(1494, 250),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax.text(1494, 262, "cooling → longer λ", color="white", fontsize=10)
    ax.annotate("", xy=(1546, 250), xytext=(1552, 250),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax.text(1543, 262, "cooling → shorter λ", color="white", fontsize=10)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Time (min)")
    ax.set_title("The whole run in one image: 730 spectra, 44 → 23 °C, white lines = tracked fringes")
    ax.grid(False)
    if data.has_temperature:
        _temperature_axis(ax, data)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def map_vs_temperature(data, tracks, out):
    order = np.argsort(data.temperature)
    fig, ax = plt.subplots(figsize=(12, 6.4))
    mesh = ax.pcolormesh(data.wavelengths, data.temperature[order], data.spectra[order],
                         cmap=CMAP, shading="auto", rasterized=True)
    cb = plt.colorbar(mesh, ax=ax)
    cb.set_label("Transmittance")
    for label, tr in tracks.items():
        ok = tr.dropna(subset=["wavelength"])
        if len(ok) < 0.9 * len(data):
            continue
        ax.plot(ok["wavelength"], ok["temperature"], color="white", lw=0.9, alpha=0.85)
    ax.axvline(LAMBDA_C, color="#ff5555", ls="--", lw=1.4)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Same data indexed by temperature: each fringe becomes a calibration line, slope = sensitivity")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def zooms(data, tracks, smap, out):
    panels = [
        ((1476, 1512), "Below λc: fringes drift to longer λ,\nsensitivity −88 … −320 pm/°C"),
        ((1510, 1545), "λc region: fringes broaden and merge;\nnearby fringes are lost after ~40 min"),
        ((1538, 1575), "Above λc: fringes drift to shorter λ,\nsensitivity +115 … +403 pm/°C"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    t_min = data.time / 60.0
    for ax, ((lo, hi), title) in zip(axes, panels):
        m = (data.wavelengths >= lo) & (data.wavelengths <= hi)
        ax.pcolormesh(data.wavelengths[m], t_min, data.spectra[:, m], cmap=CMAP,
                      shading="auto", rasterized=True)
        for label, tr in tracks.items():
            ok = tr.dropna(subset=["wavelength"])
            ok = ok[(ok["wavelength"] >= lo) & (ok["wavelength"] <= hi)]
            if len(ok) > 20:
                ax.plot(ok["wavelength"], ok["time"] / 60.0, color="white", lw=0.9, alpha=0.85)
        if lo < LAMBDA_C < hi:
            ax.axvline(LAMBDA_C, color="#ff5555", ls="--", lw=1.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(False)
    axes[0].set_ylabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="?", default="data/data.csv",
                   help="spectra CSV (default: data/data.csv)")
    p.add_argument("--outdir", default="figures")
    a = p.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    data = load_sms_csv(a.csv)
    tracks = track_all_extrema(data)
    smap = sensitivity_map(tracks, min_r2=0.9)

    annotated_map(data, tracks, outdir / "fig15_spectral_map_annotated.png")
    if data.has_temperature:
        map_vs_temperature(data, tracks, outdir / "fig16_spectral_map_vs_temperature.png")
    else:
        print("no temperature column -> fig16 (map vs temperature) skipped")
    zooms(data, tracks, smap, outdir / "fig17_spectral_map_zooms.png")
    print(f"written: fig15, fig16, fig17 in {outdir}/")


if __name__ == "__main__":
    main()
