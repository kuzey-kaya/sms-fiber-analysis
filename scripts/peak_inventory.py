#!/usr/bin/env python3
"""Peak inventory: export every detected peak/dip of every spectrum, visualise
their trajectories, and build interval-sampled staircase traces.

This follows the advisor's procedure for slowly-varying data:
  1. detect *all* extrema in every spectrum and write them to a file;
  2. plot every extremum against time -> each fringe draws its own drifting
     trajectory, so the direction of motion (toward shorter or longer
     wavelength) is visible per fringe;
  3. sample the tracked position only every `--interval` seconds and hold it
     in between -> the slow drift appears as a staircase, one step per
     sampling interval.

Usage:
    python scripts/peak_inventory.py --interval 1200            # uses data/data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sms_analysis import (  # noqa: E402
    detect_extrema, fit_sensitivity, load_sms_csv, track_all_extrema,
)
from sms_analysis.igor_style import IGOR_BLUE, IGOR_RED, plot_igor_tracking  # noqa: E402


def export_all_extrema(data, prominence, distance) -> pd.DataFrame:
    """Long-format table: one row per detected extremum per spectrum."""
    rows = []
    for i in range(len(data)):
        ext = detect_extrema(data.wavelengths, data.spectra[i],
                             prominence=prominence, distance=distance)
        for kind, wls, vals in (("peak", ext.peak_wavelengths, ext.peak_values),
                                ("dip", ext.dip_wavelengths, ext.dip_values)):
            for k, (w, v) in enumerate(zip(wls, vals)):
                rows.append((i, data.time[i], data.temperature[i], kind, k, w, v))
    return pd.DataFrame(rows, columns=["row", "time", "temperature", "kind",
                                       "order", "wavelength", "value"])


def interval_sample(track: pd.DataFrame, interval: float) -> pd.DataFrame:
    """Keep one point per `interval` seconds (the first row of each interval)."""
    ok = track.dropna(subset=["wavelength"])
    bucket = np.floor((ok["time"] - ok["time"].iloc[0]) / interval)
    return ok.groupby(bucket).first().reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="?", default="data/data.csv",
                   help="spectra CSV (default: data/data.csv)")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--prominence", type=float, default=0.02)
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--interval", type=float, default=1200.0,
                   help="sampling interval in seconds for the staircase view")
    p.add_argument("--fringes", nargs="*", default=None,
                   help="fringe labels to show (e.g. dip@1542.3); default: three "
                        "fringes -- one below, one near, one above the critical wavelength")
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    data = load_sms_csv(args.csv)
    print(data)

    # 1. every extremum of every spectrum -> file --------------------------
    allx = export_all_extrema(data, args.prominence, args.distance)
    allx.to_csv(outdir / "all_peaks_long.csv", index=False)
    print(f"all_peaks_long.csv: {len(allx)} extrema over {len(data)} spectra")

    # tracked (identity-resolved) version, wide: one column per fringe
    tracks = track_all_extrema(data, prominence=args.prominence, distance=args.distance)
    wide = pd.DataFrame({"time": data.time, "temperature": data.temperature})
    for label, tr in tracks.items():
        wide[label] = tr["wavelength"].to_numpy()
    wide.to_csv(outdir / "all_peaks_tracked_wide.csv", index=False)
    print(f"all_peaks_tracked_wide.csv: {len(tracks)} tracked fringes")

    # direction table
    rows = []
    for label, tr in tracks.items():
        w = tr["wavelength"].dropna()
        if w.size < 3:
            continue
        row = dict(fringe=label, start_nm=w.iloc[0], end_nm=w.iloc[-1],
                   total_shift_nm=w.iloc[-1] - w.iloc[0],
                   direction="→ longer λ" if w.iloc[-1] > w.iloc[0] else "→ shorter λ",
                   sensitivity_pm_per_C=float("nan"), r_squared=float("nan"),
                   locked_rows=int(w.size))
        if data.has_temperature:
            try:
                fit = fit_sensitivity(tr)
                row.update(sensitivity_pm_per_C=fit.slope * 1e3, r_squared=fit.r_squared)
            except ValueError:
                pass
        rows.append(row)
    direction = pd.DataFrame(rows).sort_values("start_nm").reset_index(drop=True)
    direction.to_csv(outdir / "fringe_directions.csv", index=False)
    print("\nDirection of every fringe during the run:")
    print(direction.round(3).to_string(index=False))

    # 2. trajectory map: all extrema vs time --------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, color, marker in (("peak", IGOR_RED, "."), ("dip", IGOR_BLUE, ".")):
        sel = allx[allx["kind"] == kind]
        ax.plot(sel["wavelength"], sel["time"], marker, ms=1.5, color=color,
                alpha=0.6, label=f"{kind}s", rasterized=True)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Every detected extremum vs time (each streak = one fringe drifting)")
    ax.legend(markerscale=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig13_peak_trajectories.png", dpi=130)
    plt.close(fig)

    # 3. interval-sampled staircase for selected fringes --------------------
    if args.fringes:
        chosen = args.fringes
    else:
        good = direction[direction["locked_rows"] >= 0.9 * len(data)]
        if good.empty:
            good = direction
        below = good[good["start_nm"] < 1515]
        above = good[good["start_nm"] > 1535]
        chosen = []
        if len(below):
            chosen.append(below.iloc[-1]["fringe"])   # closest below λc
        if len(above):
            chosen.append(above.iloc[0]["fringe"])    # closest above λc
        chosen.append(good.iloc[-1]["fringe"])        # band edge
        chosen = list(dict.fromkeys(chosen))          # unique, keep order

    fig, axes = plt.subplots(len(chosen), 1, figsize=(9, 3.2 * len(chosen)), sharex=True)
    for ax, label in zip(np.atleast_1d(axes), chosen):
        tr = tracks[label]
        samp = interval_sample(tr, args.interval)
        ax.plot(tr["time"], tr["wavelength"], color=IGOR_RED, lw=0.6, alpha=0.3)
        ax.step(samp["time"], samp["wavelength"], where="post", color=IGOR_RED, lw=1.6)
        ax.set_ylabel("Peak Wavelength (nm)")
        ax.set_title(f"{label}: sampled every {args.interval:.0f} s", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(tr["time"], tr["temperature"], color=IGOR_BLUE, lw=1.0)
        ax2.set_ylabel("Temperature, °C")
        ax.grid(True, alpha=0.25)
    np.atleast_1d(axes)[-1].set_xlabel("Time, seconds")
    fig.tight_layout()
    fig.savefig(outdir / "fig14_interval_staircase.png", dpi=130)
    plt.close(fig)

    # Igor-style single figure for the most sensitive chosen fringe
    label = chosen[1] if len(chosen) > 1 else chosen[0]
    samp = interval_sample(tracks[label], args.interval)
    fig, ax, ax2 = plot_igor_tracking(tracks[label], wl_ticks=2, temp_ticks=5,
                                      time_ticks=5000, temp_limits=(20, 45))
    for line in list(ax.lines):
        line.remove()
    ax.step(samp["time"], samp["wavelength"], where="post", color=IGOR_RED, lw=1.5)
    fig.savefig(outdir / "fig14_interval_staircase_igor.png", dpi=130)
    plt.close(fig)

    print(f"\nWritten to {outdir}/: all_peaks_long.csv, all_peaks_tracked_wide.csv, "
          f"fringe_directions.csv, fig13_peak_trajectories.png, "
          f"fig14_interval_staircase.png, fig14_interval_staircase_igor.png")


if __name__ == "__main__":
    main()
