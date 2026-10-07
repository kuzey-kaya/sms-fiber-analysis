#!/usr/bin/env python3
"""Step-like views of the continuous cooldown (figs 9-12), all in the Igor style.

None of these steps is a physical jump of the sensor: fig9 is binning of the
smooth drift, figs 10-12 are fringe-hops of the tracker (step height = fringe
spacing).

Produces:
  fig9_staircase_3C.png     - dip starting at 1542.3 nm, averaged in 3 °C
                              temperature bins and drawn as plateaus
  staircase_plateaus.csv    - the plateau table behind fig9 (Table 2 of the report)
  fig10_legacy_igor.png     - original Colab tracker (track_feature_legacy,
                              default parameters): one fringe-hop at t ≈ 7660 s
  fig11_multistep_igor.png  - legacy tracker with an 8 nm window + median
                              filter 15: several fringe-hop steps
  fig12_desc_med7.png       - strongest peak inside 1475-1535 nm
                              (track_band_extremum, median filter 7)

Usage: python scripts/staircase_views.py [csv] --outdir figures     (default csv: data/data.csv)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sms_analysis import load_sms_csv, track_feature  # noqa: E402
from sms_analysis.igor_style import plot_igor_tracking  # noqa: E402
from sms_analysis.staircase import plot_staircase, staircase_bin  # noqa: E402
from sms_analysis.tracking import track_band_extremum, track_feature_legacy  # noqa: E402

START_NM = 1542.3
STEP_C = 3.0
BAND = (1475.0, 1535.0)


def _save(fig, out):
    fig.savefig(out, dpi=130)
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
    print(data)
    written = []

    # fig9: temperature-bin plateaus of the cleanly tracked dip ---------------
    if data.has_temperature:
        track = track_feature(data, start_wavelength=START_NM, kind="dip")
        steps = staircase_bin(track, step_c=STEP_C)
        fig, ax, _ = plot_staircase(track, steps, wl_ticks=2, temp_ticks=5, time_ticks=5000)
        ax.set_title(f"Step-averaged tracking (real data, {STEP_C:g} °C bins, "
                     f"dip@{START_NM:.0f})", fontsize=10)
        _save(fig, outdir / "fig9_staircase_3C.png")
        steps.round(3).to_csv(outdir / "staircase_plateaus.csv", index=False)
        written += ["fig9_staircase_3C.png", "staircase_plateaus.csv"]
    else:
        print("no temperature column -> fig9 (temperature-bin staircase) skipped")

    # fig10: exact reproduction of the old Colab tracking ---------------------
    legacy = track_feature_legacy(data)
    fig, _, _ = plot_igor_tracking(legacy, wl_ticks=5, temp_ticks=5, time_ticks=5000)
    _save(fig, outdir / "fig10_legacy_igor.png")
    written.append("fig10_legacy_igor.png")

    # fig11: wider window + median filter -> several fringe-hop steps ---------
    multistep = track_feature_legacy(data, search_window=8, median_kernel=15)
    fig, _, _ = plot_igor_tracking(multistep, wl_ticks=5, temp_ticks=5, time_ticks=5000)
    _save(fig, outdir / "fig11_multistep_igor.png")
    written.append("fig11_multistep_igor.png")

    # fig12: strongest peak in a fixed band -> identity hops ------------------
    band = track_band_extremum(data, band=BAND, mode="peak", median_kernel=7)
    fig, _, _ = plot_igor_tracking(band, wl_ticks=10, temp_ticks=5, time_ticks=5000)
    _save(fig, outdir / "fig12_desc_med7.png")
    written.append("fig12_desc_med7.png")

    print(f"Written to {outdir}/: " + ", ".join(written))


if __name__ == "__main__":
    main()
