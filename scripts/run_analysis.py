#!/usr/bin/env python3
"""Full SMS sensor analysis: load CSV -> detect & track fringes -> temperature
calibration -> figures.

Usage:
    python scripts/run_analysis.py                     # uses data/data.csv
    python scripts/run_analysis.py path/to/other.csv --outdir figures

Outputs (in --outdir, default figures/):
    fig1..fig7                  figures (fig5/fig6/fig7 need a temperature column)
    tracked_all_long.csv        every tracked fringe: fringe, time, temperature, wavelength, value
    tracked_all_wide.csv        every tracked fringe: one wavelength column per fringe
    tracked_feature.csv         the single fringe used for the detailed plots
    sensitivity_map.csv         per-fringe sensitivity table (needs temperature)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sms_analysis import (  # noqa: E402
    fit_sensitivity,
    load_sms_csv,
    sensitivity_map,
    track_all_extrema,
    track_feature,
)
from sms_analysis.plotting import (  # noqa: E402
    plot_correlation,
    plot_sensitivity_map,
    plot_spectra_overlay,
    plot_spectral_map,
    plot_spectrum_with_extrema,
    plot_tracking,
)

DEFAULT_CSV = "data/data.csv"


def export_all_tracks(data, tracks, outdir: Path) -> None:
    """Write every tracked fringe in long and wide layouts."""
    long_df = pd.concat(
        [tr.assign(fringe=label) for label, tr in tracks.items()], ignore_index=True
    )[["fringe", "time", "temperature", "wavelength", "value"]]
    long_df.to_csv(outdir / "tracked_all_long.csv", index=False)

    wide_df = pd.DataFrame({"time": data.time, "temperature": data.temperature})
    for label, tr in tracks.items():
        wide_df[label] = tr["wavelength"].to_numpy()
    wide_df.to_csv(outdir / "tracked_all_wide.csv", index=False)
    print(f"tracked_all_long.csv / tracked_all_wide.csv: "
          f"{len(tracks)} fringes x {len(data)} frames")


def main() -> None:
    # Redirected output on Windows is cp1252 and cannot encode "≥" or "→": replace, never crash.
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=DEFAULT_CSV,
                   help=f"spectra CSV (default: {DEFAULT_CSV})")
    p.add_argument("--outdir", default="figures", help="output directory")
    p.add_argument("--prominence", type=float, default=0.02)
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--search-window", type=float, default=1.5,
                   help="tracking half-window in nm")
    p.add_argument("--track", type=float, default=None,
                   help="wavelength (nm) of the feature to track in detail; "
                        "default: the most sensitive reliably-tracked fringe")
    p.add_argument("--track-kind", choices=["peak", "dip"], default="dip")
    args = p.parse_args()

    if not Path(args.csv).exists():
        sys.exit(f"CSV not found: {args.csv}\n"
                 f"Put your file at {DEFAULT_CSV} or pass its path as the first argument.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_sms_csv(args.csv)
    print(data)
    has_temp = data.has_temperature

    # --- Fig 1: first spectrum with detected extrema -------------------------
    ax = plot_spectrum_with_extrema(data, row=0,
                                    prominence=args.prominence, distance=args.distance)
    ax.set_title("Transmission spectrum with detected extrema (first frame)")
    plt.tight_layout(); plt.savefig(outdir / "fig1_spectrum_peaks.png"); plt.close()

    # --- Fig 2: spectra at several points of the run -------------------------
    rows = [0, len(data) // 4, len(data) // 2, 3 * len(data) // 4, len(data) - 1]
    ax = plot_spectra_overlay(data, rows)
    ax.set_title("Spectra across the run")
    plt.tight_layout(); plt.savefig(outdir / "fig2_spectra_overlay.png"); plt.close()

    # --- Fig 3: the whole run as a spectral map ------------------------------
    ax = plot_spectral_map(data)
    ax.set_title("Spectral map of the full run")
    plt.tight_layout(); plt.savefig(outdir / "fig3_spectral_map.png"); plt.close()

    # --- Track every fringe ---------------------------------------------------
    tracks = track_all_extrema(data, prominence=args.prominence,
                               distance=args.distance, search_window=args.search_window)
    export_all_tracks(data, tracks, outdir)

    good = None
    if has_temp:
        smap = sensitivity_map(tracks, min_r2=0.0)
        smap.to_csv(outdir / "sensitivity_map.csv", index=False)
        good = smap[smap["r_squared"] >= 0.90]
        print(f"\nSensitivity across the spectrum ({len(good)}/{len(smap)} fringes with R² ≥ 0.90):")
        print(good[["label", "start_wavelength", "sensitivity_pm_per_C", "r_squared", "n_points"]]
              .to_string(index=False))

        ax = plot_sensitivity_map(good)
        ax.set_title("Fringe sensitivity vs wavelength")
        plt.tight_layout(); plt.savefig(outdir / "fig6_sensitivity_map.png"); plt.close()
    else:
        print("\nNo temperature column in this file: maps and tracking are produced; "
              "the temperature calibration (fig5, fig6, fig7, sensitivity_map.csv) is skipped.")

    # --- One feature in detail -----------------------------------------------
    if args.track is not None:
        track = track_feature(data, start_wavelength=args.track,
                              kind=args.track_kind, search_window=args.search_window)
        kind = args.track_kind
        label = f"{kind}@{args.track:.1f}"
    elif has_temp and good is not None and len(good):
        # Auto-pick: the most sensitive fringe among the reliably tracked ones.
        candidates = good[good["n_points"] >= 0.9 * len(data)]
        if candidates.empty:
            candidates = good
        best = candidates.loc[candidates["sensitivity_pm_per_C"].abs().idxmax()]
        label, kind = best["label"], best["kind"]
        track = tracks[label]
    else:
        # Auto-pick: the fringe that stays locked longest, then moves the most.
        def score(k):
            w = tracks[k]["wavelength"].dropna()
            return (w.size, (w.max() - w.min()) if w.size else 0.0)
        label = max(tracks, key=score)
        track, kind = tracks[label], label.split("@")[0]
    print(f"\nFeature for the detailed plots: {label}")

    track.to_csv(outdir / "tracked_feature.csv", index=False)
    w0 = track["wavelength"].dropna().iloc[0]
    n_ok = int(track["wavelength"].notna().sum())

    ax = plot_tracking(track, title=f"Tracked {kind} near {w0:.0f} nm")
    plt.tight_layout(); plt.savefig(outdir / "fig4_tracking_vs_time.png"); plt.close()

    if has_temp:
        fit = fit_sensitivity(track)
        print(f"Tracked {kind} starting at {w0:.2f} nm ({n_ok}/{len(track)} rows locked) -> {fit}")

        ax = plot_correlation(track, fit,
                              title=f"Temperature calibration of the {kind} near {w0:.0f} nm")
        plt.tight_layout(); plt.savefig(outdir / "fig5_correlation.png"); plt.close()

        # --- Fig 7: Igor Pro-style rendering of the tracking graph -----------
        from sms_analysis.igor_style import plot_igor_tracking  # noqa: E402
        fig, _, _ = plot_igor_tracking(track, wl_ticks=5, temp_ticks=5, time_ticks=5000)
        fig.savefig(outdir / "fig7_igor_style.png", dpi=130)
        plt.close(fig)
    else:
        print(f"Tracked {kind} starting at {w0:.2f} nm ({n_ok}/{len(track)} rows locked)")

    print(f"\nFigures and CSV outputs written to {outdir}/")


if __name__ == "__main__":
    main()
