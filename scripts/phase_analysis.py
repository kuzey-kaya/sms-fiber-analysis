#!/usr/bin/env python3
"""Phase-unwrapping readout (Salik et al., Optica Sensing 2025, SM1E.2) on a run.

Produces:
  fig21_phase_profile.png   - first spectrum with the extrema numbered λ±n around
                              λc, and their phases with the cubic Δφ(λ) fit
                              (the counterpart of the paper's Fig. 1)
  fig22_phase_readout.png   - |φ(λb) − φ(λa)| vs time with the condition (Igor
                              colours), the readout vs the condition with a
                              linear fit, and λc vs the condition
                              (the counterpart of the paper's Fig. 2)
  phase_series.csv          - per spectrum: λc, φ(λa), φ(λb), readout, fit RMS
  phase_pair_scan.csv       - sensitivity of every wavelength pair on a 5 nm grid
  phase_vs_tracking.csv     - phase change implied by peak tracking, per fringe

Usage:
  python scripts/phase_analysis.py                          # data/data.csv, 1545/1570 nm as in the paper
  python scripts/phase_analysis.py other.csv --lambda-a 1485 --lambda-b 1565 --outdir figures
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
from sms_analysis import load_sms_csv, track_all_extrema  # noqa: E402
from sms_analysis.igor_style import IGOR_BLUE, IGOR_RED  # noqa: E402
from sms_analysis.phase import (PAPER_LAMBDA_A, PAPER_LAMBDA_B, phase_fits,  # noqa: E402
                                phase_sensitivity, phase_series, scan_pairs, tracked_phase_change)
from sms_analysis.plotting import COLOR_FIT, COLOR_TEMPERATURE, COLOR_WAVELENGTH  # noqa: E402


def profile_figure(data, fit, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(data.wavelengths, data.spectra[0], color=COLOR_WAVELENGTH, lw=1.2)
    y0 = np.interp(fit.wavelengths, data.wavelengths, data.spectra[0])
    for w, y, k, lab in zip(fit.wavelengths, y0, fit.kinds, fit.labels()):
        ax1.plot(w, y, "o" if k == "peak" else "v", color=COLOR_TEMPERATURE, ms=5)
        n = 0 if lab == "λc" else int(lab[2:])
        if n <= 3:   # label the extrema around λc, as in the paper's figure
            ax1.annotate(lab, (w, y), textcoords="offset points",
                         xytext=(0, 7 if k == "peak" else -13), ha="center", fontsize=8)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Transmittance")
    ax1.set_title("(a) First spectrum, extrema numbered outward from λc")

    ok = np.isfinite(fit.phases)
    xs = np.linspace(data.wavelengths.min(), data.wavelengths.max(), 400)
    ax2.plot(fit.wavelengths[ok], fit.phases[ok], "o", mfc="none", color=COLOR_WAVELENGTH, ms=5,
             label="Δφ(λ±n) = −(n−1)π")
    ax2.plot(xs, fit.phase(xs), color=COLOR_FIT, lw=1.4,
             label=f"cubic fit, RMS {fit.rms:.3f} rad")
    ax2.axvline(fit.lambda_c, color="#999999", ls="--", lw=1)
    ax2.annotate(f"λc = {fit.lambda_c:.1f} nm", (fit.lambda_c, fit.phase(fit.lambda_c)),
                 textcoords="offset points", xytext=(8, -18), fontsize=9)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Δφ (radians)")
    ax2.set_title("(b) Phase from the extrema and cubic Δφ(λ)")
    ax2.legend(frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def readout_figure(series, sens, la, lb, has_temp, out):
    ncols = 3 if has_temp else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.4), squeeze=False)
    ax = axes[0, 0]
    ax.plot(series["time"], series["phase_diff"], color=IGOR_RED, lw=1.2)
    ax.set_xlabel("Time, seconds")
    ax.set_ylabel(f"|φ({lb:g} nm) − φ({la:g} nm)| (radians)", color=IGOR_RED)
    ax.set_title("(a) Phase-difference readout")
    if has_temp:
        ax2 = ax.twinx()
        ax2.plot(series["time"], series["temperature"], color=IGOR_BLUE, lw=1.2)
        ax2.set_ylabel("Temperature, °C", color=IGOR_BLUE, rotation=270, labelpad=14)
        ax2.grid(False)

        ax = axes[0, 1]
        ok = series.dropna(subset=["phase_diff", "temperature"])
        ax.plot(ok["temperature"], ok["phase_diff"], ".", ms=3, color=COLOR_WAVELENGTH, alpha=0.5)
        t = np.linspace(ok["temperature"].min(), ok["temperature"].max(), 50)
        ax.plot(t, sens.slope * t + sens.intercept, color=COLOR_FIT, lw=1.5,
                label=f"slope {sens.slope * 1e3:+.1f} mrad/°C, R² = {sens.r_squared:.4f}")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Readout (radians)")
        ax.set_title(f"(b) Readout vs temperature — span {sens.span_rad:.2f} rad")
        ax.legend(frameon=False, fontsize=8.5)

        ax = axes[0, 2]
        okc = series.dropna(subset=["lambda_c", "temperature"])
        k = np.polyfit(okc["temperature"], okc["lambda_c"], 1)
        ax.plot(okc["temperature"], okc["lambda_c"], ".", ms=3, color=COLOR_TEMPERATURE, alpha=0.5)
        ax.plot(t, np.polyval(k, t), color=COLOR_FIT, lw=1.5, label=f"{k[0] * 1e3:+.1f} pm/°C")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("λc from the cubic (nm)")
        ax.set_title("(c) Critical wavelength")
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="data/data.csv", help="spectra CSV (default: data/data.csv)")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--lambda-a", type=float, default=PAPER_LAMBDA_A, help="first wavelength (nm), paper: 1545")
    p.add_argument("--lambda-b", type=float, default=PAPER_LAMBDA_B, help="second wavelength (nm), paper: 1570")
    p.add_argument("--prominence", type=float, default=0.02)
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_sms_csv(a.csv)
    print(data)

    fits = phase_fits(data, prominence=a.prominence)
    fit0 = next((f for f in fits if f is not None), None)
    if fit0 is None:
        sys.exit("No spectrum has at least two extrema on each side of a critical wavelength; "
                 "the phase-unwrapping readout needs a λc inside the measured band.")
    n_ok = sum(f is not None for f in fits)
    print(f"cubic phase fits: {n_ok}/{len(fits)} spectra; first spectrum λc = {fit0.lambda_c:.2f} nm, "
          f"RMS {fit0.rms:.3f} rad, {len(fit0.wavelengths)} extrema")
    profile_figure(data, fit0, outdir / "fig21_phase_profile.png")

    series = phase_series(data, a.lambda_a, a.lambda_b, fits=fits)
    series.to_csv(outdir / "phase_series.csv", index=False)
    sens = None
    if data.has_temperature:
        sens = phase_sensitivity(series)
        print(f"readout |φ({a.lambda_b:g}) − φ({a.lambda_a:g})|: {sens.slope * 1e3:+.1f} ± "
              f"{sens.stderr * 1e3:.1f} mrad/°C, R² = {sens.r_squared:.4f}, span {sens.span_rad:.2f} rad "
              f"over the run ({sens.n_points} frames)")
        grid = np.arange(np.ceil(data.wavelengths.min() / 5) * 5 + 5, data.wavelengths.max() - 4, 5.0)
        scan = scan_pairs(fits, data.temperature, grid)
        scan.to_csv(outdir / "phase_pair_scan.csv", index=False)
        best = scan.sort_values("resolution").iloc[0]
        print(f"best pair on a 5 nm grid: {best.lambda_a:g}–{best.lambda_b:g} nm, "
              f"{abs(best.slope) * 1e3:.1f} mrad/°C, R² {best.r_squared:.4f}, "
              f"resolution ≈ {best.resolution:.2f} °C")
        common = tracked_phase_change(fit0, track_all_extrema(data, prominence=a.prominence))
        common.to_csv(outdir / "phase_vs_tracking.csv", index=False)
        print(f"phase change implied by peak tracking: {common.phase_change_rad.mean():+.2f} rad on average "
              f"(= {common.phase_change_rad.mean() / np.pi:+.2f}π, common to all fringes), "
              f"spread {common.phase_change_rad.min():.2f}…{common.phase_change_rad.max():.2f} rad "
              f"(the part a phase difference can see)")
    else:
        print("no temperature column: readout vs time only")
    readout_figure(series, sens, a.lambda_a, a.lambda_b, data.has_temperature, outdir / "fig22_phase_readout.png")
    written = ["fig21_phase_profile.png", "fig22_phase_readout.png", "phase_series.csv"]
    if data.has_temperature:
        written += ["phase_pair_scan.csv", "phase_vs_tracking.csv"]
    print(f"Written to {outdir}/: " + ", ".join(written))


if __name__ == "__main__":
    main()
