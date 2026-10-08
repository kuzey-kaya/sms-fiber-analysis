#!/usr/bin/env python3
"""Check the phase-unwrapping readout against SIMULATED runs with a known answer.

The simulated spectra use the real sensor's phase profile (sms_analysis.synthetic)
and change it in controlled ways: a common phase shift (invisible to a phase
difference) and a tilt (the change of φ(λb) − φ(λa) itself).  Nothing here is
measured data.

Produces:
  fig23_phase_validation.png  - SIMULATED: readout vs the true phase difference for
                                four scenarios, and how many fringes the peak
                                tracker keeps over the same runs
  phase_validation.csv        - the numbers behind it

Usage (needs no data file):
       python scripts/validate_phase.py --outdir figures
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
from sms_analysis import track_all_extrema  # noqa: E402
from sms_analysis.phase import phase_series  # noqa: E402
from sms_analysis.plotting import COLOR_FIT, COLOR_TEMPERATURE, COLOR_WAVELENGTH  # noqa: E402
from sms_analysis.synthetic import simulate_run  # noqa: E402

SCENARIOS = [
    ("A: common 3π only (cooldown-like)", 3 * np.pi, 0.0),
    ("B: common 10π + tilt 7 rad (strain-like, wide range)", 10 * np.pi, 7.0),
    ("C: tilt 7 rad only", 0.0, 7.0),
    ("D: common −6π + tilt −3 rad", -6 * np.pi, -3.0),
]


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--frames", type=int, default=300)
    a = p.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.2 * len(SCENARIOS), 4.2), sharey=False)
    for ax, (name, common, tilt) in zip(axes, SCENARIOS):
        sim, truth = simulate_run(n_frames=a.frames, common_rad=common, tilt_rad=tilt)
        s = phase_series(sim)
        got = s["phase_diff_signed"].to_numpy()
        err = got - truth
        err = err - np.nanmedian(err)            # the absolute offset carries no information
        tracks = track_all_extrema(sim)
        kept = sum(v["wavelength"].notna().mean() >= 0.9 for v in tracks.values())
        rows.append(dict(scenario=name, common_rad=common, tilt_rad=tilt,
                         readout_change=float(got[-1] - got[0]), true_change=float(truth[-1] - truth[0]),
                         max_abs_error_rad=float(np.nanmax(np.abs(err))),
                         largest_frame_jump_rad=float(np.nanmax(np.abs(np.diff(got)))),
                         failed_fits=int(s["phase_diff"].isna().sum()),
                         fringes=len(tracks), fringes_kept_by_tracker=int(kept)))
        f = sim.temperature
        ax.plot(f, truth - truth[0], color=COLOR_FIT, lw=2, label="true φ(λb) − φ(λa)")
        ax.plot(f, got - got[0], ".", ms=2.5, color=COLOR_WAVELENGTH, alpha=0.6, label="phase-unwrapping readout")
        ax.set_title(f"{name}\nmax error {rows[-1]['max_abs_error_rad']:.2f} rad · tracker keeps "
                     f"{kept}/{len(tracks)} fringes", fontsize=9)
        ax.set_xlabel("Fraction of the simulated run")
        ax.legend(frameon=False, fontsize=8, loc="upper left" if tilt >= 0 else "lower left")
    axes[0].set_ylabel("Change of the readout (rad)")
    fig.suptitle("SIMULATED spectra (real phase profile, known changes) — not measured data",
                 color=COLOR_TEMPERATURE, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "fig23_phase_validation.png", dpi=130)
    plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "phase_validation.csv", index=False)
    print(table[["scenario", "readout_change", "true_change", "max_abs_error_rad", "failed_fits",
                 "fringes_kept_by_tracker", "fringes"]].round(3).to_string(index=False))
    print(f"Written to {outdir}/: fig23_phase_validation.png, phase_validation.csv")


if __name__ == "__main__":
    main()
