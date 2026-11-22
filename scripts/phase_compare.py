"""Compare phase readouts: several wavelength pairs on one run, the cubic against the local
fit, and what the curve-fit settings change (the Phase tab's comparison, from the command line).

Outputs (in --outdir, default figures/):
  fig24_phase_pairs.png        - every pair against the condition: change from the first spectrum
                                 (top) and deviation from its own straight line (bottom), cubic
                                 alone (open) and with the local fit (filled)
  fig25_phase_fit_effect.png   - scatter about the line and resolution per pair, cubic vs local fit
  phase_pairs.csv              - the numbers: slope, R², span, scatter, bow, noise, resolution, gain
  phase_pairs_readouts.csv     - per spectrum: time, condition and both readouts of every pair
with --study:
  fig26_phase_settings.png     - one fit setting changed at a time: slope and scatter against the
                                 published settings, per pair
  phase_settings_study.csv     - every pair, method and setting
  phase_settings_spread.csv    - how far each pair's slope moves across the settings that still work

Usage:
  python scripts/phase_compare.py                              # data/data.csv, automatic pairs
  python scripts/phase_compare.py --pairs 1545/1570,1485/1565  # chosen pairs
  python scripts/phase_compare.py data/strain.csv --study      # a load-step run, with the settings study
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
from sms_analysis import load_sms_csv  # noqa: E402
from sms_analysis.auto import choose  # noqa: E402
from sms_analysis.phase import PAPER_LAMBDA_A, PAPER_LAMBDA_B, phase_fits, scan_pairs  # noqa: E402
from sms_analysis.phase_compare import (compare_pairs, default_pairs, deviation, parameter_study,  # noqa: E402
                                        plateau_groups, study_spread, study_summary)

COLORS = ["#2E5FA3", "#C4402F", "#4B9B78", "#E69F00", "#7B5AA6", "#1B9E9E"]


def pairs_figure(readouts, cond_label, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)
    for i, (label, r) in enumerate(readouts.items()):
        color, x = COLORS[i % len(COLORS)], r["condition"].to_numpy(float)
        for col, kw in (("cubic", dict(mfc="none", alpha=0.25, ms=3)), ("local", dict(alpha=0.7, ms=2.5))):
            y = r[col].to_numpy(float)
            ok = np.isfinite(y)
            change = y - y[ok][0] if ok.any() else y
            name = f"{label} nm, {'cubic only' if col == 'cubic' else 'local fit'}"
            axes[0].plot(x, change, "o", color=color, label=name, **kw)
            axes[1].plot(x, deviation(y, x) * 1e3, "o", color=color, **kw)
    axes[0].set_ylabel("Change from the first spectrum (rad)")
    axes[0].set_title("Phase readout of several wavelength pairs (open = cubic only, filled = with the local fit)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].axhline(0, color="#888888", lw=0.8)
    axes[1].set_ylabel("Deviation from its straight line (mrad)")
    axes[1].set_xlabel(cond_label)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def effect_figure(table, unit, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    pairs = list(dict.fromkeys(table["pair"]))
    x = np.arange(len(pairs))
    for k, (method, color) in enumerate((("cubic only", "#9a9a9a"), ("local fit", "#ff0000"))):
        part = table[table["method"] == method].set_index("pair").reindex(pairs)
        bars = axes[0].bar(x + (k - 0.5) * 0.38, part["scatter"] * 1e3, 0.38, color=color, label=method)
        axes[1].bar(x + (k - 0.5) * 0.38, part["resolution"], 0.38, color=color)
        if method == "local fit":
            for b, g, r in zip(bars, part["gain"], part["refined_pct"]):
                axes[0].annotate(f"{g:.1f}×" if r > 0 else "not refined", (b.get_x() + b.get_width() / 2, b.get_height()),
                                 textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    for ax, title in zip(axes, ("Scatter about the straight line (mrad)", f"Resolution ({unit})")):
        ax.set_xticks(x, pairs)
        ax.set_xlabel("Wavelength pair (nm)")
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def study_figure(study, out):
    loc = study[study["method"] == "local fit"]
    order = list(dict.fromkeys(loc["variant"]))
    ypos = {v: len(order) - 1 - i for i, v in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.34 * len(order) + 2), sharey=True)
    for i, (pair, part) in enumerate(loc.groupby("pair", sort=False)):
        color = COLORS[i % len(COLORS)]
        y = [ypos[v] for v in part["variant"]]
        ok = (part["refined_pct"] >= 50).to_numpy()
        for ax, col, clip in ((axes[0], "slope_change_pct", (-60, 60)), (axes[1], "scatter_ratio", (0.2, 30))):
            v = part[col].clip(*clip).to_numpy(float)
            ax.plot(v[ok], np.array(y)[ok], "o", color=color, ms=7, label=f"{pair} nm" if ax is axes[0] else None)
            ax.plot(v[~ok], np.array(y)[~ok], "x", color=color, ms=7)
    axes[0].axvline(0, color="#888888", lw=0.8)
    axes[1].axvline(1, color="#888888", lw=0.8)
    axes[1].set_xscale("log")
    axes[0].set_yticks(list(ypos.values()), list(ypos.keys()), fontsize=8.5)
    axes[0].set_xlabel("slope against the published settings (%), clipped at ±60")
    axes[1].set_xlabel("scatter against the published settings (×, log), clipped at 0.2 … 30")
    axes[0].set_title("What each fit setting does to the slope")
    axes[1].set_title("… and to the scatter (× = local fit mostly refused)")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#dddddd", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="data/data.csv", help="spectra CSV (default: data/data.csv)")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--pairs", default=None, help='pairs in nm, e.g. "1545/1570,1485/1565" (default: automatic)')
    p.add_argument("--prominence", type=float, default=0.02)
    p.add_argument("--study", action="store_true", help="also run the fit-settings study (about a minute)")
    p.add_argument("--study-frames", type=int, default=150, help="spectra used by the study, spread over the run")
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_sms_csv(a.csv)
    print(data)
    fits = phase_fits(data, prominence=a.prominence)
    fit0 = next((f for f in fits if f is not None), None)
    if fit0 is None:
        sys.exit("No spectrum has two extrema on each side of a critical wavelength: nothing to compare.")

    # the condition: the file's column, or the load steps when the run is a load-step run (as in the app)
    cond, groups, unit, cond_label = np.asarray(data.temperature, float), None, "°C", "Temperature (°C)"
    choice = choose(data, prominence=a.prominence)
    if choice.is_steps:
        cond, groups = choice.step_cond, plateau_groups(len(data), choice.plateaus)
        unit, cond_label = "g", "Mass (g, 50 g per step assumed)"
        print(f"load-step run: {len(choice.plateaus)} plateaus, condition = the load steps")
    if not np.isfinite(cond).any():
        sys.exit("The file has no condition column and no load steps: slopes cannot be compared.")

    wl_min, wl_max = float(data.wavelengths.min()), float(data.wavelengths.max())
    if a.pairs:
        pairs = [tuple(sorted(float(v) for v in item.split("/"))) for item in a.pairs.split(",")]
    else:
        grid = np.arange(np.ceil(wl_min / 5) * 5 + 5, wl_max - 4, 5.0)
        pairs = default_pairs(PAPER_LAMBDA_A, PAPER_LAMBDA_B, fit0.lambda_c, wl_min, wl_max,
                              scan_pairs(fits, cond, grid))
    print("pairs (nm):", ", ".join(f"{x:g}/{y:g}" for x, y in pairs))

    readouts, table = compare_pairs(data, fits, pairs, cond, groups)
    table.to_csv(outdir / "phase_pairs.csv", index=False)
    wide = pd.DataFrame({"time": data.time, "condition": cond})
    for label, r in readouts.items():
        wide[f"{label} cubic"], wide[f"{label} local"], wide[f"{label} local_ok"] = r["cubic"], r["local"], r["local_ok"]
    wide.to_csv(outdir / "phase_pairs_readouts.csv", index=False)
    pairs_figure(readouts, cond_label, outdir / "fig24_phase_pairs.png")
    effect_figure(table, unit, outdir / "fig25_phase_fit_effect.png")
    show = table[["pair", "method", "slope", "r_squared", "span", "scatter", "bow_pct", "noise", "resolution",
                  "refined_pct", "gain"]].copy()
    for c in ("slope", "scatter", "noise"):
        show[c] *= 1e3
    print(show.rename(columns={"slope": f"slope (mrad/{unit})", "scatter": "scatter (mrad)", "noise": "noise (mrad)",
                               "span": "span (rad)", "resolution": f"resolution ({unit})"}).round(3).to_string(index=False))
    written = ["fig24_phase_pairs.png", "fig25_phase_fit_effect.png", "phase_pairs.csv", "phase_pairs_readouts.csv"]

    if a.study:
        rows = np.unique(np.linspace(0, len(data) - 1, min(len(data), a.study_frames)).astype(int))
        study = parameter_study(data, pairs[:4], cond, groups, rows=rows, prominence=a.prominence,
                                progress=lambda i, n, name: print(f"  [{i + 1}/{n}] {name}") if name else None)
        study.to_csv(outdir / "phase_settings_study.csv", index=False)
        spread = study_spread(study)
        spread.to_csv(outdir / "phase_settings_spread.csv", index=False)
        study_figure(study, outdir / "fig26_phase_settings.png")
        print(study_summary(study).round(2).to_string(index=False))
        if len(spread):
            s = spread.copy()
            for c in ("slope", "slope_min", "slope_max"):
                s[c] *= 1e3
            print("\nslope across the settings that still work (mrad per unit):")
            print(s.round(3).to_string(index=False))
        written += ["fig26_phase_settings.png", "phase_settings_study.csv", "phase_settings_spread.csv"]
    print("wrote:", ", ".join(written), "->", outdir)


if __name__ == "__main__":
    main()
