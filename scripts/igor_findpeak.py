"""Igor Pro's FindPeak, run the group's way: one fixed range on every spectrum → Igor-style graph.

    python scripts/igor_findpeak.py data/strain.csv                      # the app's automatic fringe, auto box
    python scripts/igor_findpeak.py data/strain.csv --fringe dip@1574.0   # one choice, like the tab
    python scripts/igor_findpeak.py data/strain.csv --points 735 766 --min
    python scripts/igor_findpeak.py data/data.csv --range 1490 1497 --max --box 11

Writes <outdir>/findpeak_<a>-<b>.csv (time, temperature, wavelength = V_PeakLoc, value, width,
found) and <outdir>/findpeak_<a>-<b>.png (red V_PeakLoc on the left axis, blue condition on the
right, time in seconds — the Igor graph).  Uses only the samples inside the range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sms_analysis import fit_sensitivity, load_sms_csv  # noqa: E402
from sms_analysis.auto import choose  # noqa: E402
from sms_analysis.findpeak import points_to_nm, track_findpeak  # noqa: E402
from sms_analysis.igor_style import plot_igor_tracking  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="data/data.csv")
    p.add_argument("--outdir", default="figures")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--range", nargs=2, type=float, metavar=("START_NM", "END_NM"),
                   help="Igor /R=(startX,endX); start > end searches backwards")
    g.add_argument("--points", nargs=2, type=int, metavar=("START_P", "END_P"),
                   help="Igor /R=[startP,endP], sample numbers from 0 (the advisor's calls: 735 766, 274 353)")
    k = p.add_mutually_exclusive_group()
    k.add_argument("--min", action="store_true", help="look for a minimum (Igor /N) — the default")
    k.add_argument("--max", action="store_true", help="look for a maximum")
    g.add_argument("--fringe", metavar="LABEL", help="a fringe of the first spectrum, e.g. dip@1574.0: range = its whole "
                                                     "excursion +- 1.5 nm, kind from the label (the app's one choice)")
    p.add_argument("--box", type=int, default=None, help="box-smoothing width in samples (Igor /B); default = a quarter "
                                                           "of the fringe's half period (5..41, odd) when the range comes "
                                                           "from a fringe, else 5")
    p.add_argument("--min-level", type=float, default=None, help="Igor /M=minLevel")
    a = p.parse_args()

    data = load_sms_csv(a.csv)
    box = a.box
    negative = not a.max
    auto = None
    if a.points or a.range:
        start, end = (points_to_nm(data, *a.points) if a.points else tuple(a.range))
        box = 5 if box is None else box
    else:
        # no range given: the fringe named with --fringe, or the app's automatic choice
        auto = choose(data)
        label = a.fringe or auto.fringe
        if label not in auto.tracks:
            sys.exit(f"{label} is not a fringe of the first spectrum; choose one of: " + ", ".join(auto.tracks))
        w = auto.tracks[label]["wavelength"].to_numpy(float)
        start, end = float(np.nanmin(w) - 1.5), float(np.nanmax(w) + 1.5)
        negative = label.startswith("dip") if not (a.min or a.max) else not a.max
        pos = np.sort([float(k.split("@")[1]) for k in auto.tracks])
        i = int(np.argmin(np.abs(pos - float(label.split("@")[1]))))
        half = min(abs(pos[i] - pos[j]) for j in (i - 1, i + 1) if 0 <= j < pos.size)
        step_nm = float(np.median(np.diff(data.wavelengths)))
        if box is None:
            box = int(np.clip(int(0.25 * half / step_nm) | 1, 5, 41))
        print(f"{label}: range {start:.1f}-{end:.1f} nm (excursion +- 1.5 nm), box {box} = 1/4 of its half period "
              f"({half:.1f} nm), {'minimum' if negative else 'maximum'}")
    a.box = box
    a.max = not negative
    tr = track_findpeak(data, start, end, box=box, negative=negative, min_level=a.min_level)
    w = tr["wavelength"].to_numpy(float)
    print(data)
    width_txt = (f"width {np.nanmedian(tr['width']):.2f} nm (median)" if np.isfinite(tr["width"]).any()
                 else "width n/a (the peak's edges lie outside the range)")
    print(f"FindPeak /R=({start:.1f},{end:.1f}) /B={a.box} {'/N' if not a.max else ''}"
          f"{f' /M={a.min_level}' if a.min_level is not None else ''}: "
          f"found in {100 * tr['found'].mean():.1f} % of the spectra, "
          f"V_PeakLoc {np.nanmin(w):.3f}–{np.nanmax(w):.3f} nm, {width_txt}")
    # a load-step run gets the app's step condition (mass per plateau) instead of the temperature column
    auto_steps = auto if auto is not None else choose(data)
    if auto_steps.is_steps:
        tr = tr.assign(temperature=auto_steps.step_cond)
        pl = auto_steps.plateaus
        parts = []
        for _, p_ in pl.iterrows():
            seg = w[int(p_["start"]) + 2:int(p_["end"]) - 1]
            seg = seg[np.isfinite(seg)]
            if seg.size > 5:
                parts.append(seg - seg.mean())
        noise = np.std(np.concatenate(parts)) * 1e3 if parts else float("nan")
        print(f"load-step run: {len(pl)} plateaus, {int((auto_steps.step_values > 0).sum())} with a load "
              f"(50 g steps assumed - edit the masses in the app); noise on the plateaus {noise:.0f} pm")
        if np.isfinite(tr["temperature"]).sum() >= 3:
            f = fit_sensitivity(tr)
            print(f"vs mass: {f.slope * 1e3:+.2f} ± {f.stderr * 1e3:.2f} pm/g, R² {f.r_squared:.4f}")
    elif data.has_temperature and np.isfinite(w).sum() >= 3:
        f = fit_sensitivity(tr)
        print(f"vs the condition column: {f.slope * 1e3:+.1f} ± {f.stderr * 1e3:.1f} pm per unit, R² {f.r_squared:.4f}")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"findpeak_{start:.0f}-{end:.0f}"
    tr.to_csv(outdir / f"{stem}.csv", index=False)
    plot_igor_tracking(tr)
    plt.gcf().suptitle(f"FindPeak /R=({start:.1f},{end:.1f}) /B={a.box} {'/N' if not a.max else ''} — {Path(a.csv).name}",
                       fontsize=9)
    plt.savefig(outdir / f"{stem}.png", dpi=130)
    print(f"Written to {outdir}/: {stem}.csv, {stem}.png")


if __name__ == "__main__":
    main()
