"""Igor Pro's FindPeak, run the group's way: one fixed range on every spectrum → Igor-style graph.

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
from sms_analysis.findpeak import points_to_nm, track_findpeak  # noqa: E402
from sms_analysis.igor_style import plot_igor_tracking  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="data/data.csv")
    p.add_argument("--outdir", default="figures")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--range", nargs=2, type=float, metavar=("START_NM", "END_NM"),
                   help="Igor /R=(startX,endX); start > end searches backwards")
    g.add_argument("--points", nargs=2, type=int, metavar=("START_P", "END_P"),
                   help="Igor /R=[startP,endP], sample numbers from 0 (the advisor's calls: 735 766, 274 353)")
    k = p.add_mutually_exclusive_group()
    k.add_argument("--min", action="store_true", help="look for a minimum (Igor /N) — the default")
    k.add_argument("--max", action="store_true", help="look for a maximum")
    p.add_argument("--box", type=int, default=5, help="box-smoothing width in samples (Igor /B, default 5)")
    p.add_argument("--min-level", type=float, default=None, help="Igor /M=minLevel")
    a = p.parse_args()

    data = load_sms_csv(a.csv)
    start, end = (points_to_nm(data, *a.points) if a.points else tuple(a.range))
    tr = track_findpeak(data, start, end, box=a.box, negative=not a.max, min_level=a.min_level)
    w = tr["wavelength"].to_numpy(float)
    print(data)
    print(f"FindPeak /R=({start:.1f},{end:.1f}) /B={a.box} {'/N' if not a.max else ''}"
          f"{f' /M={a.min_level}' if a.min_level is not None else ''}: "
          f"found in {100 * tr['found'].mean():.1f} % of the spectra, "
          f"V_PeakLoc {np.nanmin(w):.3f}–{np.nanmax(w):.3f} nm, width {np.nanmedian(tr['width']):.2f} nm (median)")
    if data.has_temperature and np.isfinite(w).sum() >= 3:
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
