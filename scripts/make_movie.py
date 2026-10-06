#!/usr/bin/env python3
"""Animate the run as a movie: spectrum + extrema frame by frame, the spectral
map with a time cursor, and the most sensitive tracked fringe growing in time
(the Python version of Igor Pro's movie export).

Usage:
    python scripts/make_movie.py                       # data/data.csv -> figures/movie_run.gif
    python scripts/make_movie.py other.csv --format mp4 --every 3 --fps 15 --track 1542.3

MP4 needs ffmpeg on the PATH; otherwise a GIF is written (Pillow).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sms_analysis import load_sms_csv, sensitivity_map, track_all_extrema, track_feature  # noqa: E402
from sms_analysis.movie import render_movie  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default="data/data.csv", help="spectra CSV (default: data/data.csv)")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--format", choices=["gif", "mp4"], default="gif")
    p.add_argument("--every", type=int, default=5, help="render every N-th spectrum (default 5)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--track", type=float, default=None,
                   help="wavelength (nm) of the fringe for the lower-right panel; "
                        "default: the most sensitive reliably tracked fringe")
    p.add_argument("--track-kind", choices=["peak", "dip"], default="dip")
    p.add_argument("--no-track", action="store_true", help="leave the fringe panel out")
    a = p.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_sms_csv(a.csv)
    print(data)

    track, label = None, ""
    if not a.no_track:
        if a.track is not None:
            track = track_feature(data, start_wavelength=a.track, kind=a.track_kind)
            label = f"{a.track_kind}@{a.track:.1f}"
        else:
            tracks = track_all_extrema(data)
            if data.has_temperature:
                smap = sensitivity_map(tracks, min_r2=0.9)
                smap = smap[smap["n_points"] >= 0.9 * len(data)]
                if len(smap):
                    label = smap.loc[smap["sensitivity_pm_per_C"].abs().idxmax(), "label"]
            if not label:
                label = max(tracks, key=lambda k: tracks[k]["wavelength"].notna().sum())
            track = tracks[label]
        print(f"fringe panel: {label}")

    def progress(i, n):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"  frame {i}/{n}", flush=True)

    out = render_movie(data, outdir / f"movie_run.{a.format}", track=track, track_label=label,
                       every=a.every, fps=a.fps, progress=progress)
    print(f"written: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
