"""Plateaus of a stepped run (load put on and taken off) and a condition built from them.

A strain / load experiment changes the condition in steps: the tracked
wavelength sits on a plateau, jumps within one or two spectra, sits on the
next plateau.  The acquisition file usually carries no load column, so the
load has to be entered by hand, one value per plateau.  :func:`detect_plateaus`
finds the plateaus from a tracked fringe, :func:`step_condition` turns a
list of per-plateau values into a per-spectrum condition array that the
calibration code accepts in place of the temperature column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import medfilt


def robust_jitter(x: np.ndarray) -> float:
    """Frame-to-frame scatter of a series (nm), from the MAD of its differences; spikes ignored."""
    d = np.diff(x)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return 0.0
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)


def detect_plateaus(
    time: np.ndarray,
    wavelength: np.ndarray,
    min_jump: float | None = None,
    min_points: int = 5,
    median_kernel: int = 5,
) -> pd.DataFrame:
    """Split a tracked wavelength series into plateaus separated by jumps.

    Parameters
    ----------
    time, wavelength : the track (NaN = lost lock; such rows join the plateau around them).
    min_jump : a change between consecutive (median-filtered) samples larger than this
        (nm) starts a new plateau.  ``None`` = max(0.15 nm, 8 × the frame-to-frame jitter).
    min_points : plateaus shorter than this are merged into their neighbour.
    median_kernel : odd kernel of the median filter that removes single-frame spikes
        before the jumps are looked for.

    Returns
    -------
    DataFrame with one row per plateau: ``plateau`` (0-based), ``start`` / ``end`` (row
    indices, inclusive), ``t_start`` / ``t_end`` (s), ``n``, ``mean`` / ``std`` (nm, over the
    locked rows), ``shift`` (mean − mean of the first plateau).
    """
    w = np.asarray(wavelength, float)
    t = np.asarray(time, float)
    n = w.size
    if n == 0:
        return pd.DataFrame(columns=["plateau", "start", "end", "t_start", "t_end", "n", "mean", "std", "shift"])
    filled = pd.Series(w).ffill().bfill().to_numpy()
    k = max(1, median_kernel | 1)
    smooth = medfilt(filled, k) if n > k else filled
    if min_jump is None:
        min_jump = max(0.15, 8.0 * robust_jitter(w))
    jumps = np.flatnonzero(np.abs(np.diff(smooth)) > min_jump) + 1
    edges = [0, *jumps.tolist(), n]
    segs = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    # merge short segments into the neighbour whose level is closer
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b) in enumerate(segs):
            if b - a >= min_points:
                continue
            level = np.nanmean(smooth[a:b])
            cands = []
            if i > 0:
                cands.append((abs(level - np.nanmean(smooth[segs[i - 1][0]:segs[i - 1][1]])), i - 1))
            if i < len(segs) - 1:
                cands.append((abs(level - np.nanmean(smooth[segs[i + 1][0]:segs[i + 1][1]])), i + 1))
            j = min(cands)[1]
            lo, hi = min(i, j), max(i, j)
            segs[lo:hi + 1] = [(segs[lo][0], segs[hi][1])]
            changed = True
            break
    rows = []
    for i, (a, b) in enumerate(segs):
        seg = w[a:b]
        ok = seg[np.isfinite(seg)]
        rows.append(dict(plateau=i, start=a, end=b - 1, t_start=t[a], t_end=t[b - 1], n=b - a,
                         mean=ok.mean() if ok.size else np.nan, std=ok.std() if ok.size > 1 else np.nan))
    out = pd.DataFrame(rows)
    out["shift"] = out["mean"] - out["mean"].iloc[0]
    return out


def step_condition(n_rows: int, plateaus: pd.DataFrame, values, settle: int = 2) -> np.ndarray:
    """Per-spectrum condition from one value per plateau.

    ``values`` is one number per plateau row (NaN = leave that plateau out).  The
    first and last ``settle`` rows of every plateau are set to NaN so the spectra
    recorded while the load was changing do not enter the calibration.
    """
    cond = np.full(n_rows, np.nan)
    vals = np.asarray(values, float)
    for (_, p), v in zip(plateaus.iterrows(), vals):
        if not np.isfinite(v):
            continue
        a, b = int(p["start"]) + settle, int(p["end"]) + 1 - settle
        if b > a:
            cond[a:b] = v
    return cond


def classify_plateaus(plateaus: pd.DataFrame, tol: float | None = None) -> np.ndarray:
    """True for the plateaus that carry a load, False for the unloaded (baseline) ones.

    The baseline is allowed to drift (temperature changes during the run): a plateau
    counts as baseline when it lies within ``tol`` of the *previous* baseline plateau,
    and that level then becomes the new reference.  ``None`` = max(0.15 nm, a quarter
    of the median jump between consecutive plateaus).
    """
    means = plateaus["mean"].to_numpy(float)
    if means.size == 0:
        return np.zeros(0, bool)
    if tol is None:
        jumps = np.abs(np.diff(means))
        tol = max(0.15, 0.25 * float(np.median(jumps))) if jumps.size else 0.15
    loaded = np.zeros(means.size, bool)
    level = means[0]
    for i in range(1, means.size):
        if abs(means[i] - level) <= tol:
            level = means[i]
        else:
            loaded[i] = True
    return loaded


def default_step_values(plateaus: pd.DataFrame, start: float = 50.0, increment: float = 50.0,
                        baseline: float = 0.0) -> np.ndarray:
    """Guess one value per plateau: baseline plateaus get ``baseline``, the loaded ones
    ``start``, ``start + increment``, … in order of appearance.  A starting point for the
    user to edit, not a measurement."""
    loaded = classify_plateaus(plateaus)
    vals = np.full(len(plateaus), float(baseline))
    vals[loaded] = start + increment * np.arange(loaded.sum())
    return vals


def baseline_drift(n_rows: int, plateaus: pd.DataFrame, loaded: np.ndarray) -> np.ndarray:
    """Per-spectrum baseline wavelength, interpolated in time between the unloaded plateaus.

    Subtracting it from a track removes a slow drift of the unloaded level (e.g. the
    temperature rising during a load experiment) so the plateaus can be compared.
    """
    base = plateaus[~np.asarray(loaded, bool)]
    out = np.full(n_rows, np.nan)
    if base.empty:
        return out
    centre = (base["start"].to_numpy(float) + base["end"].to_numpy(float)) / 2.0
    out[:] = np.interp(np.arange(n_rows), centre, base["mean"].to_numpy(float))
    return out
