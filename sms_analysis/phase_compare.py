"""Comparing phase readouts: several wavelength pairs side by side, the cubic against the local
fit, and what the curve-fit settings change.

Asked for by the advisor through the owner on 2026-11-22: "compare different wavelengths in the
readout-vs-condition graph, layered; one deviates more, one less — what is the effect of the curve
fit; look into the details of the fit algorithm and play with its parameters; try different
wavelength pairs".

Three pieces, all on top of :mod:`sms_analysis.phase`:

* :func:`readout_metrics` — one readout against the condition, in numbers that can be compared
  between pairs whose slopes differ: slope, R², scatter about the straight line, the part of that
  scatter that is a smooth bow (``bow_pct``: the largest gap between the best parabola and the best
  line, in % of the span) and the part that is noise (``noise``: scatter about the parabola),
  frame-to-frame noise, noise on the load plateaus, resolution.
* :func:`compare_pairs` — the readouts of several pairs (each wavelength fitted once, see
  ``phase.wavelength_phase``) and the metrics table, cubic and local fit side by side.
* :func:`parameter_study` — the same pairs with one fit setting changed at a time
  (:data:`VARIANTS`): polynomial degree, inner extrema left out, smoothing, prominence, local
  window width, free phase terms, mode powers fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import SpectraSet
from .phase import (LOCAL_CYCLES, LOCAL_ORDER, PAPER_LAMBDA_A, PAPER_LAMBDA_B, pair_readout, phase_fits,
                    wavelength_phase)

METHODS = {"cubic": "cubic only", "local": "local fit"}


def pair_label(a: float, b: float) -> str:
    return f"{a:g}/{b:g}"


def default_pairs(la: float, lb: float, lam_c: float, wl_min: float, wl_max: float,
                  scan: pd.DataFrame | None = None, limit: int = 5) -> list[tuple[float, float]]:
    """The pairs a comparison starts with: the one in use, the paper's, the mirror image of the
    pair in use on the other side of λc, the pair across λc made of the two outer wavelengths, and
    the pair the cubic scan (``phase.scan_pairs``) ranks best. Pairs outside the band (2 nm
    margin) and repeats are dropped."""
    def r5(x):
        return float(5 * round(x / 5))

    cands = [(float(la), float(lb)), (PAPER_LAMBDA_A, PAPER_LAMBDA_B)]
    if np.isfinite(lam_c):
        ma, mb = r5(2 * lam_c - lb), r5(2 * lam_c - la)
        cands += [(ma, mb), (ma, float(lb))] if la > lam_c else [(ma, mb), (float(la), mb)]
    if scan is not None and len(scan):
        best = scan.sort_values("resolution").iloc[0]
        cands.append((float(best["lambda_a"]), float(best["lambda_b"])))
    out = []
    for a, b in cands:
        a, b = sorted((a, b))
        if wl_min + 2 <= a < b <= wl_max - 2 and (a, b) not in out:
            out.append((a, b))
    return out[:limit]


def plateau_groups(n: int, plateaus: pd.DataFrame | None, settle: int = 2) -> np.ndarray | None:
    """Plateau number per frame (−1 outside, and on the ``settle`` frames at each end)."""
    if plateaus is None or len(plateaus) == 0:
        return None
    g = np.full(n, -1, int)
    for k, (_, pl) in enumerate(plateaus.iterrows()):
        a, b = int(pl["start"]) + settle, int(pl["end"]) + 1 - settle
        if b - a > 5:
            g[a:b] = k
    return g


def readout_metrics(readout, condition, groups=None) -> dict:
    """Numbers that describe one readout against the condition (see the module docstring).

    ``readout`` and ``condition`` are per-frame arrays in time order (NaN = leave out), ``groups``
    an optional plateau number per frame (−1 = none) for the noise where nothing should move.
    Units: rad for the readout columns, the condition's unit for ``resolution``."""
    y, x = np.asarray(readout, float), np.asarray(condition, float)
    out = dict(n=0, slope=np.nan, intercept=np.nan, r_squared=np.nan, span=np.nan, scatter=np.nan,
               max_dev=np.nan, bow=np.nan, bow_pct=np.nan, noise=np.nan, frame_noise=np.nan,
               plateau_noise=np.nan, resolution=np.nan, curvature=np.nan)
    ok = np.isfinite(y) & np.isfinite(x)
    out["n"] = int(ok.sum())
    d = np.diff(y)
    d = d[np.isfinite(d)]
    if d.size > 3:                                   # robust: a load step is an outlier, not noise
        out["frame_noise"] = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))
    if groups is not None:
        parts = []
        gr = np.asarray(groups)
        for k in np.unique(gr[gr >= 0]):
            seg = y[(gr == k) & np.isfinite(y)]
            if seg.size > 5:
                parts.append(seg - seg.mean())
        if parts:
            out["plateau_noise"] = float(np.std(np.concatenate(parts)))
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return out
    xs, ys = x[ok], y[ok]
    k1 = np.polyfit(xs, ys, 1)
    r1 = ys - np.polyval(k1, xs)
    span = abs(k1[0]) * np.ptp(xs)                   # the change the line itself goes through
    out.update(slope=float(k1[0]), intercept=float(k1[1]), scatter=float(np.std(r1)),
               max_dev=float(np.max(np.abs(r1))), span=float(span),
               r_squared=float(1 - np.sum(r1 ** 2) / np.sum((ys - ys.mean()) ** 2)) if np.ptp(ys) else np.nan,
               resolution=float(np.std(r1) / abs(k1[0])) if k1[0] else np.nan)
    if len(np.unique(xs)) >= 3:
        k2 = np.polyfit(xs - xs.mean(), ys, 2)
        grid = np.linspace(xs.min(), xs.max(), 101)
        bow = float(np.max(np.abs(np.polyval(k2, grid - xs.mean()) - np.polyval(k1, grid))))
        out.update(curvature=float(k2[0]), bow=bow, bow_pct=float(100 * bow / span) if span else np.nan,
                   noise=float(np.std(ys - np.polyval(k2, xs - xs.mean()))))
    return out


def deviation(readout, condition) -> np.ndarray:
    """The readout minus its own straight line against the condition (NaN where either is)."""
    y, x = np.asarray(readout, float), np.asarray(condition, float)
    ok = np.isfinite(y) & np.isfinite(x)
    out = np.full(y.shape, np.nan)
    if ok.sum() >= 3 and np.ptp(x[ok]):
        out[ok] = y[ok] - np.polyval(np.polyfit(x[ok], y[ok], 1), x[ok])
    return out


def compare_pairs(
    data: SpectraSet,
    fits,
    pairs,
    condition,
    groups=None,
    phases: dict | None = None,
    rows=None,
    cycles: float = LOCAL_CYCLES,
    order: int = LOCAL_ORDER,
    fit_powers: bool = True,
    smooth_window: int = 11,
    envelope: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Readouts and metrics of several wavelength pairs on the same run.

    ``pairs`` is a list of (λa, λb) in nm; ``condition`` the per-spectrum condition (the step
    values on a load run); ``phases`` an optional {wavelength: wavelength_phase table} to reuse
    (missing wavelengths are fitted here and added to it); ``rows`` restricts the local fits to
    those frames. Returns ({"a/b": table with frame, time, condition, cubic, local, local_ok}, the
    metrics table: one row per pair and method, with ``refined_pct`` and — on the local row —
    ``gain`` = cubic scatter ÷ local scatter)."""
    phases = {} if phases is None else phases
    idx = np.arange(len(data)) if rows is None else np.asarray(rows, dtype=int)
    cond = np.asarray(condition, float)[idx]
    grp = None if groups is None else np.asarray(groups)[idx]
    readouts, table = {}, []
    for a, b in pairs:
        for lam in (a, b):
            if lam not in phases:
                phases[lam] = wavelength_phase(data, fits, lam, True, cycles, order, fit_powers,
                                               smooth_window, rows=idx, envelope=envelope)
        r = pair_readout(phases[a], phases[b])
        r.insert(1, "time", np.asarray(data.time, float)[idx])
        r.insert(2, "condition", cond)
        label = pair_label(a, b)
        readouts[label] = r
        refined = 100.0 * float(r["local_ok"].mean()) if len(r) else 0.0
        rms_ab = np.r_[phases[a]["rms_local"].to_numpy(float), phases[b]["rms_local"].to_numpy(float)]
        fit_rms = float(np.nanmedian(rms_ab)) if np.isfinite(rms_ab).any() else np.nan
        m = {k: readout_metrics(r[k], cond, grp) for k in METHODS}
        for k, name in METHODS.items():
            row = dict(pair=label, lambda_a=float(a), lambda_b=float(b), method=name, **m[k],
                       refined_pct=refined if k == "local" else np.nan, gain=np.nan,
                       fit_rms=fit_rms if k == "local" else np.nan)
            if k == "local" and m["local"]["scatter"]:
                row["gain"] = m["cubic"]["scatter"] / m["local"]["scatter"]
            table.append(row)
    return readouts, pd.DataFrame(table)


# One setting changed at a time, against the paper's defaults. "where" says which fit it belongs to:
# the profile through the numbered extrema (changes both readouts) or the local two-mode fit.
VARIANTS = [
    ("defaults (paper)", "—", {}),
    ("profile: parabola (degree 2)", "profile", dict(degree=2)),
    ("profile: degree 4", "profile", dict(degree=4)),
    ("profile: degree 5", "profile", dict(degree=5)),
    ("profile: without the λ±1 pair", "profile", dict(skip_inner=1)),
    ("detection: smoothing 5 points", "profile", dict(smooth_window=5)),
    ("detection: smoothing 21 points", "profile", dict(smooth_window=21)),
    ("detection: prominence 0.01", "profile", dict(prominence=0.01)),
    ("detection: prominence 0.05", "profile", dict(prominence=0.05)),
    ("local: 1-cycle window", "local", dict(cycles=1.0)),
    ("local: 1.5-cycle window", "local", dict(cycles=1.5)),
    ("local: 3-cycle window", "local", dict(cycles=3.0)),
    ("local: 4-cycle window", "local", dict(cycles=4.0)),
    ("local: only the phase free", "local", dict(order=0)),
    ("local: phase + slope free", "local", dict(order=1)),
    ("local: phase + slope + chirp free", "local", dict(order=2)),
    ("local: t1, t2 fixed (Eq. 9)", "local", dict(fit_powers=False)),
    ("local: t1, t2 sloped across the window", "local", dict(envelope=True)),
]
PROFILE_KEYS = ("degree", "skip_inner", "smooth_window", "prominence", "distance")


def parameter_study(
    data: SpectraSet,
    pairs,
    condition,
    groups=None,
    rows=None,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
    variants=None,
    progress=None,
) -> pd.DataFrame:
    """The readout of every pair with one fit setting changed at a time.

    One row per variant, pair and method ("cubic only" rows only for the variants that change the
    profile — the local settings do not touch it). Besides the :func:`readout_metrics` columns:
    ``fitted_pct`` (spectra the profile could be fitted to), ``refined_pct``, ``slope_change_pct``
    and ``scatter_ratio`` against the same pair and method with the defaults, and ``shift`` (rad,
    the mean offset of the readout against the default one — a setting may move the readout as a
    whole without changing its slope). ``progress(i, n, name)`` is called before each variant."""
    variants = VARIANTS if variants is None else variants
    idx = np.arange(len(data)) if rows is None else np.asarray(rows, dtype=int)
    cond = np.asarray(condition, float)[idx]
    base = dict(prominence=prominence, distance=distance, smooth_window=smooth_window, degree=3, skip_inner=0)
    fit_cache: dict = {}
    out, ref = [], {}
    for n_done, (name, where, change) in enumerate(variants):
        if progress is not None:
            progress(n_done, len(variants), name)
        prof = {**base, **{k: v for k, v in change.items() if k in PROFILE_KEYS}}
        key = tuple(sorted(prof.items()))
        if key not in fit_cache:
            fit_cache[key] = phase_fits(data, **prof)
        fits = fit_cache[key]
        loc = dict(cycles=change.get("cycles", LOCAL_CYCLES), order=change.get("order", LOCAL_ORDER),
                   fit_powers=change.get("fit_powers", True), smooth_window=prof["smooth_window"],
                   envelope=change.get("envelope", False))
        readouts, table = compare_pairs(data, fits, pairs, condition, groups, rows=idx, **loc)
        fitted = 100.0 * float(np.mean([fits[i] is not None for i in idx]))
        for _, row in table.iterrows():
            method = row["method"]
            if where == "local" and method == METHODS["cubic"]:
                continue
            col = "cubic" if method == METHODS["cubic"] else "local"
            y = readouts[row["pair"]][col].to_numpy(float)
            r = dict(variant=name, where=where, **row.to_dict(), fitted_pct=fitted,
                     slope_change_pct=np.nan, scatter_ratio=np.nan, shift=np.nan)
            if where == "—":
                ref[(row["pair"], method)] = (row["slope"], row["scatter"], y)
                r.update(slope_change_pct=0.0, scatter_ratio=1.0, shift=0.0)
            elif (row["pair"], method) in ref:
                s0, sc0, y0 = ref[(row["pair"], method)]
                both = np.isfinite(y) & np.isfinite(y0) & np.isfinite(cond)
                r.update(slope_change_pct=100.0 * (row["slope"] - s0) / abs(s0) if s0 else np.nan,
                         scatter_ratio=row["scatter"] / sc0 if sc0 else np.nan,
                         shift=float(np.mean(y[both] - y0[both])) if both.any() else np.nan)
            out.append(r)
    if progress is not None:
        progress(len(variants), len(variants), "")
    return pd.DataFrame(out)


def study_summary(study: pd.DataFrame) -> pd.DataFrame:
    """One row per variant (local-fit readout, all pairs together): the largest slope change, the
    scatter ratio's range, the share of frames refined — the table to read first."""
    loc = study[study["method"] == METHODS["local"]]
    rows = []
    for name, part in loc.groupby("variant", sort=False):
        rows.append(dict(variant=name, where=part["where"].iloc[0],
                         slope_change_max_pct=float(part["slope_change_pct"].abs().max()),
                         scatter_ratio_min=float(part["scatter_ratio"].min()),
                         scatter_ratio_max=float(part["scatter_ratio"].max()),
                         scatter_ratio_median=float(part["scatter_ratio"].median()),
                         refined_pct=float(part["refined_pct"].mean()),
                         fitted_pct=float(part["fitted_pct"].mean())))
    return pd.DataFrame(rows)


def study_spread(study: pd.DataFrame, min_refined: float = 90.0, max_scatter_ratio: float = 1.5) -> pd.DataFrame:
    """How far the slope of each pair moves across the settings that still work — the systematic
    part of its uncertainty, next to the statistical one of a single fit.

    A setting counts for a pair when its local-fit readout is refined on at least ``min_refined`` %
    of the frames and scatters at most ``max_scatter_ratio`` times the default's (a setting that
    breaks the fit says nothing about the slope). Columns: pair, slope (defaults), slope_min,
    slope_max, spread_pct = half the range in % of the default slope, n_settings, and the two
    settings at the ends."""
    loc = study[study["method"] == METHODS["local"]]
    rows = []
    for pair, part in loc.groupby("pair", sort=False):
        base = part[part["where"] == "—"]
        if base.empty or not np.isfinite(base["slope"].iloc[0]) or base["slope"].iloc[0] == 0:
            continue
        s0 = float(base["slope"].iloc[0])
        ok = part[(part["refined_pct"] >= min_refined) & (part["scatter_ratio"] <= max_scatter_ratio)
                  & part["slope"].notna()]
        if ok.empty:
            continue
        lo, hi = ok.loc[ok["slope"].idxmin()], ok.loc[ok["slope"].idxmax()]
        rows.append(dict(pair=pair, slope=s0, slope_min=float(lo["slope"]), slope_max=float(hi["slope"]),
                         spread_pct=float(50.0 * (hi["slope"] - lo["slope"]) / abs(s0)),
                         n_settings=int(len(ok)), lowest_with=str(lo["variant"]), highest_with=str(hi["variant"])))
    return pd.DataFrame(rows)
