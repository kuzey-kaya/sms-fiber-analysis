"""Tracking one spectral feature (a fringe peak or dip) across the time series.

The tracker looks for the extremum inside a window centred on where the
feature was last seen, which keeps it locked to the *same* fringe even when
neighbouring fringes are only a few nm away.  Positions are refined by
parabolic interpolation on the raw data for sub-sample (< 0.1 nm) precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import SpectraSet
from .peaks import detect_extrema, parabolic_refine, smooth


def track_feature(
    data: SpectraSet,
    start_wavelength: float | None = None,
    kind: str = "dip",
    search_window: float = 1.5,
    smooth_window: int = 11,
    max_step: float | None = 0.5,
    smoothed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Track a single peak or dip through all spectra.

    Parameters
    ----------
    data : the loaded spectra set.
    start_wavelength : where the feature is in the first spectrum (nm).
        ``None`` auto-picks the global extremum of the first spectrum.
    kind : ``"peak"`` (maximum) or ``"dip"`` (minimum).
    search_window : half-width (nm) of the search window around the last
        known position.  Must be smaller than the local fringe spacing.
    max_step : if set, a jump larger than this (nm) between consecutive rows
        is treated as a lost lock and recorded as NaN.  With ~20 s between
        spectra a real fringe moves well under 0.1 nm per row, so anything
        bigger is a fringe-hop or a noise spike, not signal.
    smoothed : optional pre-smoothed copy of ``data.spectra`` (same shape,
        smoothed with ``smooth_window``); lets callers that track many
        fringes smooth each spectrum once instead of once per fringe.

    Notes
    -----
    A hit at the very edge of the search window means the window no longer
    contains a true local extremum (the fringe has faded or merged, e.g.
    when it approaches the critical wavelength).  Such rows are recorded as
    NaN instead of letting the tracker "surf" along a slope onto the next
    fringe -- this is what causes the multi-nm cliffs seen with naive
    windowed tracking.

    Returns
    -------
    DataFrame with columns: time, temperature, wavelength, value.
    """
    if kind not in ("peak", "dip"):
        raise ValueError("kind must be 'peak' or 'dip'")

    wl = data.wavelengths
    pick = np.argmin if kind == "dip" else np.argmax

    if start_wavelength is None:
        y0 = smooth(data.spectra[0], smooth_window)
        start_wavelength = float(wl[pick(y0)])

    last = start_wavelength
    records = []
    for i in range(len(data)):
        y_raw = data.spectra[i]
        y_s = smooth(y_raw, smooth_window) if smoothed is None else smoothed[i]

        mask = np.abs(wl - last) <= search_window
        in_window = np.flatnonzero(mask)
        if in_window.size < 3:
            records.append((data.time[i], data.temperature[i], np.nan, np.nan))
            continue

        local = pick(y_s[mask])
        # A hit at the window edge is not a real extremum -> lost lock.
        if local in (0, in_window.size - 1):
            records.append((data.time[i], data.temperature[i], np.nan, np.nan))
            continue

        idx = in_window[local]
        # Refine on the smoothed curve: sub-sample precision without letting
        # single noisy samples throw the vertex off.
        w, v = parabolic_refine(wl, y_s, idx)

        if max_step is not None and abs(w - last) > max_step:
            records.append((data.time[i], data.temperature[i], np.nan, np.nan))
            continue

        records.append((data.time[i], data.temperature[i], w, v))
        last = w

    return pd.DataFrame(records, columns=["time", "temperature", "wavelength", "value"])


def track_all_extrema(
    data: SpectraSet,
    prominence: float = 0.02,
    distance: int = 5,
    search_window: float = 1.5,
    smooth_window: int = 11,
    max_step: float | None = 0.5,
) -> dict[str, pd.DataFrame]:
    """Track every extremum found in the first spectrum.

    Returns a dict mapping a label like ``"dip@1526.2"`` (its wavelength in
    the first spectrum) to that feature's tracking DataFrame.  Useful for
    mapping how sensitivity varies across the spectrum: fringes close to the
    critical wavelength move much more per degree than fringes at the band
    edges, and the two sides of the critical wavelength can move in opposite
    directions -- which is why peak motion looks "irregular" if features are
    not tracked individually.
    """
    first = detect_extrema(
        data.wavelengths, data.spectra[0], prominence=prominence,
        distance=distance, smooth_window=smooth_window,
    )
    # Smooth every spectrum once; each fringe's tracker reuses the result.
    smoothed = np.array([smooth(y, smooth_window) for y in data.spectra])
    tracks: dict[str, pd.DataFrame] = {}
    for kind, wls in (("peak", first.peak_wavelengths), ("dip", first.dip_wavelengths)):
        for w in wls:
            label = f"{kind}@{w:.1f}"
            tracks[label] = track_feature(
                data, start_wavelength=float(w), kind=kind,
                search_window=search_window, smooth_window=smooth_window,
                max_step=max_step, smoothed=smoothed,
            )
    return tracks


def track_feature_legacy(
    data: SpectraSet,
    kind: str = "dip",
    search_window: float = 4.0,
    smooth_window: int = 7,
    median_kernel: int | None = None,
) -> pd.DataFrame:
    """Reproduce the original Colab tracking exactly (fringe-hops included).

    Same algorithm as the old notebook cell: start from the global extremum
    of the first spectrum, search within ``search_window`` nm of the last
    position, take the extremum even if it sits at the window edge, refine
    on the *raw* data with the *unguarded* parabolic formula, and never
    reject a jump.  On this data that combination produces the big ~11 nm
    step at t ≈ 7660 s: the unguarded vertex formula on noisy raw samples
    occasionally throws the position several nm, the search window teleports
    with it, and the tracker lands in the deep critical-wavelength trough.
    Kept because it reproduces the original graphs exactly; use
    :func:`track_feature` for calibration-grade tracking.

    Larger ``search_window`` values make the tracker hop between fringes
    more often (each hop is one step in the time trace); ``median_kernel``
    (odd, e.g. 15) applies a median filter to the tracked wavelength
    afterwards, which suppresses single-row toggles and leaves clean
    plateaus separated by the fringe-hop steps.
    """
    wl = data.wavelengths
    pick = np.argmin if kind == "dip" else np.argmax

    def _refine_unguarded(y: np.ndarray, idx: int) -> tuple[float, float]:
        # Verbatim behaviour of the original notebook (no |offset| <= 1 guard).
        if idx <= 0 or idx >= len(y) - 1:
            return float(wl[idx]), float(y[idx])
        y_l, y_c, y_r = y[idx - 1], y[idx], y[idx + 1]
        denom = y_l - 2.0 * y_c + y_r
        if denom == 0.0:
            return float(wl[idx]), float(y_c)
        offset = 0.5 * (y_l - y_r) / denom
        dx = wl[idx + 1] - wl[idx]
        return float(wl[idx] + offset * dx), float(y_c - 0.25 * (y_l - y_r) * offset)

    y0 = smooth(data.spectra[0], smooth_window)
    last = float(wl[pick(y0)])

    records = []
    for i in range(len(data)):
        y_raw = data.spectra[i]
        y_s = smooth(y_raw, smooth_window)
        mask = np.abs(wl - last) <= search_window
        if not mask.any():
            records.append((data.time[i], data.temperature[i], np.nan, np.nan))
            continue
        idx = np.flatnonzero(mask)[pick(y_s[mask])]
        w, v = _refine_unguarded(y_raw, idx)
        records.append((data.time[i], data.temperature[i], w, v))
        last = w

    out = pd.DataFrame(records, columns=["time", "temperature", "wavelength", "value"])
    if median_kernel:
        from scipy.signal import medfilt
        out["wavelength"] = medfilt(out["wavelength"].to_numpy(), median_kernel)
    return out


def track_band_extremum(
    data: SpectraSet,
    band: tuple[float, float],
    mode: str = "peak",
    smooth_window: int = 11,
    median_kernel: int | None = 21,
) -> pd.DataFrame:
    """Track the strongest extremum inside a fixed wavelength band.

    Unlike :func:`track_feature` (which follows one fringe), this follows
    whichever fringe currently has the highest peak (``mode="peak"``) or the
    deepest dip (``mode="dip"``) inside ``band``.  As the spectrum evolves
    with temperature, that identity hops from fringe to fringe, so the time
    trace is a staircase: flat plateaus (one per winning fringe) separated
    by steps of one fringe spacing.  A median filter (odd ``median_kernel``)
    removes single-row toggles when two fringes have nearly equal strength.
    """
    if mode not in ("peak", "dip"):
        raise ValueError("mode must be 'peak' or 'dip'")
    wl = data.wavelengths
    m = (wl >= band[0]) & (wl <= band[1])
    idx = np.flatnonzero(m)
    pick = np.argmin if mode == "dip" else np.argmax

    records = []
    for i in range(len(data)):
        y_s = smooth(data.spectra[i], smooth_window)
        j = idx[pick(y_s[m])]
        records.append((data.time[i], data.temperature[i], float(wl[j]),
                        float(data.spectra[i][j])))
    out = pd.DataFrame(records, columns=["time", "temperature", "wavelength", "value"])
    if median_kernel:
        from scipy.signal import medfilt
        out["wavelength"] = medfilt(out["wavelength"].to_numpy(), median_kernel)
    return out
