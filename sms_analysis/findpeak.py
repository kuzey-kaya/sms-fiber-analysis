"""Igor Pro's ``FindPeak`` operation, and a tracker built on it (the advisor's workflow).

The group's Igor procedure (``trackpeaknew("n", 1, 1, 914, wave0, "min1576a", temp, tim,
735, 766, -1)`` in the 2026-10-22 screen share) calls FindPeak once per spectrum inside a
fixed point range.  Igor's reference for the operation:

* the input is box-smoothed (sliding average, ``box`` points; Igor's BoxSmooth),
* the peak centre is where the first derivative of the smoothed wave crosses zero,
  located by linear interpolation of the derivative,
* the edges are where the second derivative crosses zero (``V_LeadingEdgeLoc``,
  ``V_TrailingEdgeLoc``; their distance is ``V_PeakWidth``),
* the peak value is the greater of the two *unsmoothed* samples around the centre
  (the lesser with ``/N``),
* ``/N`` looks for a minimum, ``/M=minLevel`` ignores peaks below (above, with ``/N``) that
  level in the smoothed wave, ``/R=(startX,endX)`` sets the range and the search direction
  (the first peak from ``startX`` wins), ``/I`` accepts one-sample impulses.

:func:`find_peak` is that operation on one spectrum; :func:`track_findpeak` repeats it on
every spectrum of a run inside the same fixed wavelength range, which is what the Igor
graphs ("peakmin1576a vs timemin1576a") show.  It uses only the samples inside the range —
nothing else of the spectrum — so it is as simple and as local as Igor's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import SpectraSet


@dataclass
class FindPeakResult:
    """What Igor returns in its V_ variables (NaN where Igor sets NaN; ``flag`` 0 = found)."""

    flag: int
    peak_loc: float
    peak_val: float
    peak_width: float
    leading_edge: float
    trailing_edge: float


def box_smooth(y: np.ndarray, box: int) -> np.ndarray:
    """Sliding average over ``box`` points (Igor's BoxSmooth, ends reflected); box ≤ 1 = no smoothing."""
    y = np.asarray(y, float)
    box = int(box)
    if box <= 1 or y.size < 3:
        return y.copy()
    box = min(box | 1, (y.size - 1) | 1)        # odd, and never wider than the data
    h = box // 2
    padded = np.pad(y, h, mode="reflect")
    return np.convolve(padded, np.ones(box) / box, mode="valid")


def _zero_crossings(x: np.ndarray, d: np.ndarray, downward: bool) -> list[float]:
    """x positions where ``d`` crosses zero (downward: + → −; else − → +), linearly interpolated."""
    out = []
    for i in range(d.size - 1):
        a, b = d[i], d[i + 1]
        hit = (a > 0 >= b) if downward else (a < 0 <= b)
        if hit:
            if a == b:
                out.append(float(x[i]))
            else:
                out.append(float(x[i] + (x[i + 1] - x[i]) * a / (a - b)))
    return out


def find_peak(
    x: np.ndarray,
    y: np.ndarray,
    start: float | None = None,
    end: float | None = None,
    box: int = 5,
    negative: bool = False,
    min_level: float | None = None,
) -> FindPeakResult:
    """Igor's ``FindPeak [/B=box /N /M=minLevel /R=(start,end)]`` on one wave.

    ``start`` > ``end`` searches backwards, as in Igor.  Returns ``flag != 0`` and NaNs when
    no peak is found inside the range.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nan = FindPeakResult(1, np.nan, np.nan, np.nan, np.nan, np.nan)
    if start is None:
        start = x[0]
    if end is None:
        end = x[-1]
    lo, hi = min(start, end), max(start, end)
    m = (x >= lo) & (x <= hi)
    if m.sum() < 4:
        return nan
    # Igor smooths the whole wave and then searches the range: smoothing only the range would
    # bend the ends of the box average and plant false zero crossings at the range edges
    ys_all = box_smooth(y, box)
    xs, ys_raw, ys = x[m], y[m], ys_all[m]
    d1 = np.gradient(ys, xs)
    # candidate centres, in the search direction (Igor takes the first one it meets)
    centres = _zero_crossings(xs, d1, downward=not negative)
    if start > end:
        centres = centres[::-1]
    for c in centres:
        # value of the smoothed wave at the centre, for the /M test
        v_s = float(np.interp(c, xs, ys))
        if min_level is not None and ((not negative and v_s <= min_level) or (negative and v_s >= min_level)):
            continue
        j = int(np.searchsorted(xs, c))            # the two unsmoothed samples around the centre
        pair = ys_raw[max(j - 1, 0):min(j + 1, ys_raw.size)]
        v = float(pair.min() if negative else pair.max())
        d2 = np.gradient(d1, xs)
        # edges: the inflection points (zero crossings of the second derivative, either way)
        # nearest the centre on each side; NaN when one lies outside the range, as in Igor
        edges = sorted(_zero_crossings(xs, d2, True) + _zero_crossings(xs, d2, False))
        left = [e for e in edges if e < c]
        right = [e for e in edges if e > c]
        lead = left[-1] if left else np.nan
        trail = right[0] if right else np.nan
        if start > end:
            lead, trail = trail, lead
        width = abs(trail - lead) if np.isfinite(lead) and np.isfinite(trail) else np.nan
        return FindPeakResult(0, float(c), v, width, lead, trail)
    return nan


def track_findpeak(
    data: SpectraSet,
    start_nm: float,
    end_nm: float,
    box: int = 5,
    negative: bool = True,
    min_level: float | None = None,
) -> pd.DataFrame:
    """Igor's FindPeak on every spectrum inside one fixed wavelength range.

    Exactly the group's ``trackpeaknew`` graph: the same range for every spectrum, the first
    minimum (``negative=True``, Igor's ``/N``) or maximum met from ``start_nm``.  A spectrum
    whose range holds no such extremum gets NaN (Igor's V_flag ≠ 0).  Columns: time,
    temperature, wavelength (V_PeakLoc), value (V_PeakVal), width (V_PeakWidth), found.
    """
    rows = []
    wl = data.wavelengths
    for i in range(len(data)):
        r = find_peak(wl, data.spectra[i], start_nm, end_nm, box, negative, min_level)
        rows.append((data.time[i], data.temperature[i], r.peak_loc, r.peak_val, r.peak_width, r.flag == 0))
    return pd.DataFrame(rows, columns=["time", "temperature", "wavelength", "value", "width", "found"])


def points_to_nm(data: SpectraSet, start_point: int, end_point: int) -> tuple[float, float]:
    """Igor point range (``/R=[startP,endP]``, as in the advisor's calls) → wavelengths of this file."""
    n = len(data.wavelengths) - 1
    a, b = int(np.clip(start_point, 0, n)), int(np.clip(end_point, 0, n))
    return float(data.wavelengths[a]), float(data.wavelengths[b])
