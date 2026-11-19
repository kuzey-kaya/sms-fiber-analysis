"""Smoothing of a per-spectrum series for drawing (the fits never see it).

``running_median`` exists because ``scipy.signal.medfilt`` pads the ends of the series with
zeros. On a wavelength track (about 1500 nm) the zeros always sort to the bottom of the window,
so near an end the "median" is not the middle of the neighbourhood any more: a fringe that moves
to shorter wavelengths came out with its first three values identical (5 frames: x2, x2, x2, x3,
...), a fringe that moves to longer wavelengths with its last three. Found by the advisor on
2026-11-18 with a simulated, noise-free temperature ramp of 31 spectra, where three repeated
values out of 31 are plain to see; on the 730-frame cooldown the same two frames at each end
were wrong too, only invisible.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def running_median(values, frames: int) -> np.ndarray:
    """Running median over ``frames`` (made odd) points; near the ends the window shrinks
    symmetrically to what is there (1, 3, 5, ... points), so nothing is padded in.

    A monotonic series comes back unchanged — the median of a symmetric window of a ramp is its
    centre — and the first and last value are always the series' own. ``values`` must be finite;
    callers with gaps filter the finite part.
    """
    x = np.asarray(values, float)
    frames = int(frames) | 1
    half = frames // 2
    n = len(x)
    if half < 1 or n < 3:
        return x.copy()
    out = median_filter(x, size=frames, mode="nearest") if n >= frames else x.copy()
    for i in range(min(half, (n + 1) // 2)):
        out[i] = np.median(x[:2 * i + 1])
        out[n - 1 - i] = np.median(x[n - 1 - 2 * i:])
    if n < frames:                       # shorter than the window: every point is "near an end"
        for i in range(n):
            k = min(i, n - 1 - i)
            out[i] = np.median(x[i - k:i + k + 1])
    return out
