"""Peak / dip detection on a single transmission spectrum.

SMS sensors are modal Mach-Zehnder interferometers: the spectrum is a fringe
pattern T(lambda) = t1 + t2 + 2*sqrt(t1*t2)*cos(dphi(lambda)) whose maxima and
minima shift with temperature or strain.  Fringes are narrow at the edges of
the band and widen towards the critical wavelength (dispersion turning point).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass
class ExtremaSet:
    """Detected maxima ("peaks") and minima ("dips") of one spectrum."""

    peak_wavelengths: np.ndarray
    peak_values: np.ndarray
    dip_wavelengths: np.ndarray
    dip_values: np.ndarray


def smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Light Savitzky-Golay smoothing so noise does not create fake extrema."""
    window = min(window if window % 2 else window + 1, len(y) - 1)
    return savgol_filter(y, window, poly)


def parabolic_refine(x: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float]:
    """Refine an extremum position to sub-sample precision.

    Fits a parabola through the extremum sample and its two neighbours and
    returns the vertex (wavelength, value).  Falls back to the raw sample at
    the array edges or on a degenerate (flat) neighbourhood.
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(x[idx]), float(y[idx])
    y_l, y_c, y_r = y[idx - 1], y[idx], y[idx + 1]
    denom = y_l - 2.0 * y_c + y_r
    if denom == 0.0:
        return float(x[idx]), float(y_c)
    offset = 0.5 * (y_l - y_r) / denom  # in units of one sample step
    if abs(offset) > 1.0:
        # Nearly-flat neighbourhood: the parabola vertex is unphysical
        # (can shoot off by orders of magnitude) -- keep the raw sample.
        return float(x[idx]), float(y_c)
    dx = x[idx + 1] - x[idx]
    return float(x[idx] + offset * dx), float(y_c - 0.25 * (y_l - y_r) * offset)


def detect_extrema(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
    refine: bool = True,
) -> ExtremaSet:
    """Detect all maxima and minima of one spectrum.

    Detection runs on a smoothed copy; positions are refined by parabolic
    interpolation on the *raw* data so smoothing does not bias them.
    """
    y_s = smooth(spectrum, smooth_window)
    up_idx, _ = find_peaks(y_s, prominence=prominence, distance=distance)
    dn_idx, _ = find_peaks(-y_s, prominence=prominence, distance=distance)

    def _refine(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not refine:
            return wavelengths[indices], spectrum[indices]
        pairs = [parabolic_refine(wavelengths, spectrum, i) for i in indices]
        if not pairs:
            return np.array([]), np.array([])
        wl, val = map(np.array, zip(*pairs))
        return wl, val

    pw, pv = _refine(up_idx)
    dw, dv = _refine(dn_idx)
    return ExtremaSet(pw, pv, dw, dv)
