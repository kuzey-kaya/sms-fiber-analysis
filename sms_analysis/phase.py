"""Phase-unwrapping readout of an SMS spectrum (Salik et al., Optica Sensing
Congress 2025, paper SM1E.2).

Method, as published:

1. The extrema on each side of the critical wavelength λc are numbered
   outward: λ+1, λ+2, … above λc and λ−1, λ−2, … below it.
2. The extrema λ+n and λ−n have the same phase, and successive extrema are π
   apart, so Δφ(λ±n) = −(n−1)π relative to the pair closest to λc (the phase
   is highest at λc).
3. A cubic Δφ(λ) = K0 + K1 λ + K2 λ² + K3 λ³ is fitted through these points;
   its maximum is λc.
4. The readout is the phase difference between two fixed wavelengths, e.g.
   φ(1570 nm) − φ(1545 nm).

Why the dynamic range is wide: the absolute phase is unknown, so the labelling
is only fixed up to a common offset — if the extrema nearest λc are missed or a
new pair appears there, every point shifts by the same multiple of π and only
K0 changes.  The difference between two wavelengths does not see K0, so it
varies continuously over many fringe spacings, where single-peak tracking hops.

What it measures on a temperature run: the readout is a phase *difference*,
so a change that shifts every phase by the same amount is invisible to it.  On
the 2024-10-17 cooldown, peak tracking implies a common phase gain of about π
(one new extremum appears at λc) plus a tilt of ~2 rad across the band; only
the tilt reaches the readout (see :func:`tracked_phase_change`).  The paper
demonstrates strain, where the profile changes shape, and leaves temperature
to future work.

Implementation choices not spelled out in the paper (documented here so they
can be checked against the group's own code):

* λc is located per spectrum by trying the few extrema nearest the previous
  frame's λc as the central one and keeping the labelling whose cubic fits
  best (a wrong centre shifts one side by π and the cubic no longer fits).
* The fit uses wavelengths centred on the spectrum's middle for numerical
  conditioning; the coefficients are stored in that centred variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .io import SpectraSet
from .peaks import detect_extrema

PAPER_LAMBDA_A, PAPER_LAMBDA_B = 1545.0, 1570.0   # wavelengths used in SM1E.2


@dataclass
class PhaseFit:
    """Cubic phase model Δφ(λ) of one spectrum."""

    coefficients: np.ndarray   # cubic in (λ − x0), highest power first
    x0: float                  # centring wavelength (nm)
    lambda_c: float            # wavelength of the phase maximum (nm)
    rms: float                 # rad, residual of the cubic fit
    wavelengths: np.ndarray    # extrema used (nm), sorted
    phases: np.ndarray         # their assigned phases (rad)
    kinds: np.ndarray          # "peak" / "dip"
    center: int                # index of the λc extremum in `wavelengths`

    def phase(self, wl) -> np.ndarray:
        """Δφ at wavelength(s) ``wl`` (rad, relative to the λ±1 pair)."""
        return np.polyval(self.coefficients, np.asarray(wl, dtype=float) - self.x0)

    def labels(self) -> list[str]:
        """'λ−2', 'λ−1', 'λc', 'λ+1', … for the extrema used."""
        out = []
        for i in range(len(self.wavelengths)):
            k = i - self.center
            out.append("λc" if k == 0 else f"λ{'+' if k > 0 else '−'}{abs(k)}")
        return out


def _ordered_extrema(wavelengths, spectrum, prominence, distance, smooth_window):
    ext = detect_extrema(wavelengths, spectrum, prominence=prominence, distance=distance,
                         smooth_window=smooth_window)
    wl = np.r_[ext.peak_wavelengths, ext.dip_wavelengths]
    kind = np.array(["peak"] * len(ext.peak_wavelengths) + ["dip"] * len(ext.dip_wavelengths))
    order = np.argsort(wl)
    return wl[order], kind[order]


def _fit_with_center(wl, kind, c, x0):
    """Assign Δφ(λ±n) = −(n−1)π around extremum c and fit the cubic."""
    n = np.abs(np.arange(len(wl)) - c)
    keep = n > 0                       # the λc extremum has no assigned phase
    phi = -(n[keep] - 1) * np.pi
    x = wl[keep] - x0
    if keep.sum() < 5 or (np.arange(len(wl)) < c).sum() < 2 or (np.arange(len(wl)) > c).sum() < 2:
        return None
    coef = np.polyfit(x, phi, 3)
    rms = float(np.sqrt(np.mean((np.polyval(coef, x) - phi) ** 2)))
    return coef, rms, wl[keep], phi, kind[keep]


def _vertex(coef, x0, lo, hi):
    """Wavelength of the cubic's maximum inside [lo, hi], or NaN."""
    d = np.polyder(coef)
    roots = np.roots(d)
    roots = roots[np.isreal(roots)].real + x0
    roots = [r for r in roots if lo <= r <= hi and np.polyval(np.polyder(d), r - x0) < 0]
    return float(roots[0]) if roots else float("nan")


def fit_phase(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    lambda_c_guess: float | None = None,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
    candidates: int = 5,
) -> PhaseFit | None:
    """Fit the SM1E.2 cubic phase model to one spectrum.

    ``lambda_c_guess`` (nm) is where λc is expected (e.g. the previous frame's
    value); without it, the extremum flanked by the widest spacings is used.
    The ``candidates`` extrema nearest the guess are each tried as λc and the
    best-fitting labelling wins.  Returns None when fewer than two extrema
    lie on either side.
    """
    wl, kind = _ordered_extrema(wavelengths, spectrum, prominence, distance, smooth_window)
    if len(wl) < 6:
        return None
    x0 = float(np.mean([wavelengths.min(), wavelengths.max()]))
    if lambda_c_guess is None or not np.isfinite(lambda_c_guess):
        gaps = np.diff(wl)
        span = np.r_[0, gaps] + np.r_[gaps, 0]          # spacing on both sides of each extremum
        lambda_c_guess = float(wl[int(np.argmax(span))])
    near = np.argsort(np.abs(wl - lambda_c_guess))[:candidates]

    best = None
    for c in near:
        res = _fit_with_center(wl, kind, int(c), x0)
        if res is None:
            continue
        coef, rms, w_used, phi, k_used = res
        if best is None or rms < best[1]:
            best = (coef, rms, w_used, phi, k_used, int(c))
    if best is None:
        return None
    coef, rms, w_used, phi, k_used, c = best
    # store the λc extremum too, so labels/plots can show it (phase = NaN)
    w_all = np.insert(w_used, c, wl[c])
    p_all = np.insert(phi.astype(float), c, np.nan)
    k_all = np.insert(k_used, c, kind[c])
    lam_c = _vertex(coef, x0, float(wavelengths.min()), float(wavelengths.max()))
    return PhaseFit(coefficients=coef, x0=x0, lambda_c=lam_c, rms=rms,
                    wavelengths=w_all, phases=p_all, kinds=k_all, center=c)


def phase_fits(
    data: SpectraSet,
    lambda_c_guess: float | None = None,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
) -> list[PhaseFit | None]:
    """:func:`fit_phase` for every spectrum; each frame's λc seeds the next."""
    fits, guess = [], lambda_c_guess
    for i in range(len(data)):
        fit = fit_phase(data.wavelengths, data.spectra[i], guess, prominence, distance, smooth_window)
        fits.append(fit)
        if fit is not None and np.isfinite(fit.lambda_c):
            guess = fit.lambda_c
    return fits


def phase_series(
    data: SpectraSet,
    lambda_a: float = PAPER_LAMBDA_A,
    lambda_b: float = PAPER_LAMBDA_B,
    lambda_c_guess: float | None = None,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
    fits: list[PhaseFit | None] | None = None,
) -> pd.DataFrame:
    """The SM1E.2 readout for every spectrum of a run.

    Columns: time, temperature, lambda_c (nm), phi_a, phi_b (rad, relative to
    each frame's λ±1 pair), phase_diff = |φ(λb) − φ(λa)| (rad, the paper's
    plotted quantity), phase_diff_signed, rms (rad, cubic-fit residual),
    n_extrema, extrema_between (number of extrema detected between λa and λb;
    |phase_diff|/π should stay within about one of it).
    Pass ``fits`` (from :func:`phase_fits`) to reuse them for another pair.
    """
    if fits is None:
        fits = phase_fits(data, lambda_c_guess, prominence, distance, smooth_window)
    lo, hi = sorted((lambda_a, lambda_b))
    rows = []
    for i, fit in enumerate(fits):
        if fit is None:
            rows.append((data.time[i], data.temperature[i], np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.nan, 0, 0))
            continue
        pa, pb = fit.phase([lambda_a, lambda_b])
        between = int(((fit.wavelengths > lo) & (fit.wavelengths < hi)).sum())
        rows.append((data.time[i], data.temperature[i], fit.lambda_c, pa, pb, abs(pb - pa), pb - pa,
                     fit.rms, len(fit.wavelengths), between))
    return pd.DataFrame(rows, columns=["time", "temperature", "lambda_c", "phi_a", "phi_b", "phase_diff",
                                       "phase_diff_signed", "rms", "n_extrema", "extrema_between"])


def scan_pairs(fits: list[PhaseFit | None], condition: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """Linear sensitivity of φ(λb) − φ(λa) to the condition for every pair on ``grid``.

    Columns: lambda_a, lambda_b, slope (rad per unit), r_squared, resid_rad
    (scatter about the line), resolution (resid / |slope|, in condition units —
    an upper bound, since curvature counts as scatter), span_rad.
    """
    ok = [i for i, f in enumerate(fits) if f is not None and np.isfinite(condition[i])]
    if len(ok) < 3:
        return pd.DataFrame(columns=["lambda_a", "lambda_b", "slope", "r_squared", "resid_rad",
                                     "resolution", "span_rad"])
    x = np.asarray(condition, dtype=float)[ok]
    P = np.array([fits[i].phase(grid) for i in ok])
    rows = []
    for i, a in enumerate(grid):
        for j in range(i + 1, len(grid)):
            y = P[:, j] - P[:, i]
            r = stats.linregress(x, y)
            resid = float(np.std(y - (r.slope * x + r.intercept)))
            rows.append((float(a), float(grid[j]), float(r.slope), float(r.rvalue ** 2), resid,
                         resid / abs(r.slope) if r.slope else np.inf, float(np.ptp(y))))
    return pd.DataFrame(rows, columns=["lambda_a", "lambda_b", "slope", "r_squared", "resid_rad",
                                       "resolution", "span_rad"])


def tracked_phase_change(fit0: PhaseFit, tracks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Phase change implied by peak tracking, fringe by fringe.

    For each tracked fringe, integrates the first spectrum's phase slope
    dΔφ/dλ along the path from its first to its last wavelength.  If the
    condition shifted every phase by the same amount (a common mode), all rows
    show the same value; the spread between them is the part the SM1E.2
    readout (a phase *difference*) can see.
    """
    dphi = np.polyder(fit0.coefficients)
    rows = []
    for label, tr in tracks.items():
        w = tr["wavelength"].to_numpy(float)
        if not (np.isfinite(w[0]) and np.isfinite(w[-1])):
            continue
        xs = np.linspace(w[0], w[-1], 200)
        delta = float(np.trapezoid(np.polyval(dphi, xs - fit0.x0), xs))
        rows.append((label, float(w[0]), float(w[-1] - w[0]), delta))
    return pd.DataFrame(rows, columns=["fringe", "start_nm", "shift_nm", "phase_change_rad"]) \
        .sort_values("start_nm").reset_index(drop=True)


@dataclass
class PhaseSensitivity:
    slope: float        # rad per unit of the condition (rad/°C for temperature)
    intercept: float
    stderr: float
    r_squared: float
    n_points: int
    span_rad: float     # total change of the readout over the run (rad)

    @property
    def span_fringes(self) -> float:
        """The same change in fringes (π rad per extremum, 2π per full fringe)."""
        return self.span_rad / (2 * np.pi)


def phase_sensitivity(series: pd.DataFrame, max_rms: float | None = 0.5) -> PhaseSensitivity:
    """Linear fit of the phase-difference readout against the condition.

    Frames whose cubic fit is poor (``rms`` above ``max_rms`` rad) are left
    out; they are the frames where the extrema next to λc merge.
    """
    ok = series.dropna(subset=["phase_diff", "temperature"])
    if max_rms is not None:
        ok = ok[ok["rms"] <= max_rms]
    if len(ok) < 3:
        raise ValueError("Not enough valid frames to fit")
    res = stats.linregress(ok["temperature"], ok["phase_diff"])
    return PhaseSensitivity(slope=float(res.slope), intercept=float(res.intercept),
                            stderr=float(res.stderr), r_squared=float(res.rvalue ** 2),
                            n_points=len(ok),
                            span_rad=float(ok["phase_diff"].max() - ok["phase_diff"].min()))
