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
5. (IEEE PTL manuscript, Salik et al. 2026, "Wide-Range and Linear Strain Sensing in SMS
   Fiber Sensors Using Phase Unwrapping", Sec. III) The phase at each of the two
   wavelengths is then refined by a local nonlinear least-squares fit of the two-mode
   model T(λ) = t1² + t2² + 2 t1 t2 cos φ(λ) (Eq. 1) over at least one fringe cycle,
   with the cubic's K_m and the mode powers t1, t2 as parameters; t1, t2 start from the
   nearest maximum and minimum, t1 = (√Tmax + √Tmin)/2, t2 = (√Tmax − √Tmin)/2 (Eq. 9).
   :func:`local_fit`, ``phase_series(local=True)``.  Checked 2026-10-22: on the cooldown
   the 1545/1570 readout's scatter about its line drops from 25 to 6.6 mrad (0.80 → 0.20 °C
   resolution), 1485/1565 from 21 to 5.5 mrad; on the strain run the pair 1560/1600 (both
   right of λc, as the paper recommends) from 60 to 9 mrad with the within-plateau noise
   47 → 4 mrad; t1 ≈ 0.70–0.80, t2 ≈ 0.15–0.21 (visibility ≈ 0.5), fit RMS ≈ 0.003 (cooldown)
   / 0.01 (strain) in transmittance.  Two-cycle windows did best; a window that would hold
   λc is refused (1545 nm on the strain run, λc ≈ 1530) and the frame keeps the cubic
   values for both wavelengths.

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
from .peaks import detect_extrema, smooth

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


def _ordered_extrema(wavelengths, spectrum, prominence, distance, smooth_window, smoothed=None):
    ext = detect_extrema(wavelengths, spectrum, prominence=prominence, distance=distance,
                         smooth_window=smooth_window, smoothed=smoothed)
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
    smoothed: np.ndarray | None = None,
) -> PhaseFit | None:
    """Fit the SM1E.2 cubic phase model to one spectrum.

    ``lambda_c_guess`` (nm) is where λc is expected (e.g. the previous frame's
    value); without it, the extremum flanked by the widest spacings is used.
    The ``candidates`` extrema nearest the guess are each tried as λc and the
    best-fitting labelling wins.  Returns None when fewer than two extrema
    lie on either side.
    """
    wl, kind = _ordered_extrema(wavelengths, spectrum, prominence, distance, smooth_window, smoothed)
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
    smoothed = smooth(data.spectra, smooth_window)        # once for the run, not once per spectrum
    for i in range(len(data)):
        fit = fit_phase(data.wavelengths, data.spectra[i], guess, prominence, distance, smooth_window,
                        smoothed=smoothed[i])
        fits.append(fit)
        if fit is not None and np.isfinite(fit.lambda_c):
            guess = fit.lambda_c
    return fits


@dataclass
class LocalPhaseFit:
    """Local nonlinear least-squares fit of the two-mode model around one wavelength.

    PTL manuscript (Salik et al. 2026, Sec. III): T(λ) = t1² + t2² + 2 t1 t2 cos φ(λ), with φ a
    cubic in (λ − lam) whose coefficients start from the global cubic (:class:`PhaseFit`) and
    t1, t2 from the nearest maximum and minimum (Eq. 9); the window spans at least one full
    fringe cycle around ``lam``.
    """

    lam: float                 # wavelength the phase is wanted at (nm)
    phi: float                 # refined phase at lam (rad, same branch as the global cubic)
    phi_cubic: float           # the global cubic's value at lam, for comparison
    t1: float
    t2: float
    coefficients: np.ndarray   # cubic in (λ − lam), highest power first
    window: tuple[float, float]
    rms: float                 # transmittance residual of the local fit
    n_points: int
    converged: bool

    @property
    def visibility(self) -> float:
        """Fringe visibility 2 t1 t2 / (t1² + t2²) implied by the fitted mode powers."""
        d = self.t1 ** 2 + self.t2 ** 2
        return float(2 * self.t1 * self.t2 / d) if d else float("nan")


def _t1_t2_from_extrema(t_max: float, t_min: float) -> tuple[float, float]:
    """Eq. 9 of the paper: mode amplitudes from the nearest maximum and minimum of T."""
    a, b = np.sqrt(max(t_max, 0.0)), np.sqrt(max(t_min, 0.0))
    return float((a + b) / 2), float((a - b) / 2)


LOCAL_CYCLES = 2.0     # window width in local fringe periods (checked 2026-10-22 on both runs)


EDGE_CLEARANCE = 0.4   # periods, measured with the local period at the window edge (2026-10-29)


def window_clearance(wavelengths: np.ndarray, fit: PhaseFit, lam: float,
                     cycles: float = LOCAL_CYCLES) -> float:
    """How far the local-fit window around ``lam`` stays from λc, in *local* fringe periods
    measured at the window edge nearest λc (where the period is longest). NaN when the cubic
    is unusable there, 0 when λc lies inside the window.

    Why the edge and not ``lam`` (found 2026-10-29 while checking the site against the advisor's
    papers): on the strain run the 1560 nm window kept 0.57 periods of clearance measured at
    1560 nm on every frame, yet under the heaviest load the broad fringes next to λc had moved
    into it (period at the edge ≈ 75 nm, clearance there 0.20) and the fit went wrong on that one
    step; 1565 nm keeps ≥ 0.51 at the edge and stays linear. The cooldown's 1545 nm sits at 0.48.
    The app's default-pair walk requires ≥ ``EDGE_CLEARANCE`` on frames spread over the run."""
    wl = np.asarray(wavelengths, float)
    c = fit.coefficients
    slope = float(np.polyval(np.polyder(c), lam - fit.x0))
    if not np.isfinite(slope) or abs(slope) < 1e-6:
        return float("nan")
    period = 2 * np.pi / abs(slope)
    half = 0.5 * max(cycles, 1.0) * period
    lo, hi = lam - half, lam + half
    if lo < wl.min() or hi > wl.max():
        half = 0.5 * period
        lo, hi = lam - half, lam + half
        if lo < wl.min() or hi > wl.max():
            return float("nan")
    if not np.isfinite(fit.lambda_c):
        return float("inf")
    if lo <= fit.lambda_c <= hi:
        return 0.0
    edge = lo if abs(lo - fit.lambda_c) < abs(hi - fit.lambda_c) else hi
    s_edge = float(np.polyval(np.polyder(c), edge - fit.x0))
    if not np.isfinite(s_edge) or abs(s_edge) < 1e-9:
        return 0.0
    return float(abs(edge - fit.lambda_c) * abs(s_edge) / (2 * np.pi))


def local_fit(wavelengths: np.ndarray, spectrum: np.ndarray, fit: PhaseFit, lam: float,
              cycles: float = LOCAL_CYCLES, smooth_window: int = 11) -> LocalPhaseFit | None:
    """Refine the intermodal phase at ``lam`` by fitting Eq. 1 to the spectrum around it.

    The window is ``cycles`` local fringe periods wide (the period from the global cubic's
    slope at ``lam``), never less than one full cycle, and shrinks to one cycle when it would
    leave the band; t1 and t2 start from the nearest maximum and minimum inside it; the four
    cubic coefficients start from the global fit re-centred at ``lam``.  Returns None when
    even a one-cycle window leaves the band, when the window would come within half a period
    of λc (the two-mode model is still right there, but the phase has its extremum next to the
    window and the fit is not anchored — the readout then keeps the cubic values), or when the
    cubic's slope vanishes at ``lam``.
    """
    from scipy.optimize import least_squares

    wl = np.asarray(wavelengths, float)
    y = np.asarray(spectrum, float)
    c = fit.coefficients
    x_l = lam - fit.x0
    slope = float(np.polyval(np.polyder(c), x_l))
    if not np.isfinite(slope) or abs(slope) < 1e-6:
        return None
    period = 2 * np.pi / abs(slope)
    half = 0.5 * max(cycles, 1.0) * period
    lo, hi = lam - half, lam + half
    if lo < wl.min() or hi > wl.max():
        half = 0.5 * period                                   # one full cycle, the paper's minimum
        lo, hi = lam - half, lam + half
        if lo < wl.min() or hi > wl.max():
            return None
    # a window that holds λc, or starts within half a period of it, is not anchored: the phase
    # has its extremum there and the fit drifts (checked on the strain run: 1555 nm with the edge
    # 0.17 period from λc doubled the plateau noise, 1560 nm at 0.57 period cut it 7-fold)
    if np.isfinite(fit.lambda_c) and (lo - 0.5 * period) <= fit.lambda_c <= (hi + 0.5 * period):
        return None
    m = (wl >= lo) & (wl <= hi)
    if m.sum() < 12:
        return None
    xw, yw = wl[m] - lam, y[m]
    # the global cubic re-centred at lam (Taylor expansion = polynomial shift)
    d1, d2, d3 = (float(np.polyval(np.polyder(c, k), x_l)) for k in (1, 2, 3))
    c0 = [d3 / 6.0, d2 / 2.0, d1, float(np.polyval(c, x_l))]       # highest power first
    ys = smooth(yw, smooth_window) if yw.size > smooth_window else yw
    t1, t2 = _t1_t2_from_extrema(float(ys.max()), float(ys.min()))
    if t2 <= 0:
        t2 = 0.05 * t1

    def model(p, x):
        a, b = p[0], p[1]
        return a * a + b * b + 2 * a * b * np.cos(np.polyval(p[2:], x))

    p0 = np.array([t1, t2, *c0])
    rms0 = float(np.sqrt(np.mean((model(p0, xw) - yw) ** 2)))
    res = least_squares(lambda p: model(p, xw) - yw, p0, method="lm", max_nfev=400)
    coef = res.x[2:]
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    # A fit that ends worse than its starting point, or far from the cubic's phase, has slid to a
    # wrong solution (cos φ has mirror and 2π-shifted minima; seen on noisy SIMULATED runs where the
    # residual was 0.25 instead of 0.008) — report it as not converged, the caller keeps the cubic.
    ok = (bool(res.success) and rms <= max(rms0, 1e-6) and rms <= 0.05
          and abs(float(coef[-1]) - c0[-1]) <= 1.0)
    return LocalPhaseFit(lam=float(lam), phi=float(coef[-1]), phi_cubic=float(c0[-1]),
                         t1=float(abs(res.x[0])), t2=float(abs(res.x[1])), coefficients=coef,
                         window=(float(lo), float(hi)), rms=rms, n_points=int(m.sum()),
                         converged=ok)


def phase_series(
    data: SpectraSet,
    lambda_a: float = PAPER_LAMBDA_A,
    lambda_b: float = PAPER_LAMBDA_B,
    lambda_c_guess: float | None = None,
    prominence: float = 0.02,
    distance: int = 5,
    smooth_window: int = 11,
    fits: list[PhaseFit | None] | None = None,
    local: bool = False,
    cycles: float = LOCAL_CYCLES,
) -> pd.DataFrame:
    """The SM1E.2 readout for every spectrum of a run.

    Columns: time, temperature, lambda_c (nm), phi_a, phi_b (rad, relative to
    each frame's λ±1 pair), phase_diff = |φ(λb) − φ(λa)| (rad, the paper's
    plotted quantity), phase_diff_signed, rms (rad, cubic-fit residual),
    n_extrema, extrema_between (number of extrema detected between λa and λb;
    |phase_diff|/π should stay within about one of it).
    Pass ``fits`` (from :func:`phase_fits`) to reuse them for another pair.

    ``local=True`` adds the PTL manuscript's refinement: :func:`local_fit` of the two-mode model
    around λa and λb; phi_a/phi_b/phase_diff then come from the local fits and the columns
    phi_a_cubic, phi_b_cubic, phase_diff_cubic keep the global cubic's values, with t1_a, t2_a,
    t1_b, t2_b, rms_local (transmittance) and local_ok alongside.  A frame where either local
    fit fails keeps the cubic values for both wavelengths (local_ok = False).
    """
    if fits is None:
        fits = phase_fits(data, lambda_c_guess, prominence, distance, smooth_window)
    lo, hi = sorted((lambda_a, lambda_b))
    cols = ["time", "temperature", "lambda_c", "phi_a", "phi_b", "phase_diff",
            "phase_diff_signed", "rms", "n_extrema", "extrema_between"]
    if local:
        cols += ["phi_a_cubic", "phi_b_cubic", "phase_diff_cubic", "t1_a", "t2_a", "t1_b", "t2_b",
                 "rms_local", "local_ok"]
    rows = []
    for i, fit in enumerate(fits):
        if fit is None:
            rows.append([data.time[i], data.temperature[i], np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.nan, 0, 0] + ([np.nan] * 8 + [False] if local else []))
            continue
        pa, pb = fit.phase([lambda_a, lambda_b])
        between = int(((fit.wavelengths > lo) & (fit.wavelengths < hi)).sum())
        row = [data.time[i], data.temperature[i], fit.lambda_c, pa, pb, abs(pb - pa), pb - pa,
               fit.rms, len(fit.wavelengths), between]
        if local:
            la = local_fit(data.wavelengths, data.spectra[i], fit, lambda_a, cycles, smooth_window)
            lb = local_fit(data.wavelengths, data.spectra[i], fit, lambda_b, cycles, smooth_window)
            ok = la is not None and lb is not None and la.converged and lb.converged
            # both wavelengths refined, or neither: a refined φ carries a small systematic offset
            # against the cubic, and mixing one of each would put that offset into the readout
            qa, qb = (la.phi, lb.phi) if ok else (pa, pb)
            row[3:7] = [qa, qb, abs(qb - qa), qb - qa]
            row += [pa, pb, abs(pb - pa),
                    la.t1 if la is not None else np.nan, la.t2 if la is not None else np.nan,
                    lb.t1 if lb is not None else np.nan, lb.t2 if lb is not None else np.nan,
                    float(np.mean([f.rms for f in (la, lb) if f is not None]))
                    if (la is not None or lb is not None) else np.nan,
                    ok]
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


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
