"""Correlation between tracked wavelength and temperature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SensitivityFit:
    """Linear fit lambda = slope * T + intercept for one tracked feature."""

    slope: float        # nm per degC (sensitivity)
    intercept: float    # nm
    r_value: float
    r_squared: float
    stderr: float       # standard error of the slope
    n_points: int

    def __repr__(self) -> str:
        return (
            f"sensitivity = {self.slope * 1e3:+.1f} +/- {self.stderr * 1e3:.1f} pm/degC "
            f"(R^2 = {self.r_squared:.4f}, n = {self.n_points})"
        )


def fit_sensitivity(track: pd.DataFrame) -> SensitivityFit:
    """Fit tracked wavelength vs temperature with a straight line.

    NaN rows (lost tracking lock) are dropped before fitting.
    """
    ok = track.dropna(subset=["wavelength", "temperature"])
    if len(ok) < 3:
        raise ValueError("Not enough valid points to fit")
    res = stats.linregress(ok["temperature"], ok["wavelength"])
    return SensitivityFit(
        slope=float(res.slope),
        intercept=float(res.intercept),
        r_value=float(res.rvalue),
        r_squared=float(res.rvalue**2),
        stderr=float(res.stderr),
        n_points=len(ok),
    )


@dataclass
class CalibrationFit:
    """Polynomial calibration lambda(T) = sum(c_k * T^k) for one tracked feature.

    ``degree=1`` is the usual linear calibration; ``degree=2`` adds the
    curvature that appears close to the critical wavelength, where the
    sensitivity itself depends on how far the fringe has moved.
    """

    degree: int
    coefficients: np.ndarray   # highest power first, as numpy.polyfit returns them
    r_squared: float
    rmse: float                # nm, root-mean-square residual
    n_points: int
    t_min: float               # fitted temperature range
    t_max: float

    def predict(self, t) -> np.ndarray:
        return np.polyval(self.coefficients, np.asarray(t, dtype=float))

    def sensitivity(self, t) -> np.ndarray:
        """Local sensitivity d(lambda)/dT in nm/degC at temperature(s) ``t``."""
        return np.polyval(np.polyder(self.coefficients), np.asarray(t, dtype=float))

    def residuals(self, track: pd.DataFrame) -> pd.DataFrame:
        """Rows of ``track`` with a ``residual`` column (nm, measured - fitted)."""
        ok = track.dropna(subset=["wavelength", "temperature"]).copy()
        ok["residual"] = ok["wavelength"] - self.predict(ok["temperature"])
        return ok


def fit_calibration(track: pd.DataFrame, degree: int = 2) -> CalibrationFit:
    """Least-squares polynomial fit of tracked wavelength vs temperature.

    NaN rows (lost tracking lock) are dropped.  ``degree=1`` reproduces
    :func:`fit_sensitivity` (same slope and R^2); ``degree=2`` is the
    curved calibration.
    """
    ok = track.dropna(subset=["wavelength", "temperature"])
    if len(ok) < degree + 2:
        raise ValueError("Not enough valid points to fit")
    t = ok["temperature"].to_numpy(float)
    w = ok["wavelength"].to_numpy(float)
    coef = np.polyfit(t, w, degree)
    resid = w - np.polyval(coef, t)
    ss_tot = float(np.sum((w - w.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return CalibrationFit(degree=degree, coefficients=coef, r_squared=r2,
                          rmse=float(np.sqrt(np.mean(resid**2))), n_points=len(ok),
                          t_min=float(t.min()), t_max=float(t.max()))


@dataclass
class CurvatureTest:
    """Linear vs quadratic calibration of the same track."""

    linear: CalibrationFit
    quadratic: CalibrationFit
    f_statistic: float
    p_value: float             # nested-model F-test for the quadratic term
    rmse_improvement: float    # fraction by which the quadratic fit lowers the RMSE

    @property
    def curvature_pm_per_C2(self) -> float:
        """d^2(lambda)/dT^2 in pm/degC^2 (twice the quadratic coefficient)."""
        return float(2.0 * self.quadratic.coefficients[0] * 1e3)

    @property
    def sensitivity_at_ends_pm(self) -> tuple[float, float]:
        """Local sensitivity (pm/degC) at the lowest and highest fitted temperature."""
        q = self.quadratic
        return (float(q.sensitivity(q.t_min) * 1e3), float(q.sensitivity(q.t_max) * 1e3))


def curvature_test(track: pd.DataFrame) -> CurvatureTest:
    """Fit a line and a parabola to the same track and compare them.

    The F-test treats the points as independent; consecutive spectra are
    strongly correlated, so its p-value overstates the evidence.  Use the
    RMSE improvement and the sensitivity at the two ends as the practical
    measure of curvature; the p-value only ranks fringes.
    """
    lin = fit_calibration(track, degree=1)
    quad = fit_calibration(track, degree=2)
    n = quad.n_points
    rss_lin, rss_quad = lin.rmse**2 * n, quad.rmse**2 * n
    if n > 3 and rss_quad > 0:
        f = (rss_lin - rss_quad) / (rss_quad / (n - 3))
        p = float(stats.f.sf(f, 1, n - 3))
    else:
        f, p = float("nan"), float("nan")
    improvement = 1.0 - quad.rmse / lin.rmse if lin.rmse > 0 else 0.0
    return CurvatureTest(linear=lin, quadratic=quad, f_statistic=float(f), p_value=p,
                         rmse_improvement=float(improvement))


def sensitivity_map(tracks: dict[str, pd.DataFrame], min_r2: float = 0.0) -> pd.DataFrame:
    """Fit every tracked feature and tabulate sensitivity vs wavelength.

    Returns a DataFrame (one row per feature) with the feature's starting
    wavelength, kind, sensitivity in nm/degC and pm/degC, R^2 and the number
    of valid points, followed by the curved-calibration columns from
    :func:`curvature_test` (RMSE of both models in pm, curvature in
    pm/degC^2, local sensitivity at the coldest and warmest fitted
    temperature, F-test p-value).  Set ``min_r2`` to filter out
    badly-tracked features.
    """
    rows = []
    for label, track in tracks.items():
        kind, w0 = label.split("@")
        try:
            fit = fit_sensitivity(track)
            curv = curvature_test(track)
        except ValueError:
            continue
        s_lo, s_hi = curv.sensitivity_at_ends_pm
        rows.append(
            dict(
                label=label,
                kind=kind,
                start_wavelength=float(w0),
                sensitivity_nm_per_C=fit.slope,
                sensitivity_pm_per_C=fit.slope * 1e3,
                r_squared=fit.r_squared,
                stderr_pm_per_C=fit.stderr * 1e3,
                n_points=fit.n_points,
                rmse_linear_pm=curv.linear.rmse * 1e3,
                rmse_quadratic_pm=curv.quadratic.rmse * 1e3,
                curvature_pm_per_C2=curv.curvature_pm_per_C2,
                sensitivity_at_tmin_pm_per_C=s_lo,
                sensitivity_at_tmax_pm_per_C=s_hi,
                curvature_p_value=curv.p_value,
            )
        )
    columns = ["label", "kind", "start_wavelength", "sensitivity_nm_per_C",
               "sensitivity_pm_per_C", "r_squared", "stderr_pm_per_C", "n_points",
               "rmse_linear_pm", "rmse_quadratic_pm", "curvature_pm_per_C2",
               "sensitivity_at_tmin_pm_per_C", "sensitivity_at_tmax_pm_per_C",
               "curvature_p_value"]
    if not rows:  # e.g. a file without a temperature column
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns).sort_values("start_wavelength").reset_index(drop=True)
    return out[out["r_squared"] >= min_r2].reset_index(drop=True)
