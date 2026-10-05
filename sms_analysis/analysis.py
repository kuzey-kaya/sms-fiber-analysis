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


def sensitivity_map(tracks: dict[str, pd.DataFrame], min_r2: float = 0.0) -> pd.DataFrame:
    """Fit every tracked feature and tabulate sensitivity vs wavelength.

    Returns a DataFrame (one row per feature) with the feature's starting
    wavelength, kind, sensitivity in nm/degC and pm/degC, R^2 and the number
    of valid points.  Set ``min_r2`` to filter out badly-tracked features.
    """
    rows = []
    for label, track in tracks.items():
        kind, w0 = label.split("@")
        try:
            fit = fit_sensitivity(track)
        except ValueError:
            continue
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
            )
        )
    columns = ["label", "kind", "start_wavelength", "sensitivity_nm_per_C",
               "sensitivity_pm_per_C", "r_squared", "stderr_pm_per_C", "n_points"]
    if not rows:  # e.g. a file without a temperature column
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows, columns=columns).sort_values("start_wavelength").reset_index(drop=True)
    return out[out["r_squared"] >= min_r2].reset_index(drop=True)
