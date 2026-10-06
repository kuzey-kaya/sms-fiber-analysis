"""Publication-style figures for SMS sensor analysis.

Conventions follow the group's papers (Optica Sensing Congress 2025):
spectra as transmittance vs wavelength, tracking as wavelength + temperature
vs time (dual axis, wavelength left / temperature right), and calibration as
wavelength vs temperature with a linear fit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import CurvatureTest, SensitivityFit
from .io import SpectraSet
from .peaks import detect_extrema

# Colorblind-safe base colors used consistently across figures.
COLOR_WAVELENGTH = "#C4402F"   # red   -> tracked wavelength / spectra
COLOR_TEMPERATURE = "#2E5FA3"  # blue  -> temperature
COLOR_FIT = "#333333"
SPECTRUM_COLORS = ["#2E5FA3", "#E69F00", "#C4402F", "#4B9B78", "#7B5AA6"]

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def plot_spectrum_with_extrema(
    data: SpectraSet,
    row: int = 0,
    prominence: float = 0.02,
    distance: int = 5,
    ax: plt.Axes | None = None,
):
    """One spectrum with its detected peaks (circles) and dips (triangles)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))
    y = data.spectra[row]
    ext = detect_extrema(data.wavelengths, y, prominence=prominence, distance=distance)

    ax.plot(data.wavelengths, y, color="#666666", lw=1.2,
            label=f"T = {data.temperature[row]:.1f} °C")
    ax.plot(ext.peak_wavelengths, ext.peak_values, "o", ms=6,
            mfc=COLOR_WAVELENGTH, mec="black", mew=0.5, label="Peaks (maxima)")
    ax.plot(ext.dip_wavelengths, ext.dip_values, "v", ms=6,
            mfc=COLOR_TEMPERATURE, mec="black", mew=0.5, label="Dips (minima)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance")
    ax.legend(frameon=False)
    return ax


def plot_spectra_overlay(data: SpectraSet, rows: list[int], ax: plt.Axes | None = None):
    """A few spectra at different temperatures overlaid (like Fig. 1 of the papers)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))
    for color, row in zip(SPECTRUM_COLORS, rows):
        ax.plot(data.wavelengths, data.spectra[row], lw=1.2, color=color,
                label=f"{data.temperature[row]:.1f} °C")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmittance")
    ax.legend(frameon=False, title="Temperature")
    return ax


def plot_spectral_map(data: SpectraSet, ax: plt.Axes | None = None):
    """Heatmap of all spectra: wavelength vs time, transmittance as color.

    The whole experiment in one image -- fringes appear as bright/dark bands
    drifting as the temperature changes; the critical-wavelength region is
    the wide slow band in the middle.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.6))
    mesh = ax.pcolormesh(
        data.wavelengths, data.time / 60.0, data.spectra,
        cmap="viridis", shading="auto", rasterized=True,
    )
    plt.colorbar(mesh, ax=ax, label="Transmittance")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Time (min)")
    ax.grid(False)
    return ax


def plot_tracking(track: pd.DataFrame, title: str = "", ax: plt.Axes | None = None):
    """Tracked wavelength (left axis) and temperature (right axis) vs time.

    Reproduces the Igor-style monitoring graph.  The two y-axes are
    color-coded to their curves so the shared time axis stays readable.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(track["time"], track["wavelength"], color=COLOR_WAVELENGTH, lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracked wavelength (nm)", color=COLOR_WAVELENGTH)
    ax.tick_params(axis="y", labelcolor=COLOR_WAVELENGTH)

    ax2 = ax.twinx()
    ax2.plot(track["time"], track["temperature"], color=COLOR_TEMPERATURE, lw=1.2)
    ax2.set_ylabel("Temperature (°C)", color=COLOR_TEMPERATURE)
    ax2.tick_params(axis="y", labelcolor=COLOR_TEMPERATURE)
    ax2.spines.right.set_visible(True)
    ax2.grid(False)
    if title:
        ax.set_title(title)
    return ax


def plot_correlation(
    track: pd.DataFrame,
    fit: SensitivityFit,
    title: str = "",
    ax: plt.Axes | None = None,
):
    """Tracked wavelength vs temperature with the linear calibration fit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.6))
    ok = track.dropna(subset=["wavelength", "temperature"])
    ax.plot(ok["temperature"], ok["wavelength"], ".", ms=3,
            color=COLOR_WAVELENGTH, alpha=0.5, label="Tracked feature")
    t = np.linspace(ok["temperature"].min(), ok["temperature"].max(), 50)
    ax.plot(t, fit.slope * t + fit.intercept, "-", color=COLOR_FIT, lw=1.5,
            label=f"Fit: {fit.slope * 1e3:+.1f} pm/°C  (R² = {fit.r_squared:.4f})")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Wavelength (nm)")
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    return ax


def plot_sensitivity_map(smap: pd.DataFrame, ax: plt.Axes | None = None):
    """Sensitivity of every tracked fringe vs its wavelength (cf. Chen Fig. 3b).

    Shows why raw peak motion looks irregular: sensitivity magnitude grows
    toward the critical wavelength and the sign can differ across it.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.4))
    for kind, marker in (("peak", "o"), ("dip", "v")):
        sel = smap[smap["kind"] == kind]
        ax.errorbar(
            sel["start_wavelength"], sel["sensitivity_pm_per_C"],
            yerr=sel["stderr_pm_per_C"], fmt=marker, ms=6, lw=0, elinewidth=1,
            mfc=COLOR_WAVELENGTH if kind == "peak" else COLOR_TEMPERATURE,
            mec="black", mew=0.5, ecolor="#999999",
            label=f"{kind.capitalize()}s",
        )
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("Fringe wavelength at start (nm)")
    ax.set_ylabel("Sensitivity (pm/°C)")
    ax.legend(frameon=False)
    return ax


def plot_calibration_models(
    track: pd.DataFrame,
    curv: CurvatureTest,
    title: str = "",
    axes: tuple[plt.Axes, plt.Axes] | None = None,
):
    """Linear vs quadratic calibration of one fringe, with the residuals.

    Top: tracked points with both fits.  Bottom: measured minus fitted
    wavelength for each model, so a systematic bow in the linear residuals
    (curvature near the critical wavelength) is visible at a glance.
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax_fit, ax_res = axes
    lin, quad = curv.linear, curv.quadratic
    ok = quad.residuals(track)
    t = np.linspace(quad.t_min, quad.t_max, 100)
    s_lo, s_hi = curv.sensitivity_at_ends_pm

    ax_fit.plot(ok["temperature"], ok["wavelength"], ".", ms=3,
                color=COLOR_WAVELENGTH, alpha=0.4, label="Tracked feature")
    ax_fit.plot(t, lin.predict(t), "--", color="#888888", lw=1.4,
                label=f"Linear: {lin.coefficients[0] * 1e3:+.1f} pm/°C, RMSE {lin.rmse * 1e3:.0f} pm")
    ax_fit.plot(t, quad.predict(t), "-", color=COLOR_FIT, lw=1.6,
                label=f"Quadratic: {s_lo:+.0f} → {s_hi:+.0f} pm/°C "
                      f"({quad.t_min:.0f}→{quad.t_max:.0f} °C), RMSE {quad.rmse * 1e3:.0f} pm")
    ax_fit.set_ylabel("Wavelength (nm)")
    ax_fit.legend(frameon=False, fontsize=8.5)
    if title:
        ax_fit.set_title(title)

    ax_res.axhline(0, color="#999999", lw=0.8)
    ax_res.plot(ok["temperature"], (ok["wavelength"] - lin.predict(ok["temperature"])) * 1e3,
                ".", ms=3, color="#888888", alpha=0.6, label="Linear residual")
    ax_res.plot(ok["temperature"], ok["residual"] * 1e3, ".", ms=3,
                color=COLOR_TEMPERATURE, alpha=0.6, label="Quadratic residual")
    ax_res.set_xlabel("Temperature (°C)")
    ax_res.set_ylabel("Residual (pm)")
    ax_res.legend(frameon=False, fontsize=8.5, ncol=2)
    return axes
