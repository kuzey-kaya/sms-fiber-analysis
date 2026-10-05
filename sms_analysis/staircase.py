"""Step-averaged ("staircase") representation of a tracked feature.

The continuous cooldown is divided into temperature bins; within each bin the
tracked wavelength and the temperature are averaged.  Drawing those bin means
as flat plateaus against time turns the smooth run into the familiar
staircase view of the Igor graphs -- built purely from the measured data
(no simulation).  Because the cooldown is exponential, the plateaus get
longer as the run progresses, so the steps are not evenly spaced in time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .igor_style import IGOR_BLUE, IGOR_RED, plot_igor_tracking


def staircase_bin(track: pd.DataFrame, step_c: float = 3.0) -> pd.DataFrame:
    """Average the tracked wavelength inside consecutive temperature bins.

    Parameters
    ----------
    track : output of :func:`~sms_analysis.tracking.track_feature`.
    step_c : temperature bin width in deg C (one bin -> one plateau).

    Returns
    -------
    DataFrame with one row per plateau: t_start, t_end (s), temp_mean,
    wavelength_mean, wavelength_std, n_points.
    """
    ok = track.dropna(subset=["wavelength"]).reset_index(drop=True)
    t_hi = np.ceil(ok["temperature"].max() / step_c) * step_c
    t_lo = np.floor(ok["temperature"].min() / step_c) * step_c
    edges = np.arange(t_hi, t_lo - step_c / 2, -step_c)  # descending for a cooldown

    rows = []
    for hi, lo in zip(edges[:-1], edges[1:]):
        sel = ok[(ok["temperature"] <= hi) & (ok["temperature"] > lo)]
        if len(sel) < 3:
            continue
        rows.append(dict(
            t_start=sel["time"].min(), t_end=sel["time"].max(),
            temp_mean=sel["temperature"].mean(),
            wavelength_mean=sel["wavelength"].mean(),
            wavelength_std=sel["wavelength"].std(),
            n_points=len(sel),
        ))
    return pd.DataFrame(rows)


def plot_staircase(
    track: pd.DataFrame,
    steps: pd.DataFrame,
    show_raw: bool = True,
    step_temperature: bool = True,
    **igor_kwargs,
):
    """Igor-style staircase figure from the binned plateaus.

    The red trace is drawn as flat plateaus (bin means) joined by vertical
    steps; the blue temperature is drawn the same way so the two staircases
    correspond bin by bin.  ``show_raw`` underlays the raw tracked data as a
    faint line so it stays visible that the steps are averages of real data.
    """
    fig, ax, ax2 = plot_igor_tracking(track, **igor_kwargs)
    # Replace the continuous red trace with the staircase representation.
    for line in list(ax.lines):
        line.remove()
    if step_temperature:
        for line in list(ax2.lines):
            line.remove()

    if show_raw:
        ok = track.dropna(subset=["wavelength"])
        ax.plot(ok["time"], ok["wavelength"], color=IGOR_RED, lw=0.7, alpha=0.25)
        if step_temperature:
            ax2.plot(track["time"], track["temperature"], color=IGOR_BLUE,
                     lw=0.7, alpha=0.25)

    # Piecewise-constant curves: hold each plateau, jump at the bin boundary.
    t_edges = np.r_[steps["t_start"].to_numpy(), steps["t_end"].iloc[-1]]
    ax.step(t_edges, np.r_[steps["wavelength_mean"], steps["wavelength_mean"].iloc[-1]],
            where="post", color=IGOR_RED, lw=1.5)
    if step_temperature:
        ax2.step(t_edges, np.r_[steps["temp_mean"], steps["temp_mean"].iloc[-1]],
                 where="post", color=IGOR_BLUE, lw=1.5)
    return fig, ax, ax2
