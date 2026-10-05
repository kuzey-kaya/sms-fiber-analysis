"""Igor Pro-style rendering of the tracking graph.

Reproduces the look of the group's Igor Pro monitoring graphs: white
background, thin black frame, outward ticks with minors, pure red/blue
traces, 'Peak Wavelength (nm)' left axis and 'Temperature, °C' right axis
against 'Time, seconds'.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

IGOR_RED = "#ff0000"
IGOR_BLUE = "#0000ff"


def plot_igor_tracking(
    track: pd.DataFrame,
    wl_ticks: float | None = None,
    temp_ticks: float = 5.0,
    time_ticks: float | None = None,
    wl_limits: tuple[float, float] | None = None,
    temp_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (7.6, 4.1),
):
    """Render a tracked feature the way Igor Pro draws it.

    Wavelength (red) on the left axis, temperature (blue) on the right,
    time in seconds on the bottom.  Tick spacings and limits are chosen
    automatically to mimic the Igor graph's density but can be overridden.
    """
    ok = track.dropna(subset=["wavelength"])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- traces ---------------------------------------------------------
    ax.plot(ok["time"], ok["wavelength"], color=IGOR_RED, lw=1.3)
    ax2 = ax.twinx()
    ax2.plot(track["time"], track["temperature"], color=IGOR_BLUE, lw=1.3)

    # --- axis limits ----------------------------------------------------
    if wl_limits is None:
        lo, hi = ok["wavelength"].min(), ok["wavelength"].max()
        if wl_ticks:
            # Round outward to tick multiples so the frame starts/ends on ticks,
            # the way the Igor graphs are scaled.
            wl_limits = (np.floor(lo / wl_ticks) * wl_ticks,
                         np.ceil(hi / wl_ticks) * wl_ticks)
        else:
            pad = 0.04 * (hi - lo)
            wl_limits = (lo - pad, hi + pad)
    ax.set_ylim(*wl_limits)

    if temp_limits is None:
        t_lo = np.floor(track["temperature"].min() / temp_ticks) * temp_ticks
        t_hi = np.ceil(track["temperature"].max() / temp_ticks) * temp_ticks
        temp_limits = (t_lo, t_hi)
    ax2.set_ylim(*temp_limits)
    ax.set_xlim(track["time"].min(), track["time"].max())

    # --- Igor-like frame and ticks --------------------------------------
    for a in (ax, ax2):
        for s in a.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.8)
            s.set_color("black")
        a.grid(False)

    if wl_ticks:
        ax.yaxis.set_major_locator(MultipleLocator(wl_ticks))
    ax2.yaxis.set_major_locator(MultipleLocator(temp_ticks))
    if time_ticks:
        ax.xaxis.set_major_locator(MultipleLocator(time_ticks))

    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    tick_kw = dict(direction="out", top=False, colors="black")
    ax.tick_params(which="major", length=5, width=0.8, labelsize=11, **tick_kw)
    ax.tick_params(which="minor", length=2.5, width=0.6, **tick_kw)
    ax2.tick_params(which="major", length=5, width=0.8, labelsize=11,
                    direction="out", colors="black")
    ax2.tick_params(which="minor", length=2.5, width=0.6,
                    direction="out", colors="black")

    # --- labels (Igor wording) ------------------------------------------
    label_kw = dict(fontsize=13, fontfamily="sans-serif", color="black")
    ax.set_xlabel("Time, seconds", **label_kw)
    ax.set_ylabel("Peak Wavelength (nm)", **label_kw)
    ax2.set_ylabel("Temperature, °C", rotation=270, labelpad=16, **label_kw)

    fig.tight_layout()
    return fig, ax, ax2
