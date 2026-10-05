#!/usr/bin/env python3
"""SIMULATED stepped-temperature demo, clearly labeled as such.

The 2024-10-17 dataset is a *continuous* cooldown, so no zoom level shows
staircase steps -- the staircase in the advisor's Igor graph comes from the
experiment itself (the setpoint is changed in discrete steps and held).
This script shows what such an experiment WOULD look like through our
pipeline, using the real calibration measured from the data
(+359.3 pm/degC for the dip near 1542 nm).  It is a demo, not data.

Usage: python scripts/simulate_staircase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sms_analysis.igor_style import plot_igor_tracking  # noqa: E402

SENSITIVITY_NM_PER_C = 0.3593   # measured on the 2024-10-17 run (data/data.csv) (dip@1542.3)
LAMBDA_AT_23C = 1534.9          # nm, from the same fit
STEP_LEVELS_C = [23, 27, 31, 35, 39, 43]
STEP_HOLD_S = 300               # hold each level 5 minutes (like the papers)
THERMAL_TAU_S = 25.0            # water-bath settling time constant
DT_S = 2.0
NOISE_TEMP_C = 0.02
NOISE_WL_NM = 0.02              # tracking noise seen in the real data

rng = np.random.default_rng(42)

time = np.arange(0, STEP_HOLD_S * len(STEP_LEVELS_C), DT_S)
setpoint = np.repeat(STEP_LEVELS_C, int(STEP_HOLD_S / DT_S))

# First-order thermal response of the bath to each setpoint change.
temp = np.empty_like(time, dtype=float)
temp[0] = STEP_LEVELS_C[0]
alpha = DT_S / THERMAL_TAU_S
for i in range(1, len(time)):
    temp[i] = temp[i - 1] + alpha * (setpoint[i] - temp[i - 1])
temp += rng.normal(0, NOISE_TEMP_C, temp.shape)

wavelength = (LAMBDA_AT_23C + SENSITIVITY_NM_PER_C * (temp - 23.0)
              + rng.normal(0, NOISE_WL_NM, temp.shape))

track = pd.DataFrame({"time": time, "temperature": temp,
                      "wavelength": wavelength, "value": np.nan})

fig, ax, ax2 = plot_igor_tracking(track, wl_ticks=2, temp_ticks=5,
                                  time_ticks=500, temp_limits=(20, 45))
ax.set_title("SIMULATED stepped-temperature run "
             "(calibration +359 pm/°C from real 2024-10-17 data)",
             fontsize=10)
out = Path(__file__).resolve().parents[1] / "figures" / "fig8_staircase_simulated.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
