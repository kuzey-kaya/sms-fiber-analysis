"""Tables the app hands out as CSV: one name for the condition column, the same values in every file.

Inside the package a track's condition column is always called ``temperature`` (the first run was
a cooldown). On a load run that column holds the mass, and a file that says ``temperature`` over
a column of grams is wrong (2026-11-16, found on the Export tab with the strain run: the wide file
said ``condition``, the long one ``temperature``, and the two disagreed on the frames where a load
is being handled). Here the column is named after the condition, and every file takes the
condition from the same array.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

RESERVED = ("time", "wavelength", "value", "fringe")


def condition_column(name: str | None, has_condition: bool = True) -> str:
    """CSV header for the condition: "Temperature" → ``temperature``, "Mass" → ``mass``,
    "Applied strain" → ``applied_strain``; ``condition`` when the run has none or nothing usable
    is left of the name."""
    if not has_condition or not name:
        return "condition"
    slug = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")
    return slug if slug and slug not in RESERVED else "condition"


def wide_table(tracks: dict[str, pd.DataFrame], time, condition, name: str = "condition",
               temperature=None) -> pd.DataFrame:
    """One row per spectrum, one column per fringe: ``time``, the condition under ``name``, then the
    fringes. ``temperature`` (the file's own thermometer column on a load run) goes in front of the
    condition unless the condition is the temperature itself."""
    out = pd.DataFrame({"time": np.asarray(time, float)})
    if temperature is not None and name != "temperature":
        out["temperature"] = np.asarray(temperature, float)
    out[name] = np.asarray(condition, float)
    for label, track in tracks.items():
        out[label] = track["wavelength"].to_numpy(float)
    return out


def long_table(tracks: dict[str, pd.DataFrame], name: str = "condition") -> pd.DataFrame:
    """One row per fringe and spectrum: the tracks stacked, their ``temperature`` column renamed to
    ``name`` and a ``fringe`` column added."""
    if not tracks:
        return pd.DataFrame()
    return pd.concat([t.assign(fringe=label) for label, t in tracks.items()], ignore_index=True) \
        .rename(columns={"temperature": name})


def frame_csv(frame: pd.DataFrame, name: str = "condition", condition=None) -> str:
    """CSV text of a per-spectrum table whose ``temperature`` column holds the run's condition: the
    column is renamed to ``name``; ``condition`` (e.g. the step values with the settle rows blank)
    replaces its values when given."""
    if "temperature" not in frame.columns:
        return frame.to_csv(index=False)
    if condition is not None:
        frame = frame.assign(temperature=np.asarray(condition, float))
    return frame.rename(columns={"temperature": name}).to_csv(index=False)
