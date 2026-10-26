"""The automatic choices FringeLab makes when a file is opened, for the scripts too.

Added 2026-10-26 when the scripts were run on the strain file: ``staircase_views.py`` tracked a
fringe that does not exist in that file and ``igor_findpeak.py`` demanded a range on the command
line, while the app already decides both on its own.  Same rules here as in ``app.py``:

* tracker — the fringe-width fit when the standard tracker keeps fewer than half of the fringes
  locked ≥ 90 % (or its mean lock is below 80 %) or the spectra are noisy;
* reference fringe — among the fringes locked ≥ 90 % and not flickering, the most sensitive one
  when the condition spans something, otherwise the one with the cleanest steps (largest
  excursion relative to its frame-to-frame jitter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import fit_sensitivity
from .io import SpectraSet
from .steps import classify_plateaus, default_step_values, detect_plateaus, robust_jitter, step_condition
from .tracking import flicker_fraction, track_all_extrema

FLICKER_LIMIT_PCT = 5.0   # as in app.py
NOISE_LIMIT = 0.004      # transmittance; the strain run sits at 0.008, the cooldown at 0.0013


def spectral_noise(data: SpectraSet) -> float:
    """Noise level of the spectra from the second difference along wavelength (robust)."""
    x = np.asarray(data.spectra, float)
    if x.shape[1] < 3:
        return 0.0
    r = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    return float(np.nanstd(r) / np.sqrt(6.0))        # same formula as app.py


@dataclass
class AutoChoice:
    tracks: dict[str, pd.DataFrame]
    width_aware: bool
    fringe: str
    table: pd.DataFrame
    plateaus: pd.DataFrame | None = None      # load-step run: the plateaus of the reference fringe
    step_values: np.ndarray | None = None     # one value per plateau (0 / 50 / 100 … g until told otherwise)
    step_cond: np.ndarray | None = None       # per-spectrum condition from them, settle rows blanked

    @property
    def is_steps(self) -> bool:
        return self.plateaus is not None

    @property
    def track(self) -> pd.DataFrame:
        return self.tracks[self.fringe]

    @property
    def start_nm(self) -> float:
        return float(self.fringe.split("@")[1])

    @property
    def kind(self) -> str:
        return self.fringe.split("@")[0]


def lock_table(tracks: dict[str, pd.DataFrame], has_cond: bool) -> pd.DataFrame:
    rows = []
    for k, tr in tracks.items():
        w = tr["wavelength"].to_numpy(float)
        ok = np.isfinite(w)
        row = dict(fringe=k, locked_pct=100.0 * ok.mean(), flicker_pct=flicker_fraction(w),
                   excursion_nm=float(np.nanmax(w) - np.nanmin(w)) if ok.any() else np.nan,
                   jitter_nm=robust_jitter(w) if ok.sum() > 3 else np.nan, sensitivity=np.nan, r_squared=np.nan)
        if has_cond and ok.sum() >= 3:
            try:
                f = fit_sensitivity(tr)
                row.update(sensitivity=f.slope * 1e3, r_squared=f.r_squared)
            except ValueError:
                pass
        rows.append(row)
    t = pd.DataFrame(rows)
    t["stable"] = t["flicker_pct"] <= FLICKER_LIMIT_PCT
    return t


def poor_lock(table: pd.DataFrame, min_lock: float = 90.0) -> bool:
    if table.empty:
        return True
    return bool(table["locked_pct"].mean() < 80.0 or (table["locked_pct"] >= min_lock).mean() < 0.5)


def choose(data: SpectraSet, prominence: float = 0.02, distance: int = 5, smooth_window: int = 11,
           search_window: float = 1.5, max_step: float = 0.5, width_aware: bool | None = None,
           step_start: float = 50.0, step_increment: float = 50.0, settle: int = 2) -> AutoChoice:
    """Track every fringe with the tracker the app would pick, name the reference fringe and, on a
    load-step run, build the per-spectrum step condition (``step_start`` + k·``step_increment``)."""
    has_cond = bool(data.has_temperature)
    cond_span = float(np.nanmax(data.temperature) - np.nanmin(data.temperature)) if has_cond else 0.0
    if width_aware is None:
        std = track_all_extrema(data, prominence=prominence, distance=distance, smooth_window=smooth_window,
                                search_window=search_window, max_step=max_step)
        width_aware = poor_lock(lock_table(std, has_cond)) or spectral_noise(data) > NOISE_LIMIT
        tracks = std if not width_aware else None
    else:
        tracks = None
    if tracks is None:
        tracks = track_all_extrema(data, prominence=prominence, distance=distance, smooth_window=smooth_window,
                                   search_window=search_window, max_step=None if width_aware else max_step,
                                   width_aware=width_aware)
    table = lock_table(tracks, has_cond)
    good = table[(table["locked_pct"] >= 90.0) & table["stable"]]
    if good.empty:
        good = table[table["locked_pct"] >= 90.0] if (table["locked_pct"] >= 90.0).any() else table
    if has_cond and cond_span > 1.0 and good["r_squared"].notna().any():
        fit = good[good["r_squared"] >= 0.9]
        pool = fit if not fit.empty else good
        fringe = pool.loc[pool["sensitivity"].abs().idxmax(), "fringe"]
    else:
        score = good["excursion_nm"] / good["jitter_nm"].clip(lower=1e-4)
        fringe = good.loc[score.idxmax(), "fringe"]
    choice = AutoChoice(tracks=tracks, width_aware=bool(width_aware), fringe=str(fringe), table=table)
    if width_aware:
        # a load-step run only when the width-fit tracks really show plateaus separated by jumps
        # (the app's rule): then the condition is one value per plateau, 50 g steps until the
        # advisor says otherwise, with the frames recorded while the load changed left out
        ref = choice.track
        pl = detect_plateaus(ref["time"].to_numpy(float), ref["wavelength"].to_numpy(float))
        if len(pl) >= 3 and classify_plateaus(pl).any():
            vals = default_step_values(pl, step_start, step_increment, 0.0)
            choice.plateaus, choice.step_values = pl, vals
            choice.step_cond = step_condition(len(ref), pl, vals, settle)
    return choice
