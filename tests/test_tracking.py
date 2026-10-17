"""The fringe-width fit (``width_aware``) for wide fringes and step changes, e.g. strain/load runs."""

from __future__ import annotations

import numpy as np
import pytest

from sms_analysis import SpectraSet, fit_sensitivity, track_all_extrema
from sms_analysis.tracking import half_periods


def test_half_periods():
    assert np.allclose(half_periods(np.array([10.0, 0.0, 4.0, 13.0])), [3.0, 4.0, 4.0, 3.0])


def _error_and_lock(tracks, shift):
    errs, locks = [], []
    for tr in tracks.values():
        w = tr["wavelength"].to_numpy()
        locks.append(np.isfinite(w).mean())
        d = w - w[0] - shift
        errs.append(np.nanmax(np.abs(d)) if np.isfinite(d).any() else np.inf)
    return max(errs), min(locks)


def test_width_fit_follows_steps(steps):
    data, shift = steps
    tracks = track_all_extrema(data, width_aware=True, max_step=None)
    edge = max(tracks, key=lambda k: float(k.split("@")[1]))    # pushed off the band by the 6 nm step
    err, lock = _error_and_lock({k: v for k, v in tracks.items() if k != edge}, shift)
    assert lock == 1.0
    assert err < 0.25                       # nm, every fringe, every frame (noise alone, no hops)
    d = tracks[edge]["wavelength"].to_numpy() - (1645.0 + shift)   # true dip position
    assert np.isnan(d[-80:-40]).all()                              # lost while off the band ...
    assert np.nanmax(np.abs(d)) < 0.25                             # ... and never wrong


def test_standard_tracker_loses_steps(steps):
    """Why the advisory exists: the fixed ±1.5 nm window cannot follow a 6 nm step."""
    data, shift = steps
    _, lock = _error_and_lock(track_all_extrema(data), shift)
    assert lock < 0.9


def test_width_fit_on_the_cooldown_run(tracks, data):
    """Same physics on the shipped run (the published numbers stay on the standard tracker)."""
    wide = track_all_extrema(data, width_aware=True)
    assert fit_sensitivity(wide["peak@1493.4"]).slope * 1e3 == pytest.approx(-140.1, abs=0.5)
    assert fit_sensitivity(wide["dip@1542.3"]).slope * 1e3 == pytest.approx(364.5, abs=0.5)
