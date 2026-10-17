"""The fringe-width fit (``width_aware``) for wide fringes and step changes, e.g. strain/load runs."""

from __future__ import annotations

import numpy as np
import pytest

from sms_analysis import SpectraSet, fit_sensitivity, track_all_extrema
from sms_analysis.tracking import flicker_fraction, half_periods, refine_width_fit


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


def test_flicker_fraction():
    smooth_track = 1500 + 0.001 * np.arange(200)
    assert flicker_fraction(smooth_track) == 0.0
    flick = smooth_track.copy()
    flick[::2] += 1.0                                    # jumps between two samples on every frame
    assert flicker_fraction(flick) > 90
    one_step = smooth_track.copy()
    one_step[100:] += 5.0                                # a load step is one jump, not flicker
    assert flicker_fraction(one_step) < 1


def test_no_fringe_flickers_on_the_cooldown(tracks):
    assert all(flicker_fraction(tr["wavelength"].to_numpy()) <= 5 for tr in tracks.values())


def test_lambda_c_peak_flickers_on_the_strain_run():
    from pathlib import Path
    from sms_analysis import load_sms_csv
    p = Path(__file__).resolve().parents[1] / "data" / "strain.csv"
    if not p.exists():
        pytest.skip("data/strain.csv not present")
    tr = track_all_extrema(load_sms_csv(p), width_aware=True, max_step=None)
    flick = {k for k, v in tr.items() if flicker_fraction(v["wavelength"].to_numpy()) > 5}
    assert flick == {"peak@1529.8"}


def test_refine_width_fit_keeps_the_count_and_moves_little(data):
    from sms_analysis import detect_extrema
    ext = detect_extrema(data.wavelengths, data.spectra[0])
    ref = refine_width_fit(data.wavelengths, data.spectra[0], ext)
    assert len(ref.peak_wavelengths) == len(ext.peak_wavelengths)
    assert len(ref.dip_wavelengths) == len(ext.dip_wavelengths)
    shift = np.abs(ref.dip_wavelengths - ext.dip_wavelengths)
    assert np.median(shift) < 0.1 and shift.max() < 1.0   # the broad asymmetric dip at λc moves most
