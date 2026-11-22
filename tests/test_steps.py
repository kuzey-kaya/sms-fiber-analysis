"""Plateau detection and the hand-entered step condition (load runs without a load column)."""

from __future__ import annotations

import numpy as np
import pytest

from sms_analysis import fit_sensitivity, track_all_extrema
from sms_analysis.steps import (baseline_drift, classify_plateaus, default_step_values, detect_plateaus,
                                step_condition)


@pytest.fixture(scope="module")
def plateaus(steps):
    data, shift = steps
    tracks = track_all_extrema(data, width_aware=True, max_step=None)
    tr = tracks["dip@1524.9"]
    return tr, detect_plateaus(tr["time"].to_numpy(), tr["wavelength"].to_numpy()), shift


def test_plateaus_match_the_simulated_steps(plateaus):
    tr, p, shift = plateaus
    assert len(p) == 5
    assert p["start"].tolist() == [0, 40, 80, 120, 160]
    assert p["end"].tolist() == [39, 79, 119, 159, 199]
    assert np.allclose(p["shift"], [0, 3, 0, 6, 0], atol=0.05)
    assert classify_plateaus(p).tolist() == [False, True, False, True, False]


def test_step_condition_and_sensitivity(plateaus):
    tr, p, shift = plateaus
    values = default_step_values(p, start=50, increment=50)          # 0, 50, 0, 100, 0
    assert values.tolist() == [0, 50, 0, 100, 0]
    cond = step_condition(len(tr), p, values, settle=2)
    assert np.isnan(cond[[0, 1, 38, 39, 40, 41]]).all()             # settle rows blanked
    assert cond[2] == 0 and cond[50] == 50 and cond[150] == 100
    fit = fit_sensitivity(tr.assign(temperature=cond))
    assert fit.slope * 1e3 == pytest.approx(60.0, abs=1.0)           # 3 nm per 50 g = 60 pm/g
    assert fit.r_squared > 0.999


def test_baseline_drift_is_flat_without_drift(plateaus):
    tr, p, shift = plateaus
    bl = baseline_drift(len(tr), p, classify_plateaus(p))
    assert np.nanstd(bl) < 0.05


def test_cooldown_has_no_steps(tracks):
    """A continuous ramp must not be mistaken for a stepped run."""
    tr = tracks["dip@1542.3"]
    p = detect_plateaus(tr["time"].to_numpy(), tr["wavelength"].to_numpy())
    assert len(p) == 1


def test_a_fast_ramp_is_not_a_load_step_run(steps):
    """2026-11-19, the advisor's 90 -> 18 degC cooldown: at 1 degC per spectrum the fringe moved more
    than the jump threshold between spectra, the start was cut into two short "plateaus", the other
    1389 spectra (a 12 nm drift) became the third, and the app asked for masses. Plateaus must be flat."""
    import numpy as np
    from sms_analysis.auto import choose
    from sms_analysis.steps import detect_plateaus, classify_plateaus, is_step_run, plateau_flatness
    from sms_analysis.synthetic import simulate_run

    # an exponential cooldown seen by one fringe: fast at first, then a long slow tail
    t = np.arange(1400) * 33.0
    w = 1558.0 + 14.0 * np.exp(-t / 6000.0) + np.random.default_rng(0).normal(0, 0.005, t.size)
    p = detect_plateaus(t, w)
    if len(p) >= 3 and classify_plateaus(p).any():          # what the old rule accepted
        assert plateau_flatness(p) > 0.2
    assert not is_step_run(p)
    # a simulated ramp that the old rule took for steps (poor lock with the standard tracker)
    ramp, _ = simulate_run(n_frames=40, common_rad=8 * np.pi, tilt_rad=0.0, noise=0.001, seed=1)
    assert choose(ramp).plateaus is None
    # real staircases stay staircases
    data, _ = steps
    stepped = choose(data)
    assert stepped.plateaus is not None and plateau_flatness(stepped.plateaus) < 0.1
    assert not is_step_run(None) and not is_step_run(p.iloc[:2])


def test_the_strain_run_is_still_a_load_step_run():
    from pathlib import Path
    from sms_analysis import load_sms_csv
    from sms_analysis.auto import choose
    from sms_analysis.steps import plateau_flatness
    strain = choose(load_sms_csv(Path(__file__).resolve().parents[1] / "data" / "strain.csv"))
    assert strain.plateaus is not None and len(strain.plateaus) == 17
    assert plateau_flatness(strain.plateaus) < 0.05
