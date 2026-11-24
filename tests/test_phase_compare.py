"""Comparing phase readouts (2026-11-22, the advisor through the owner: several wavelength pairs
on one graph, what the curve fit does to each, and what its settings change)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sms_analysis.phase import fit_phase, local_fit, pair_readout, phase_series, wavelength_phase
from sms_analysis.phase_compare import (compare_pairs, deviation, parameter_study, plateau_groups,
                                        readout_metrics, study_spread, study_summary)
from sms_analysis.synthetic import simulate_run


def test_readout_metrics_separate_the_bow_from_the_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(20.0, 40.0, 400)
    y = 0.05 * x + 0.002 * (x - 30.0) ** 2 + rng.normal(0, 0.003, x.size)
    m = readout_metrics(y, x)
    assert m["slope"] == pytest.approx(0.05, abs=2e-4)
    assert m["span"] == pytest.approx(1.0, abs=0.01)
    # a parabola c·(x − x̄)² over a length L sits c·L²/12 above its best line in the middle and
    # c·L²/6 below it at the ends: the bow is the larger of the two, here 0.002·400/6 = 0.133
    assert m["bow"] == pytest.approx(0.1333, abs=0.005)
    assert m["bow_pct"] == pytest.approx(13.3, abs=0.6)
    assert m["noise"] == pytest.approx(0.003, abs=5e-4)            # what the parabola leaves
    assert m["scatter"] > 10 * m["noise"]                          # the straight line counts the bow as scatter
    assert m["frame_noise"] == pytest.approx(0.003, abs=8e-4)
    assert m["resolution"] == pytest.approx(m["scatter"] / m["slope"], rel=1e-9)
    np.testing.assert_allclose(np.nanmean(deviation(y, x)), 0.0, atol=1e-9)
    empty = readout_metrics(np.full(5, np.nan), np.arange(5.0))
    assert empty["n"] == 0 and np.isnan(empty["slope"])


def test_plateau_noise_ignores_the_steps():
    rng = np.random.default_rng(1)
    level = np.repeat([0.0, 0.5, 0.0, 1.0], 50)
    y = level + rng.normal(0, 0.004, level.size)
    plateaus = pd.DataFrame({"start": [0, 50, 100, 150], "end": [49, 99, 149, 199]})
    groups = plateau_groups(len(y), plateaus)
    assert set(np.unique(groups)) == {-1, 0, 1, 2, 3} and (groups[:2] == -1).all()     # settle rows left out
    m = readout_metrics(y, np.repeat([0.0, 50.0, 0.0, 100.0], 50), groups)
    assert m["plateau_noise"] == pytest.approx(0.004, abs=8e-4)
    assert m["frame_noise"] == pytest.approx(0.004, abs=1e-3)      # the three jumps are outliers, not noise
    assert m["slope"] == pytest.approx(0.01, abs=2e-4)
    assert plateau_groups(10, None) is None


def test_a_pair_readout_is_not_folded_when_it_passes_through_zero():
    """SIMULATED finding of 2026-11-22: |φb − φa| of a pair that straddles λc turned into a V."""
    n = 50
    a = pd.DataFrame({"frame": np.arange(n), "phi_cubic": np.zeros(n), "phi_local": np.zeros(n),
                      "local_ok": np.ones(n, bool)})
    b = a.assign(phi_cubic=np.linspace(-0.5, 2.0, n), phi_local=np.linspace(-0.5, 2.0, n))
    r = pair_readout(a, b)
    assert np.all(np.diff(r["cubic"]) > 0) and r["cubic"].iloc[0] == pytest.approx(-0.5)
    r = pair_readout(b, a)                                         # the other way round: still one straight ramp
    assert np.all(np.diff(r["cubic"]) > 0) and r["local"].iloc[-1] == pytest.approx(2.0)


def test_compare_pairs_on_the_cooldown(data, fits):
    rows = np.arange(0, len(data), 5)
    readouts, table = compare_pairs(data, fits, [(1545.0, 1570.0), (1540.0, 1565.0)], data.temperature, rows=rows)
    loc = table[table["method"] == "local fit"].set_index("pair")
    cub = table[table["method"] == "cubic only"].set_index("pair")
    # the paper's pair: every frame refined, the scatter about the line several times smaller
    assert loc.loc["1545/1570", "refined_pct"] == 100.0
    assert loc.loc["1545/1570", "gain"] > 3
    assert loc.loc["1545/1570", "slope"] * 1e3 == pytest.approx(-32.9, abs=0.5)
    assert cub.loc["1545/1570", "slope"] * 1e3 == pytest.approx(-31.6, abs=0.5)
    # 5 nm closer to λc the window of the local fit reaches it: the pair keeps the cubic and its noise
    assert loc.loc["1540/1565", "refined_pct"] == 0.0 and loc.loc["1540/1565", "gain"] == pytest.approx(1.0)
    assert loc.loc["1540/1565", "scatter"] > 2 * loc.loc["1545/1570", "scatter"]
    # the same numbers as the single-pair readout
    ser = phase_series(data, 1545.0, 1570.0, fits=fits, local=True)
    np.testing.assert_allclose(readouts["1545/1570"]["local"], ser["phase_diff"].to_numpy()[rows], atol=1e-9)
    np.testing.assert_allclose(readouts["1545/1570"]["cubic"], ser["phase_diff_cubic"].to_numpy()[rows], atol=1e-9)
    assert list(readouts["1545/1570"].columns[:3]) == ["frame", "time", "condition"]


def test_one_wavelength_table_feeds_every_pair(data, fits):
    rows = np.arange(0, len(data), 40)
    phases = {lam: wavelength_phase(data, fits, lam, rows=rows) for lam in (1550.0, 1560.0, 1570.0)}
    assert all(p["local_ok"].all() for p in phases.values())
    r1, r2, r3 = (pair_readout(phases[a], phases[b])["local"].to_numpy()
                  for a, b in ((1550.0, 1560.0), (1560.0, 1570.0), (1550.0, 1570.0)))
    np.testing.assert_allclose(r1 + r2, r3, atol=1e-9)             # phase differences add up
    readouts, _ = compare_pairs(data, fits, [(1550.0, 1570.0)], data.temperature, rows=rows, phases=phases)
    np.testing.assert_allclose(readouts["1550/1570"]["local"], r3, atol=1e-12)     # reused, not refitted


def test_the_fit_options_default_to_the_published_fit(data, fits):
    f0, wl, y = fits[0], data.wavelengths, data.spectra[0]
    ref = local_fit(wl, y, f0, 1570.0)
    same = local_fit(wl, y, f0, 1570.0, order=3, fit_powers=True, envelope=False)
    assert ref.phi == same.phi and ref.rms == same.rms
    assert ref.rms < ref.rms_start                                 # the fit improves on where it starts
    # it stops by itself on a tolerance after a handful of evaluations, far below the ceiling
    assert 3 <= ref.nfev <= 30 and "1e-8" in ref.stop and "ceiling" not in ref.stop
    np.testing.assert_allclose(ref.model(wl[500:510], start=True),
                               ref.start[0] ** 2 + ref.start[1] ** 2 + 2 * ref.start[0] * ref.start[1]
                               * np.cos(np.polyval(ref.start[2:], wl[500:510] - 1570.0)))
    only_phase = local_fit(wl, y, f0, 1570.0, order=0)
    np.testing.assert_allclose(only_phase.coefficients[:-1], only_phase.start[2:-1])   # slope, chirp, cubic term kept
    assert only_phase.coefficients[-1] != only_phase.start[-1]
    assert only_phase.rms >= ref.rms - 1e-12                       # fewer free parameters cannot fit better
    fixed = local_fit(wl, y, f0, 1570.0, fit_powers=False)
    assert (fixed.t1, fixed.t2) == pytest.approx((abs(fixed.start[0]), abs(fixed.start[1])))
    sloped = local_fit(wl, y, f0, 1570.0, envelope=True)
    assert sloped.rms <= ref.rms + 1e-9 and (sloped.t1_slope != 0 or sloped.t2_slope != 0)
    assert ref.t1_slope == 0 and ref.t2_slope == 0


def test_profile_degree(data):
    wl, y = data.wavelengths, data.spectra[0]
    cubic = fit_phase(wl, y)
    assert len(cubic.coefficients) == 4
    np.testing.assert_array_equal(cubic.coefficients, fit_phase(wl, y, degree=3, skip_inner=0).coefficients)
    quartic, parabola = fit_phase(wl, y, degree=4), fit_phase(wl, y, degree=2)
    assert len(quartic.coefficients) == 5 and quartic.rms <= cubic.rms + 1e-12
    assert parabola.rms > 3 * cubic.rms                           # a parabola cannot follow the profile
    assert quartic.lambda_c == pytest.approx(cubic.lambda_c, abs=1.5)
    inner = fit_phase(wl, y, skip_inner=1)
    assert inner is not None and inner.lambda_c == pytest.approx(cubic.lambda_c, abs=1.5)


def test_parameter_study_on_the_cooldown(data):
    variants = [("defaults (paper)", "—", {}), ("profile: degree 4", "profile", dict(degree=4)),
                ("local: 1.5-cycle window", "local", dict(cycles=1.5)),
                ("local: t1, t2 fixed (Eq. 9)", "local", dict(fit_powers=False))]
    seen = []
    study = parameter_study(data, [(1545.0, 1570.0)], data.temperature, rows=np.arange(0, len(data), 12),
                            variants=variants, progress=lambda i, n, name: seen.append((i, n)))
    assert seen[0] == (0, 4) and seen[-1] == (4, 4)
    loc = study[study["method"] == "local fit"].set_index("variant")
    assert loc.loc["defaults (paper)", "slope_change_pct"] == 0.0 and loc.loc["defaults (paper)", "scatter_ratio"] == 1.0
    # the profile only starts the local fit: its degree hardly moves the refined readout …
    assert abs(loc.loc["profile: degree 4", "slope_change_pct"]) < 0.5
    assert loc.loc["profile: degree 4", "scatter_ratio"] == pytest.approx(1.0, abs=0.1)
    # … the local fit's own settings do, and without free mode powers it is mostly refused
    assert abs(loc.loc["local: 1.5-cycle window", "slope_change_pct"]) > abs(loc.loc["profile: degree 4", "slope_change_pct"])
    assert loc.loc["local: t1, t2 fixed (Eq. 9)", "refined_pct"] < 50
    # "cubic only" rows exist for the settings that change the profile only
    assert set(study[study["method"] == "cubic only"]["variant"]) == {"defaults (paper)", "profile: degree 4"}
    summary = study_summary(study)
    assert list(summary["variant"]) == [v[0] for v in variants]
    spread = study_spread(study)
    assert list(spread["pair"]) == ["1545/1570"] and 0 < spread["spread_pct"].iloc[0] < 10
    # counted: the defaults and degree 4; not the 1.5-cycle window (2.5× the scatter: a window that
    # is not a whole number of cycles ties the phase to the mode powers) nor the refused setting
    assert spread["n_settings"].iloc[0] == 2
    assert loc.loc["local: 1.5-cycle window", "scatter_ratio"] > 1.5


@pytest.mark.parametrize("order", [0, 1, 3])
def test_simulated_slope_is_recovered_whatever_is_left_free(order):
    """SIMULATED, known answer: a 0.7 rad tilt of φ(1570) − φ(1545) under a 3π common shift."""
    sim, _ = simulate_run(n_frames=60, common_rad=3 * np.pi, tilt_rad=-0.7, noise=0.0013)
    variants = [("defaults (paper)", "—", {}), (f"order {order}", "local", dict(order=order))]
    study = parameter_study(sim, [(1545.0, 1570.0)], sim.temperature, variants=variants)
    row = study[(study["method"] == "local fit") & (study["variant"] == f"order {order}")].iloc[0]
    assert abs(row["slope"]) == pytest.approx(0.7, rel=0.03)
