"""Phase-unwrapping readout (Salik et al. SM1E.2): real run and SIMULATED runs with a known answer."""

from __future__ import annotations

import numpy as np
import pytest

from sms_analysis.phase import fit_phase, phase_sensitivity, phase_series
from sms_analysis.synthetic import simulate_run


def test_first_spectrum_labelling_matches_paper(data):
    """Same numbering as the paper's Fig. 1: λ−1 peak, λc dip, λ+1 peak, λ+2 the 1542 nm dip."""
    f = fit_phase(data.wavelengths, data.spectra[0])
    labs = dict(zip(f.labels(), zip(np.round(f.wavelengths, 1), f.kinds)))
    assert labs["λ−1"] == (1515.7, "peak")
    assert labs["λc"] == (1526.2, "dip")
    assert labs["λ+1"] == (1534.6, "peak")
    assert labs["λ+2"] == (1542.3, "dip")
    assert f.rms < 0.06
    assert f.lambda_c == pytest.approx(1525.2, abs=0.1)


def test_real_run_readout(data, fits):
    s = phase_series(data, fits=fits)
    assert s["phase_diff"].notna().all()                       # every spectrum fitted
    assert np.nanmax(np.abs(np.diff(s["phase_diff"]))) < 0.15  # no jumps
    sens = phase_sensitivity(s)
    assert sens.slope * 1e3 == pytest.approx(-31.6, abs=0.5)
    assert sens.span_rad == pytest.approx(0.80, abs=0.05)


@pytest.mark.parametrize("kw", [dict(lambda_c_guess=1525.0), dict(lambda_c_guess=1525.0, candidates=1),
                                dict(lambda_c_guess=1525.0, candidates=99)])
def test_result_does_not_depend_on_how_lambda_c_is_chosen(data, fits, kw):
    ref = phase_series(data, fits=fits)
    alt = phase_series(data, fits=[fit_phase(data.wavelengths, data.spectra[i], **kw) for i in range(len(data))])
    assert np.nanmax(np.abs(alt["phase_diff"] - ref["phase_diff"])) < 1e-9


@pytest.mark.parametrize("common, tilt", [(3 * np.pi, 0.0), (10 * np.pi, 7.0), (0.0, 7.0), (-6 * np.pi, -3.0)])
def test_simulated_runs_recover_the_known_phase_difference(common, tilt):
    """SIMULATED: a common phase shift is ignored, the tilt is recovered, over several fringe spacings."""
    sim, truth = simulate_run(n_frames=150, common_rad=common, tilt_rad=tilt)
    s = phase_series(sim)
    got = s["phase_diff_signed"].to_numpy()
    assert np.isfinite(got).all()
    err = got - truth
    err -= np.median(err)
    assert np.abs(err).max() < 0.2
    assert (got[-1] - got[0]) == pytest.approx(tilt, abs=0.1)



def test_local_fit_refines_the_cooldown_readout(data, fits):
    """PTL local two-mode fit: on the cooldown the 1545/1570 readout scatters about its line
    by ~6.6 mrad instead of ~25 mrad, every frame refined, with t1 ≈ 0.71 and t2 ≈ 0.18."""
    from sms_analysis.phase import local_fit, phase_series
    ser = phase_series(data, 1545.0, 1570.0, fits=fits, local=True)
    assert ser["local_ok"].all()
    ok = ser.dropna(subset=["phase_diff", "temperature"])

    def scatter(col):
        k = np.polyfit(ok["temperature"], ok[col], 1)
        return float(np.std(ok[col] - np.polyval(k, ok["temperature"])))

    assert scatter("phase_diff") < 0.012
    assert scatter("phase_diff") < 0.4 * scatter("phase_diff_cubic")
    assert 0.65 < ser["t1_a"].mean() < 0.76 and 0.15 < ser["t2_a"].mean() < 0.22
    f0 = next(f for f in fits if f is not None)
    lf = local_fit(data.wavelengths, data.spectra[0], f0, 1570.0)
    assert lf is not None and lf.converged and lf.rms < 0.01
    assert abs(lf.phi - lf.phi_cubic) < 0.3                      # same branch as the cubic
    assert local_fit(data.wavelengths, data.spectra[0], f0, f0.lambda_c + 2.0) is None   # window holds λc
