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
