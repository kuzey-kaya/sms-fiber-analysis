"""The headline numbers quoted in README/CLAUDE.md and the report stay put."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sms_analysis import curvature_test, fit_sensitivity, sensitivity_map, track_feature
from sms_analysis.staircase import staircase_bin

ROOT = Path(__file__).resolve().parents[1]


def test_22_of_24_fringes_good(tracks):
    smap = sensitivity_map(tracks)
    assert len(smap) == 24
    assert (smap["r_squared"] >= 0.90).sum() == 22


def test_validation_fringe_matches_igor(tracks):
    """Fringe near 1493 nm: −141 pm/°C vs the group's published −142 pm/°C."""
    fit = fit_sensitivity(tracks["peak@1493.4"])
    assert fit.slope * 1e3 == pytest.approx(-141.3, abs=0.5)


def test_most_sensitive_fringe(data):
    track = track_feature(data, start_wavelength=1542.3, kind="dip")
    assert track["wavelength"].notna().sum() == 730
    fit = fit_sensitivity(track)
    assert fit.slope * 1e3 == pytest.approx(359.3, abs=0.2)
    assert fit.r_squared == pytest.approx(0.9972, abs=0.0005)
    curv = curvature_test(track)
    lo, hi = curv.sensitivity_at_ends_pm
    assert curv.quadratic.rmse * 1e3 == pytest.approx(49, abs=2)
    assert lo == pytest.approx(414, abs=2) and hi == pytest.approx(283, abs=2)


def test_staircase_table_reproduces_stored_csv(data):
    """Table 2 of the report: regenerated plateaus equal figures/staircase_plateaus.csv."""
    stored = ROOT / "figures" / "staircase_plateaus.csv"
    if not stored.exists():
        pytest.skip("figures/staircase_plateaus.csv not present")
    track = track_feature(data, start_wavelength=1542.3, kind="dip")
    new = staircase_bin(track, step_c=3.0).round(3)
    old = pd.read_csv(stored)
    assert np.allclose(new[old.columns].to_numpy(float), old.to_numpy(float), atol=1e-3)
