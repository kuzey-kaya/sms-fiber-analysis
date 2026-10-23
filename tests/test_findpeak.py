"""Igor's FindPeak (sms_analysis.findpeak): the operation on known waves and the group's tracking graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sms_analysis import fit_sensitivity, load_sms_csv
from sms_analysis.findpeak import box_smooth, find_peak, points_to_nm, track_findpeak


def test_box_smooth_is_a_sliding_average():
    y = np.arange(10.0)
    assert np.allclose(box_smooth(y, 1), y)
    assert np.allclose(box_smooth(y, 3)[1:-1], y[1:-1])          # a line is unchanged inside
    assert box_smooth(y, 4).shape == y.shape                     # even boxes become odd


def test_find_peak_on_a_cosine():
    x = np.arange(1500, 1520.01, 0.1)
    y = 0.6 + 0.3 * np.cos(2 * np.pi * (x - 1507.3) / 8.0)
    r = find_peak(x, y, box=5)
    assert r.flag == 0 and r.peak_loc == pytest.approx(1507.3, abs=0.01)
    assert r.peak_val == pytest.approx(0.9, abs=0.01)
    assert r.peak_width == pytest.approx(4.0, abs=0.1)           # inflection to inflection = half a period
    rn = find_peak(x, y, box=5, negative=True)
    assert rn.peak_loc == pytest.approx(1503.3, abs=0.01)         # the first minimum from the start
    rb = find_peak(x, y, 1520, 1500, box=5, negative=True)
    assert rb.peak_loc == pytest.approx(1519.3, abs=0.01)         # searching backwards: the first from the end
    assert find_peak(x, y, 1507.5, 1510.5, box=5).flag != 0       # no maximum inside that range
    assert find_peak(x, y, box=5, min_level=0.95).flag != 0       # /M above the peak


def test_igor_ranges_on_the_strain_run():
    """The advisor's calls on the 2026-10-15 file: min1576a = points 735-766, min1489 = points 274-353."""
    p = Path(__file__).resolve().parents[1] / "data" / "strain.csv"
    if not p.exists():
        pytest.skip("data/strain.csv not present")
    d = load_sms_csv(p)
    a, b = points_to_nm(d, 735, 766)
    assert (round(a, 1), round(b, 1)) == (1572.0, 1578.2)
    tr = track_findpeak(d, a, b, box=5, negative=True)
    w = tr["wavelength"].to_numpy()
    assert tr["found"].all()
    assert np.nanmin(w) == pytest.approx(1574.0, abs=0.1) and np.nanmax(w) == pytest.approx(1576.6, abs=0.1)
    assert np.nanstd(w[:60]) < 0.05                               # the first unloaded plateau is quiet


def test_igor_range_on_the_cooldown_matches_the_published_fringe(data):
    """peak near 1493 nm in a fixed 1490-1497 band: -141 pm/°C, as the tracker and the group's Igor."""
    tr = track_findpeak(data, 1490.0, 1497.0, box=5, negative=False)
    assert tr["found"].all()
    assert fit_sensitivity(tr).slope * 1e3 == pytest.approx(-141.2, abs=1.0)
