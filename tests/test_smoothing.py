"""The running median behind "Smooth the drawn lines": nothing is padded in at the ends.

2026-11-18, the advisor with a simulated temperature ramp of 31 spectra: "it gives the peak
locations exactly the same for the first three time values". scipy's medfilt pads with zeros,
which sort below any wavelength, so the first three points of a falling fringe (and the last
three of a rising one) all came out as the third value."""

from pathlib import Path

import numpy as np
from scipy.signal import medfilt

from sms_analysis.smoothing import running_median
from sms_analysis.synthetic import simulate_run
from sms_analysis import track_all_extrema

ROOT = Path(__file__).resolve().parents[1]


def test_a_ramp_comes_back_unchanged_in_both_directions():
    falling = 1476.65 - 0.08 * np.arange(31.0)
    rising = 1535.44 + 0.26 * np.arange(31.0)
    for series in (falling, rising):
        for frames in (3, 5, 9, 61):
            np.testing.assert_array_equal(running_median(series, frames), series)
    # what the app drew before: the first three values of the falling fringe identical
    old = medfilt(falling, 5)
    assert old[0] == old[1] == old[2] == falling[2]
    assert len(set(running_median(falling, 5)[:3])) == 3


def test_it_is_the_ordinary_median_away_from_the_ends():
    x = 1500 + np.random.default_rng(1).normal(0, 1, 300)
    for frames in (3, 5, 9, 61):
        half = frames // 2
        np.testing.assert_array_equal(running_median(x, frames)[half:-half], medfilt(x, frames)[half:-half])
    spike = np.array([1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(running_median(spike, 3), [1, 2, 4, 5, 5, 6, 7])     # the spike goes
    assert running_median(spike, 5)[0] == 1.0 and running_median(spike, 5)[-1] == 7.0  # the ends stay


def test_short_series_and_even_windows():
    np.testing.assert_array_equal(running_median([5.0, 1.0], 5), [5, 1])
    np.testing.assert_array_equal(running_median([1.0, 9.0, 2.0], 9), [1, 2, 2])
    np.testing.assert_array_equal(running_median([3.0, 1.0, 2.0], 1), [3, 1, 2])
    np.testing.assert_array_equal(running_median([1.0, 2.0, 3.0, 4.0], 4), running_median([1.0, 2.0, 3.0, 4.0], 5))


def test_tracks_of_a_simulated_ramp_keep_their_first_frames_apart():
    """The advisor's case end to end: a clean ramp, tracked, then smoothed for drawing."""
    data, _ = simulate_run(n_frames=31, common_rad=3.0, tilt_rad=1.0, noise=0.0, seed=0)
    tracks = track_all_extrema(data)
    checked = 0
    for track in tracks.values():
        w = track["wavelength"].to_numpy(float)
        if np.isfinite(w).all() and abs(w[-1] - w[0]) > 0.3 and np.all(np.diff(np.sign(np.diff(w))) == 0):
            drawn = running_median(w, 5)
            np.testing.assert_array_equal(drawn, w)
            assert len(set(drawn[:3])) == 3 and len(set(drawn[-3:])) == 3
            checked += 1
    assert checked >= 5


def test_the_app_and_the_step_detection_use_it():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "medfilt(" not in app and "running_median(w[ok], frames)" in app
    assert "medfilt" not in (ROOT / "sms_analysis" / "steps.py").read_text(encoding="utf-8")
