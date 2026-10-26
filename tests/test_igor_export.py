"""Igor Text export: legal names, the advisor's naming, round trip, NaN handling."""

import numpy as np
import pandas as pd

from sms_analysis.igor_export import (findpeak_itx, igor_text, parse_igor_text, spectra_itx,
                                      track_names, tracks_itx, wave_name)


def test_wave_names_are_igor_legal():
    assert wave_name("peak@1552.8") == "peak1552_8"
    assert wave_name("1552") == "w1552"
    assert wave_name("Mass (g)") == "Mass_g"
    assert len(wave_name("x" * 60)) == 31


def test_advisor_naming():
    n = track_names("dip", 1555.3)
    assert n == {"wavelength": "peakmin1555", "time": "timemin1555", "temperature": "tempmin1555",
                 "shift": "shift1555"}
    assert track_names("peak", 1489.0, "a")["wavelength"] == "peakmax1489a"


def test_round_trip_with_nan_and_mixed_lengths():
    text = igor_text({"a": [1.0, np.nan, 3.5], "b": [0.1, 0.2, 0.3], "c": [7.0]},
                     notes={"a": 'says "hi"'}, scales={"c": (1475.0, 0.1, "nm")})
    assert text.startswith("IGOR\nWAVES/D/O a, b\nBEGIN\n")
    assert "\tNaN\t0.2" in text
    assert "WAVES/D/O c\n" in text
    assert "X SetScale/P x 1475.0,0.1,\"nm\", c" in text
    assert "X Note a, \"says 'hi'\"" in text
    back = parse_igor_text(text)
    assert np.isnan(back["a"][1]) and back["a"][2] == 3.5 and back["b"].tolist() == [0.1, 0.2, 0.3]
    assert back["c"].tolist() == [7.0]


def test_spectra_tracks_and_findpeak_files():
    wl = np.linspace(1500, 1510, 11)
    sp = np.random.default_rng(0).random((4, 11))
    s = spectra_itx(wl, sp, time=[0, 1, 2, 3], temperature=[20, 21, 22, 23], rows=[0, 3], source="x.csv")
    back = parse_igor_text(s)
    assert set(back) == {"wave0", "n0", "n3", "tim", "temp"}
    assert np.allclose(back["n3"], sp[3]) and back["tim"].tolist() == [0.0, 3.0]

    tracks = {"dip@1552.8": pd.DataFrame({"wavelength": [1552.8, np.nan, 1553.6]}),
              "dip@1553.4": pd.DataFrame({"wavelength": [1553.4, 1553.5, 1553.6]}),   # same Igor name → suffix a
              "peak@1489.0": pd.DataFrame({"wavelength": [1489.0, 1489.1, 1489.2]})}
    t = tracks_itx(tracks, time=[0, 1, 2], condition=[21, 21, 22], mass=[0, 50, 50])
    back = parse_igor_text(t)
    assert set(back) == {"tim", "temp", "mass", "peakmin1553", "peakmin1553a", "peakmax1489"}
    assert np.isnan(back["peakmin1553"][1])

    f = findpeak_itx("dip", 1576.0, [0, 1], [1574.0, 1574.3], [21.7, 21.7], shift=[0, 0.3], mass=[0, 50],
                     call="FindPeak /R=(1572.0,1578.2) /B=5 /N", suffix="a")
    back = parse_igor_text(f)
    assert set(back) == {"peakmin1576a", "timemin1576a", "tempmin1576a", "shift1576a", "mass"}
    assert "FindPeak /R=(1572.0,1578.2) /B=5 /N" in f
