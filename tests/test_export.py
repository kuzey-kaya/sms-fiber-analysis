"""The CSV tables of the Export tab: the condition column is named after the condition and every
file carries the same values (2026-11-16: on the strain run the long file said ``temperature``
over a column of grams, and the wide file disagreed with it on the settle frames)."""

from pathlib import Path

import numpy as np
import pandas as pd

from sms_analysis import load_sms_csv, track_all_extrema
from sms_analysis.auto import choose
from sms_analysis.export import condition_column, frame_csv, long_table, wide_table
from sms_analysis.igor_export import igor_text, parse_igor_text

ROOT = Path(__file__).resolve().parents[1]


def test_condition_column_names():
    assert condition_column("Temperature") == "temperature"
    assert condition_column("Mass") == "mass"
    assert condition_column("Applied strain (µε)") == "applied_strain"
    assert condition_column("Mass", has_condition=False) == "condition"
    for bad in ("", None, "°", "Time", "wavelength"):          # empty, or a header the file already has
        assert condition_column(bad) == "condition"


def test_cooldown_tables_keep_the_temperature():
    data = load_sms_csv(ROOT / "data" / "data.csv")
    tracks = dict(list(track_all_extrema(data).items())[:3])
    name = condition_column("Temperature")
    wide = wide_table(tracks, data.time, data.temperature, name)
    long = long_table(tracks, name)
    assert list(wide.columns) == ["time", "temperature", *tracks]
    assert list(long.columns) == ["time", "temperature", "wavelength", "value", "fringe"]
    assert len(wide) == len(data) and len(long) == 3 * len(data)
    np.testing.assert_allclose(wide["temperature"], data.temperature)
    first = next(iter(tracks))
    np.testing.assert_allclose(long.loc[long["fringe"] == first, "wavelength"], wide[first], equal_nan=True)


def test_load_run_tables_say_mass_and_agree_with_each_other():
    data = load_sms_csv(ROOT / "data" / "strain.csv")
    auto = choose(data)
    assert auto.step_cond is not None
    mass = np.asarray(auto.step_cond, float)
    assert np.isnan(mass).any()                                 # the settle rows are blank
    tracks = {k: t.assign(temperature=mass) for k, t in list(auto.tracks.items())[:3]}
    wide = wide_table(tracks, data.time, mass, "mass", temperature=data.temperature)
    long = long_table(tracks, "mass")
    assert list(wide.columns) == ["time", "temperature", "mass", *tracks]
    assert "temperature" not in long.columns and "mass" in long.columns
    # the thermometer column is the file's (about 22 °C), the mass column the steps (0 … 400 g)
    assert 21 < np.nanmin(wide["temperature"]) and np.nanmax(wide["temperature"]) < 23
    assert np.nanmin(wide["mass"]) == 0 and np.nanmax(wide["mass"]) >= 300
    # one fringe of the long file carries exactly the wide file's mass column, blanks included
    first = next(iter(tracks))
    np.testing.assert_array_equal(long.loc[long["fringe"] == first, "mass"].to_numpy(), wide["mass"].to_numpy())
    # as text: the blank rows are empty cells, not the word "nan"
    text = wide.to_csv(index=False)
    assert "nan" not in text.lower()
    assert text.splitlines()[0].startswith("time,temperature,mass,")


def test_frame_csv_renames_and_replaces_the_condition():
    frame = pd.DataFrame({"time": [0.0, 1.0, 2.0], "temperature": [50.0, 50.0, 100.0], "wavelength": [1.0, 2.0, 3.0]})
    text = frame_csv(frame, "mass", condition=[50.0, np.nan, 100.0])
    lines = text.splitlines()
    assert lines[0] == "time,mass,wavelength" and lines[2] == "1.0,,2.0"
    assert list(frame.columns) == ["time", "temperature", "wavelength"]          # the caller's frame is untouched
    assert frame_csv(frame[["time", "wavelength"]], "mass").splitlines()[0] == "time,wavelength"


def test_igor_text_ends_with_a_blank_line():
    """WaveMetrics, Igor Text File Format: the file must end with a blank line."""
    text = igor_text({"a": [1.0, 2.0], "b": [3.0, np.nan]}, notes={"a": "note"})
    assert text.startswith("IGOR\n") and text.endswith('"\n\n')
    back = parse_igor_text(text)
    np.testing.assert_allclose(back["a"], [1.0, 2.0])
    assert np.isnan(back["b"][1])
