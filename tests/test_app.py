"""FringeLab starts and every tab renders without an exception (headless)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_tabs(app):
    names = [t.label.split(": ")[-1] for t in app.tabs]
    assert names == ["Spectral map", "Spectrum & peaks", "Fringe tracking", "Sensitivity", "Phase",
                     "3-D view", "Movie", "Export"]


@pytest.mark.parametrize("model", ["Linear", "Quadratic", "Both"])
def test_calibration_models(app, model):
    app.radio(key="cal_model").set_value(model).run()
    assert not app.exception


@pytest.mark.parametrize("kind", ["Phase readout (SM1E.2)", "Spectra and fringes"])
def test_both_movies(app, kind):
    app.radio(key="movie_kind").set_value(kind).run()
    assert not app.exception
    app.toggle(key="movie_build").set_value(True).run()            # built on request only
    assert not app.exception


def test_movie_and_phase_are_computed_on_request_only():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.toggle(key="movie_build").value
    assert not at.toggle(key="phase_compute").value
    assert not any("λc (first" in m.value for m in at.markdown)
    at.toggle(key="phase_compute").set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("λc (first" in m.value for m in at.markdown)          # the phase tiles appeared
    assert at.toggle(key="phase_local").value                         # local fit on by default
    cmp = [d.value for d in at.dataframe if "scatter (mrad)" in d.value.columns]
    assert cmp and len(cmp[0]) == 2
    assert cmp[0]["scatter (mrad)"].iloc[1] < 0.4 * cmp[0]["scatter (mrad)"].iloc[0]   # the comparison shows it
    assert any("smaller** with the local fit" in m.value for m in at.markdown)
    at.toggle(key="phase_local").set_value(False).run()               # cubic only still works
    assert not at.exception, [str(e.value) for e in at.exception]


def test_upload_page_shows_the_format_guide():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Upload a CSV file").run()
    assert not at.exception
    assert len(at.get("download_button")) == 1          # the template CSV


def test_several_fringes_in_detail():
    # Fresh session: the module fixture has set the calibration radio, which is not drawn
    # while several fringes are selected, and AppTest cannot carry that state across.
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="detail_mode").set_value("Several fringes").run()
    ms = at.multiselect(key="detail_fringes")
    ms.set_value(ms.options[:3]).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    ms = at.multiselect(key="detail_fringes")               # re-fetch: elements belong to one run
    assert len(ms.value) == 3
    ms.set_value(ms.options[:1]).run()
    assert not at.exception


def test_stepped_run_auto_tracker_and_step_condition(steps_csv):
    """A load-step file: the app starts on the fringe-width fit, finds the plateaus and, once the
    step values are used as the condition, reports the sensitivity per gram."""
    os.environ["FRINGELAB_EXAMPLE"] = str(steps_csv)     # read on every (re)run of the script
    try:
        _check_stepped_run(AppTest.from_file(APP, default_timeout=300).run())
    finally:
        os.environ.pop("FRINGELAB_EXAMPLE", None)


def _check_stepped_run(at):
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["track_mode"] == "Fringe-width fit" and at.session_state["reject_jumps"] is False
    def kind_box():     # re-fetched after every run: elements belong to one run's tree
        found = [c for c in at.sidebar.checkbox if c.label.startswith("Load-step run")]
        assert len(found) == 1
        return found[0]

    assert kind_box().value                                                   # guessed from the file
    assert any("entered per load step" in c.value for c in at.caption)
    assert any("9-frame median" in c.value for c in at.caption)                # adjustable, not flattened
    assert at.session_state["smooth_frames"] == 9 and at.session_state["smooth_flatten"] is False
    kind_box().set_value(False).run()                                         # a continuous run again
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["track_mode"] == "Standard" and at.session_state["reject_jumps"] is True
    assert not any("entered per load step" in c.value for c in at.caption)
    kind_box().set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    sens = [d.value for d in at.dataframe if "sensitivity (pm/g)" in d.value.columns and "R²" in d.value.columns]
    assert sens and (sens[0]["R²"] > 0.99).all()
    assert sens[0]["sensitivity (pm/g)"].between(55, 65).all()        # 3 nm per 50 g


def test_strain_file():
    """The advisor's strain run (shipped for the tests): recognised as a load-step run, the
    fringe-width fit is chosen, 20/20 fringes locked, 17 plateaus."""
    strain = Path(APP).parent / "data" / "strain.csv"
    if not strain.exists():
        pytest.skip("data/strain.csv not present")
    os.environ["FRINGELAB_EXAMPLE"] = str(strain)      # read on every rerun of the script
    try:
        _check_strain_file(AppTest.from_file(APP, default_timeout=300).run())
    finally:
        os.environ.pop("FRINGELAB_EXAMPLE", None)


def _check_strain_file(at):
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.sidebar.radio[0].value == "Example run (strain.csv)"
    assert at.session_state["track_mode"] == "Fringe-width fit"
    assert any("17 plateaus found on dip@1552.8" in e.label for e in at.expander)
    assert any("20 / 20" in m.value for m in at.markdown)
    assert any("entered per load step" in c.value for c in at.caption)      # load steps chosen by default
    assert any("Flickering fringe" in c.value and "peak@1529.8" in c.value for c in at.caption)
    assert at.checkbox(key="hide_flicker").value
    assert "peak@1529.8" not in at.selectbox(key="detail_fringe").value       # never the default
    shift_sel = [m for m in at.multiselect if m.label.startswith("Fringes in the plot")][0]
    shift_sel.set_value(shift_sel.options[:1]).run()                          # one fringe alone
    assert not at.exception, [str(e.value) for e in at.exception]
    at.toggle(key="phase_compute").set_value(True).run()                      # readout drawn smoothed
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Readout drawn as 9-frame median" in c.value for c in at.caption)
    pa, pb = at.session_state["phase_a"], at.session_state["phase_b"]
    assert (pa, pb) != (1545.0, 1570.0) and 1540 <= pa < pb <= 1600               # moved right of λc ≈ 1522
    assert any("Pair moved to" in c.value for c in at.caption)
    cmp = [d.value for d in at.dataframe if "scatter (mrad)" in d.value.columns]
    assert cmp and cmp[0]["scatter (mrad)"].iloc[1] < 0.5 * cmp[0]["scatter (mrad)"].iloc[0]
    axis = [r for r in at.radio if r.key and r.key.startswith("igor_axis_")]
    assert axis and axis[0].set_value(axis[0].options[1]).run() is not None    # temperature on the right axis
    assert not at.exception, [str(e.value) for e in at.exception]
    at.checkbox(key="smooth_flatten").set_value(True).run()                   # the owner may flatten
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Readout drawn as plateau averages" in c.value for c in at.caption)
    at.slider(key="smooth_frames").set_value(1).run()
    at.checkbox(key="smooth_flatten").set_value(False).run()                  # or not smooth at all
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Lines drawn: as tracked" in c.value for c in at.caption)
    at.radio(key="detail_mode").set_value("One fringe (Igor style)").run()
    assert not at.exception
    at.radio(key="movie_kind").set_value("Spectra and fringes").run()
    at.toggle(key="movie_build").set_value(True).run()                        # smoothed + width-fit movie
    assert not at.exception, [str(e.value) for e in at.exception]
    smooth_box = [c for c in at.checkbox if c.label == "Draw the smoothed spectrum"]
    fit_box = [c for c in at.checkbox if c.label == "Width-fit the peaks and dips"]
    assert smooth_box[0].value and fit_box[0].value                           # noisy, width-fit run
