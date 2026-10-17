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
    def mode_radio():   # re-fetched after every run: elements belong to one run's tree
        found = [r for r in at.sidebar.radio if r.value in ("Column in the file", "Load steps (by hand)")]
        assert len(found) == 1
        return found[0]

    assert mode_radio().value == "Load steps (by hand)"                      # chosen automatically
    assert any("entered per load step" in c.value for c in at.caption)
    mode_radio().set_value("Column in the file").run()                       # the file's column still works
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not any("entered per load step" in c.value for c in at.caption)
    mode_radio().set_value("Load steps (by hand)").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    sens = [d.value for d in at.dataframe if "sensitivity (pm/g)" in d.value.columns and "R²" in d.value.columns]
    assert sens and (sens[0]["R²"] > 0.99).all()
    assert sens[0]["sensitivity (pm/g)"].between(55, 65).all()        # 3 nm per 50 g


def test_shipped_strain_example():
    """The second example run: the fringe-width fit is chosen, 20/20 fringes locked, 17 plateaus."""
    if not (Path(APP).parent / "data" / "strain.csv").exists():
        pytest.skip("data/strain.csv not present")
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Load steps run (data/strain.csv)").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["track_mode"] == "Fringe-width fit"
    assert any("17 plateaus found on dip@1552.8" in e.label for e in at.expander)
    assert any("20 / 20" in m.value for m in at.markdown)
    assert any("entered per load step" in c.value for c in at.caption)      # load steps chosen by default
    at.radio(key="detail_mode").set_value("One fringe (Igor style)").run()
    assert not at.exception
