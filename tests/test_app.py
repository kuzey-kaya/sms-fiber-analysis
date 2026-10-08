"""FringeLab starts and every tab renders without an exception (headless)."""

from __future__ import annotations

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
