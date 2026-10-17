"""Shared fixtures: the shipped run is loaded and tracked once per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sms_analysis import SpectraSet, load_sms_csv, track_all_extrema  # noqa: E402
from sms_analysis.phase import phase_fits  # noqa: E402

DATA_CSV = ROOT / "data" / "data.csv"


@pytest.fixture(scope="session")
def data():
    if not DATA_CSV.exists():
        pytest.skip("data/data.csv not present")
    return load_sms_csv(DATA_CSV)


@pytest.fixture(scope="session")
def tracks(data):
    return track_all_extrema(data)


@pytest.fixture(scope="session")
def fits(data):
    return phase_fits(data)


@pytest.fixture(scope="session")
def steps():
    """SIMULATED stepped run: 30 nm period fringes every 0.2 nm, noise 0.008, the whole pattern
    stepping 0 → 3 → 0 → 6 → 0 nm in single frames (a load put on and taken off)."""
    rng = np.random.default_rng(0)
    wl = np.round(np.arange(1450.0, 1650.01, 0.2), 1)
    shift = np.repeat([0.0, 3.0, 0.0, 6.0, 0.0], 40)
    spectra = np.array([0.6 + 0.3 * np.cos(2 * np.pi * (wl - 1450 - s) / 30.0) for s in shift])
    spectra += rng.normal(0, 0.008, spectra.shape)
    n = len(shift)
    data = SpectraSet(wl, np.arange(n) * 3.6, np.full(n, np.nan), spectra, layout="simulated")
    return data, shift


@pytest.fixture(scope="session")
def steps_csv(tmp_path_factory, steps):
    """The simulated stepped run as a CSV in the standard layout (time column, no condition)."""
    import pandas as pd
    data, _ = steps
    df = pd.DataFrame(data.spectra, columns=[f"{w:g}" for w in data.wavelengths])
    df.insert(0, "time", data.time)
    p = tmp_path_factory.mktemp("steps") / "steps.csv"
    df.to_csv(p, index=False)
    return p
