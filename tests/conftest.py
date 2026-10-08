"""Shared fixtures: the shipped run is loaded and tracked once per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sms_analysis import load_sms_csv, track_all_extrema  # noqa: E402
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
