"""The loader reads the shipped run and the layout variants the README promises."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sms_analysis import load_sms_csv


def test_shipped_run_shape(data):
    assert len(data) == 730
    assert len(data.wavelengths) == 1001
    assert data.wavelengths[0] == pytest.approx(1475.0)
    assert data.wavelengths[-1] == pytest.approx(1575.0)
    assert np.nanmax(data.temperature) == pytest.approx(44.06, abs=0.01)
    assert np.nanmin(data.temperature) == pytest.approx(22.77, abs=0.01)
    assert data.layout == "standard"


@pytest.fixture
def small():
    """Five spectra × 41 wavelengths (1500–1510 nm; the loader wants ≥ 5 nm of axis)."""
    wl = np.round(np.arange(1500.0, 1510.05, 0.25), 2)
    rows = np.array([0.5 + 0.3 * np.cos(2 * np.pi * (wl - 1500 - 0.1 * k) / 2.5) for k in range(5)])
    temp = np.array([30.0, 29.5, 29.0, 28.5, 28.0])
    time = np.arange(5) * 20.0
    return wl, rows, temp, time


def _check(d, small, has_temp=True):
    wl, rows, temp, time = small
    assert len(d) == 5
    assert np.allclose(d.wavelengths, wl)
    assert np.allclose(d.spectra, rows, atol=1e-6)
    if has_temp:
        assert np.allclose(d.temperature, temp)
    else:
        assert not d.has_temperature


def test_named_columns_anywhere(tmp_path, small):
    wl, rows, temp, time = small
    df = pd.DataFrame(rows, columns=[f"{w:g}" for w in wl])
    df.insert(3, "Time (s)", time)
    df.insert(0, "Temperature (°C)", temp)
    p = tmp_path / "named.csv"
    df.to_csv(p, index=True)                        # also adds an index column
    _check(load_sms_csv(p), small)


def test_semicolon_and_decimal_comma(tmp_path, small):
    wl, rows, temp, time = small
    df = pd.DataFrame(rows, columns=[f"{w:g}".replace(".", ",") for w in wl])
    df.insert(0, "time", time)
    df.insert(0, "temp", temp)
    p = tmp_path / "eu.csv"
    df.to_csv(p, sep=";", decimal=",", index=False)
    _check(load_sms_csv(p), small)


def test_without_temperature(tmp_path, small):
    wl, rows, temp, time = small
    df = pd.DataFrame(rows, columns=[f"{w:g}" for w in wl])
    df.insert(0, "Time (s)", time)
    p = tmp_path / "notemp.csv"
    df.to_csv(p, index=False)
    _check(load_sms_csv(p), small, has_temp=False)


def test_transposed(tmp_path, small):
    wl, rows, temp, time = small
    df = pd.DataFrame(rows.T, columns=[f"{t:g}" for t in time])
    df.insert(0, "wavelength", wl)
    p = tmp_path / "transposed.csv"
    df.to_csv(p, index=False)
    d = load_sms_csv(p)
    assert d.layout == "transposed"
    _check(d, small, has_temp=False)
