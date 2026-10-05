#!/usr/bin/env python3
"""Show how a CSV will be interpreted before running the analysis.

Usage: python scripts/inspect_csv.py [path/to/file.csv]     (default: data/data.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sms_analysis.io import load_sms_csv  # noqa: E402


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/data.csv")
    if not path.exists():
        sys.exit(f"File not found: {path}")
    print(f"File: {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    raw = pd.read_csv(path, sep=None, engine="python", nrows=5)
    print(f"Raw read: {raw.shape[1]} columns; first 6 headers: {list(raw.columns[:6])}")
    print(f"First row (first 6 values): {list(raw.iloc[0, :6])}")

    try:
        data = load_sms_csv(path)
    except Exception as e:  # noqa: BLE001
        print(f"\nLOAD FAILED: {e}")
        sys.exit(2)

    print(f"\nInterpreted as: {data}")
    print(f"  wavelengths: {len(data.wavelengths)} points, "
          f"step {np.median(np.diff(data.wavelengths)):.3f} nm")
    print(f"  time: first {data.time[0]:.1f}, last {data.time[-1]:.1f}, "
          f"median step {np.median(np.diff(data.time)):.2f}")
    if data.has_temperature:
        print(f"  temperature: {np.nanmin(data.temperature):.2f} .. {np.nanmax(data.temperature):.2f}")
    else:
        print("  temperature: NONE found -> maps and tracking work; calibration fits are skipped")
    print(f"  transmittance: {np.nanmin(data.spectra):.3f} .. {np.nanmax(data.spectra):.3f}, "
          f"NaN cells: {int(np.isnan(data.spectra).sum())}")
    print("\nIf a column was mis-detected, name it explicitly in Python:\n"
          "  load_sms_csv(path, temp_col='Temperature', time_col='Time')")


if __name__ == "__main__":
    main()
