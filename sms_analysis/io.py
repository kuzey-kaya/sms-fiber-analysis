"""Loading SMS sensor spectra from CSV files.

Standard layout (as produced by the acquisition setup):
    column 0 : temperature (deg C)
    column 1 : time (seconds)
    columns 2..N : transmittance, one column per wavelength; the column
                   *header* is the wavelength in nm (e.g. "1475", "1475.1").
Each row is one full transmission spectrum recorded at that time/temperature.

The loader also accepts the common variations of that layout:
  * extra non-spectral columns (an index column, a comment column, ...):
    only columns whose header parses as a wavelength are treated as spectra;
  * temperature / time columns in any position, found by their header
    ("temp", "time", "sec", ...) or, failing that, by their position;
  * a file with time but no temperature (temperature is then NaN and the
    calibration steps are skipped), or with neither (time = frame index);
  * a transposed file (rows = wavelengths, columns = spectra);
  * comma, semicolon or tab separators, and decimal commas.
Run ``python scripts/inspect_csv.py <file>`` to see how a file is interpreted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WL_MIN, WL_MAX = 200.0, 5000.0   # anything outside this is not a wavelength in nm


@dataclass
class SpectraSet:
    """A time series of transmission spectra with their conditions."""

    wavelengths: np.ndarray  # (n_wavelengths,) nm
    time: np.ndarray         # (n_rows,) seconds
    temperature: np.ndarray  # (n_rows,) deg C (NaN if the file has none)
    spectra: np.ndarray      # (n_rows, n_wavelengths) transmittance
    source: str = ""
    layout: str = "standard"

    def __len__(self) -> int:
        return self.spectra.shape[0]

    @property
    def has_temperature(self) -> bool:
        return bool(np.isfinite(self.temperature).any())

    def __repr__(self) -> str:
        temp = (f"T {np.nanmin(self.temperature):.1f}-{np.nanmax(self.temperature):.1f} degC"
                if self.has_temperature else "no temperature column")
        return (
            f"SpectraSet({len(self)} spectra, "
            f"{self.wavelengths.min():.1f}-{self.wavelengths.max():.1f} nm, "
            f"{temp}, t {self.time.min():.0f}-{self.time.max():.0f} s, layout={self.layout})"
        )


def _to_float(x) -> float:
    """Parse a header like '1475.1', ' 1475,1 ', '1475.1 nm' -> float, else NaN."""
    if x is None:
        return np.nan
    s = str(x).strip().lower().replace("nm", "").strip()
    if re.fullmatch(r"[-+]?\d+,\d+", s):
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _layout_score(df: pd.DataFrame) -> int:
    """How many wavelengths a parse exposes: wavelength headers, or a wavelength first column."""
    hdr = np.array([_to_float(h) for h in df.columns])
    n_hdr = int((np.isfinite(hdr) & (hdr >= WL_MIN) & (hdr <= WL_MAX)).sum())
    first = pd.to_numeric(df.iloc[:, 0].astype(str).str.replace(",", ".", regex=False),
                          errors="coerce").to_numpy(float) if df.shape[1] else np.array([])
    n_col = int(np.isfinite(first).sum()) if _looks_like_wavelength_axis(first) else 0
    return max(n_hdr, n_col)


def _read_table(path: Path) -> pd.DataFrame:
    """Read with separator sniffing, then keep whichever parse exposes the most wavelengths.

    Sniffing alone misreads a ';'-separated file whose headers carry decimal
    commas ("1500,1"): the commas look like separators.  So the sniffed parse
    competes with explicit ';' / tab / ',' parses (with '.' or ',' decimals)
    and the one that yields the most wavelength columns wins; ties keep the
    sniffed parse.
    """
    try:
        best = pd.read_csv(path, sep=None, engine="python")
    except Exception:  # noqa: BLE001 - sniffing can fail on odd files
        best = pd.read_csv(path)
    best_score = _layout_score(best)
    for sep in (";", "\t", ","):
        for dec in (",", "."):
            if sep == dec:
                continue
            try:
                cand = pd.read_csv(path, sep=sep, decimal=dec)
            except Exception:  # noqa: BLE001
                continue
            score = _layout_score(cand)
            if score > best_score:
                best, best_score = cand, score
    return best


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every cell to float (handles decimal commas and stray text)."""
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False),
                                   errors="coerce")
    return out


def _looks_like_wavelength_axis(values: np.ndarray) -> bool:
    """A wavelength axis: >= 20 finite values, monotonic, small uniform steps."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 20 or v.min() < WL_MIN or v.max() > WL_MAX:
        return False
    d = np.diff(v)
    if not (np.all(d > 0) or np.all(d < 0)):
        return False
    step = np.median(np.abs(d))
    return 0.001 <= step <= 5.0 and (v.max() - v.min()) >= 5.0


def _numeric_column(df: pd.DataFrame, name) -> np.ndarray:
    return pd.to_numeric(df[name].astype(str).str.replace(",", ".", regex=False),
                         errors="coerce").to_numpy(float)


def load_sms_csv(path: str | Path, temp_col=None, time_col=None) -> SpectraSet:
    """Load a spectra CSV into a :class:`SpectraSet`.

    Parameters
    ----------
    path : CSV file path.
    temp_col, time_col : optional column names or indices of the temperature
        and time columns; when omitted they are detected from the headers.
    """
    path = Path(path)
    df = _read_table(path)
    headers = list(df.columns)
    hdr_vals = np.array([_to_float(h) for h in headers])
    spectral = np.isfinite(hdr_vals) & (hdr_vals >= WL_MIN) & (hdr_vals <= WL_MAX)

    # ---- transposed layout: wavelengths run down the first column ----------
    first = pd.to_numeric(df.iloc[:, 0].astype(str).str.replace(",", ".", regex=False),
                          errors="coerce")
    headers_are_axis = _looks_like_wavelength_axis(hdr_vals[spectral]) if spectral.sum() else False
    first_is_axis = _looks_like_wavelength_axis(first.to_numpy(float))
    if first_is_axis and (not headers_are_axis or first.notna().sum() > spectral.sum()):
        if True:
            wl = first.to_numpy(float)
            body = _numeric_frame(df.iloc[:, 1:])
            spectra = body.to_numpy(float).T  # (n_spectra, n_wavelengths)
            col_vals = np.array([_to_float(h) for h in body.columns])
            if np.isfinite(col_vals).all():
                time = col_vals.astype(float)        # headers are times
            else:
                time = np.arange(spectra.shape[0], dtype=float)
            temperature = np.full(spectra.shape[0], np.nan)
            order = np.argsort(wl)
            return SpectraSet(wl[order], time, temperature, spectra[:, order], str(path),
                              layout="transposed")
    if spectral.sum() < 10:
        raise ValueError(
            "Could not find wavelength columns. Headers should be wavelengths in nm "
            f"(first headers seen: {headers[:6]}). Run: python scripts/inspect_csv.py {path}"
        )

    # ---- standard layout: wavelengths across the header --------------------
    spec_cols = [h for h, ok in zip(headers, spectral) if ok]
    other = [h for h, ok in zip(headers, spectral) if not ok]
    wavelengths = hdr_vals[spectral]
    spectra = _numeric_frame(df[spec_cols]).to_numpy(float)
    n = spectra.shape[0]

    def pick(explicit, patterns):
        if explicit is not None:
            return headers[explicit] if isinstance(explicit, int) else explicit
        for h in other:
            if re.search(patterns, str(h), re.IGNORECASE):
                return h
        return None

    t_name = pick(temp_col, r"temp|sicak|sıcak|°c|degc|celsius")
    time_name = pick(time_col, r"time|zaman|sec|^t$|\(s\)")

    # candidates for positional fallback: drop obvious index columns
    remaining = [h for h in other if h not in (t_name, time_name)
                 and not str(h).lower().startswith("unnamed")]
    if t_name is None and time_name is None:
        if len(remaining) >= 2:            # original positional layout: Temp, Time
            t_name, time_name = remaining[0], remaining[1]
        elif len(remaining) == 1:
            time_name = remaining[0]
    elif t_name is None and remaining:
        t_name = remaining[0]
    elif time_name is None and remaining:
        time_name = remaining[0]

    time = _numeric_column(df, time_name) if time_name is not None else np.arange(n, dtype=float)
    temperature = (_numeric_column(df, t_name) if t_name is not None
                   else np.full(n, np.nan))
    if not np.isfinite(time).any():
        time = np.arange(n, dtype=float)

    # drop rows whose spectrum is entirely missing (trailing blank lines etc.)
    keep = np.isfinite(spectra).any(axis=1)
    layout = "standard" if (t_name is not None and time_name is not None) else "standard (partial)"
    return SpectraSet(wavelengths, time[keep], temperature[keep], spectra[keep], str(path),
                      layout=layout)
