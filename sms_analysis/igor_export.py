"""Igor Text (.itx) export, so FringeLab's results open in Igor Pro as ordinary waves.

The advisor's Igor experiment (``sms-strain-2026-10-15-analysis``, seen in his screen recording)
names things like this: ``wave0`` is the wavelength axis, ``n1``/``n850`` are single spectra,
``trackpeaknew("n", 1, 1, 914, wave0, "min1489", temp, tim, 274, 353, -1)`` writes
``peakmin1489`` / ``timemin1489`` / ``tempmin1489`` (the followed minimum, its time and
temperature), and ``shift1555`` / ``mass`` make the ``display shift1555 vs mass`` graph. The
functions here write Igor Text files with exactly those names, so ``File → Load Waves → Load Igor
Text`` puts FringeLab's tracks, FindPeak tracks and spectra straight into an Igor experiment.

Igor Text format (Igor Pro manual, "Igor Text Files"): first line ``IGOR``; a ``WAVES`` line
declares one or more 1-D waves (``/D`` double precision, ``/O`` overwrite); ``BEGIN`` … ``END``
holds one row per point, one column per wave; ``NaN`` is written literally; lines starting with
``X`` are commands executed after loading (used here for ``SetScale`` and ``Note``). Wave names
must be ≤ 31 characters of letters, digits and underscores and must not start with a digit.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

MAX_NAME = 31


def wave_name(name: str) -> str:
    """A legal standard Igor wave name: letters, digits, underscores; not starting with a digit;
    at most 31 characters. ``peak@1552.8`` → ``peak1552_8``."""
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).replace("@", "")).strip("_")
    s = re.sub(r"_+", "_", s)
    if not s:
        s = "wave"
    if s[0].isdigit():
        s = "w" + s
    return s[:MAX_NAME]


def track_names(kind: str, start_nm: float, suffix: str = "") -> dict[str, str]:
    """The advisor's naming for a followed extremum: ``peakmin1555``, ``timemin1555``,
    ``tempmin1555`` and ``shift1555`` for a dip ("min") starting near 1555 nm; ``max`` for a peak.
    ``suffix`` lets several tracks of the same fringe coexist, as in his ``min1576a``."""
    tag = ("min" if kind.startswith("dip") or kind == "min" else "max") + f"{int(round(start_nm))}{suffix}"
    return {"wavelength": f"peak{tag}", "time": f"time{tag}", "temperature": f"temp{tag}",
            "shift": f"shift{int(round(start_nm))}{suffix}"}


def _fmt(v) -> str:
    if v is None:
        return "NaN"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NaN"
    if not np.isfinite(f):
        return "NaN"
    return repr(f) if abs(f) < 1e15 else f"{f:.10g}"


def igor_text(waves: dict[str, Iterable], notes: dict[str, str] | None = None,
              scales: dict[str, tuple[float, float, str]] | None = None) -> str:
    """One Igor Text file holding the given 1-D waves (name → values). Waves of equal length share
    one ``WAVES`` block; the others get their own. ``notes`` adds ``Note`` commands,
    ``scales`` adds ``SetScale/P x start, delta, "units"`` for a wave."""
    clean = {wave_name(k): np.asarray(list(v) if not isinstance(v, np.ndarray) else v, dtype=float).ravel()
             for k, v in waves.items()}
    if len(clean) != len(waves):
        raise ValueError("wave names collide after being made Igor-legal")
    lines = ["IGOR"]
    done: set[str] = set()
    for name, arr in clean.items():
        if name in done:
            continue
        group = [n for n, a in clean.items() if n not in done and a.shape == arr.shape]
        done.update(group)
        lines.append("WAVES/D/O " + ", ".join(group))
        lines.append("BEGIN")
        cols = [clean[n] for n in group]
        for i in range(arr.shape[0]):
            lines.append("\t" + "\t".join(_fmt(c[i]) for c in cols))
        lines.append("END")
    for name, (start, delta, units) in (scales or {}).items():
        lines.append(f'X SetScale/P x {_fmt(start)},{_fmt(delta)},"{units}", {wave_name(name)}')
    for name, text in (notes or {}).items():
        safe = str(text).replace('"', "'").replace("\r", " ").replace("\n", " ")
        lines.append(f'X Note {wave_name(name)}, "{safe}"')
    return "\n".join(lines) + "\n"


def spectra_itx(wavelengths: np.ndarray, spectra: np.ndarray, time=None, temperature=None,
                rows: Iterable[int] | None = None, source: str = "") -> str:
    """The run as the advisor keeps it in Igor: ``wave0`` = wavelength axis, ``n0``, ``n1``, … one
    wave per spectrum (``rows`` picks a subset), plus ``tim`` and ``temp`` for the frame times and
    conditions — the inputs of his ``trackpeaknew``."""
    spectra = np.asarray(spectra, float)
    idx = list(range(spectra.shape[0])) if rows is None else [int(r) for r in rows]
    waves: dict[str, Iterable] = {"wave0": np.asarray(wavelengths, float)}
    for r in idx:
        waves[f"n{r}"] = spectra[r]
    if time is not None:
        waves["tim"] = np.asarray(time, float)[idx]
    if temperature is not None:
        waves["temp"] = np.asarray(temperature, float)[idx]
    notes = {"wave0": f"wavelength axis (nm){' from ' + source if source else ''}; n<k> = spectrum k, "
                      f"tim/temp = its time (s) and condition"}
    return igor_text(waves, notes=notes)


def tracks_itx(tracks: dict[str, "object"], time, condition, cond_name: str = "temperature",
               mass=None, source: str = "") -> str:
    """Every tracked fringe as ``peakmin1552``-style waves plus ``tim``, ``temp`` and, on a load
    run, ``mass``; ``tracks`` maps ``kind@start`` labels to frames with a ``wavelength`` column."""
    waves: dict[str, Iterable] = {"tim": np.asarray(time, float)}
    waves["temp" if cond_name.lower().startswith("temp") else wave_name(cond_name)] = np.asarray(condition, float)
    if mass is not None:
        waves["mass"] = np.asarray(mass, float)
    used: dict[str, int] = {}
    notes = {}
    for label, tr in tracks.items():
        kind, start = label.split("@")
        base = track_names(kind, float(start))["wavelength"]
        n = used.get(base, 0)
        used[base] = n + 1
        name = base + ("" if n == 0 else "abcdefghij"[min(n - 1, 9)])
        waves[name] = np.asarray(tr["wavelength"], float)
        notes[name] = f"FringeLab track of {label}{' from ' + source if source else ''}; NaN = lock lost"
    return igor_text(waves, notes=notes)


def findpeak_itx(kind: str, start_nm: float, time, wavelength, temperature, shift=None, mass=None,
                 call: str = "", suffix: str = "") -> str:
    """One FindPeak track with the advisor's names (``peakmin1576a`` …); ``shift``/``mass`` make
    his ``display shift1576 vs mass`` graph."""
    names = track_names(kind, start_nm, suffix)
    waves: dict[str, Iterable] = {names["wavelength"]: wavelength, names["time"]: time,
                                  names["temperature"]: temperature}
    if shift is not None:
        waves[names["shift"]] = shift
    if mass is not None:
        waves["mass"] = mass
    notes = {names["wavelength"]: f"FringeLab FindPeak track{': ' + call if call else ''}; NaN = no peak in range"}
    return igor_text(waves, notes=notes)


def parse_igor_text(text: str) -> dict[str, np.ndarray]:
    """Read back the 1-D waves of an Igor Text file (for the tests and for round trips)."""
    waves: dict[str, np.ndarray] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith("WAVES"):
            names = [n.strip() for n in line.split(None, 1)[1].split(",")]
            i += 1
            assert lines[i].strip().upper() == "BEGIN"
            rows = []
            i += 1
            while lines[i].strip().upper() != "END":
                rows.append([float("nan") if v.upper() == "NAN" else float(v) for v in lines[i].split()])
                i += 1
            block = np.asarray(rows, float).reshape(-1, len(names))
            for j, n in enumerate(names):
                waves[n] = block[:, j]
        i += 1
    return waves
