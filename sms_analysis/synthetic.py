"""SIMULATED SMS spectra with a known phase profile — for testing only.

Nothing here is measured data.  The generator builds T(λ) = t1 + t2 +
2√(t1 t2) cos φ(λ, t) from a phase profile shaped like the real sensor's
(the cubic fitted to the first spectrum of data/data.csv, with λc ≈ 1525 nm),
and changes it frame by frame in two controlled ways:

* ``common_rad``  — the same phase added at every wavelength (what the
  2024-10-17 cooldown mostly does; new extrema are born at λc);
* ``tilt_rad``    — a change that grows linearly across the band, the part a
  phase difference between two wavelengths can see (what strain does in
  SM1E.2).  ``tilt_rad`` is the change of φ(λb) − φ(λa) itself.

Because both are known exactly, the phase-unwrapping readout and the peak
tracker can be checked against the truth over many fringe spacings.
"""

from __future__ import annotations

import numpy as np

from .io import SpectraSet

# Cubic Δφ(λ) of the first spectrum of data/data.csv, in (λ − 1525 nm), highest power first.
# Rounded from sms_analysis.phase.fit_phase on that spectrum (RMS 0.044 rad).
REAL_PROFILE = np.array([-3.656e-5, -1.4894e-2, 6.906e-3, 0.0])
LAMBDA_C0 = 1525.0


def simulate_run(
    n_frames: int = 300,
    common_rad: float = 3 * np.pi,
    tilt_rad: float = 0.0,
    lambda_a: float = 1545.0,
    lambda_b: float = 1570.0,
    wavelengths: np.ndarray | None = None,
    phi_max: float = 40.3 * np.pi,
    noise: float = 0.003,
    seed: int = 0,
) -> tuple[SpectraSet, np.ndarray]:
    """A SIMULATED run and the true readout φ(λb) − φ(λa) of every frame.

    The common phase and the tilt both grow linearly from 0 to their final
    values over the run; the "temperature" column is a 0→1 ramp (the fraction
    of the run), so a fit against it returns the total change.
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(1475.0, 1575.05, 0.1) if wavelengths is None else np.asarray(wavelengths, float)
    x = wl - LAMBDA_C0
    base = np.polyval(REAL_PROFILE, x)
    base = base - base.max() + phi_max                # absolute phase, maximum at λc
    span = lambda_b - lambda_a
    frac = np.linspace(0.0, 1.0, n_frames)
    spectra = np.empty((n_frames, len(wl)))
    truth = np.empty(n_frames)
    t1, t2 = 0.30, 0.25                               # gives T between ~0.27 and ~0.83 like the real run
    base_diff = np.interp(lambda_b, wl, base) - np.interp(lambda_a, wl, base)
    for i, f in enumerate(frac):
        phi = base + f * common_rad + f * tilt_rad * (wl - lambda_a) / span
        spectra[i] = t1 + t2 + 2 * np.sqrt(t1 * t2) * np.cos(phi) + rng.normal(0, noise, len(wl))
        truth[i] = base_diff + f * tilt_rad
    data = SpectraSet(wavelengths=wl, time=np.arange(n_frames) * 20.0, temperature=frac,
                      spectra=spectra, source="SIMULATED", layout="simulated")
    return data, truth
