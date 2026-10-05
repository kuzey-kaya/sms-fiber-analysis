"""Analysis toolkit for SMS (single-mode-multimode-single-mode) fiber sensor data.

Pipeline: load spectra from CSV -> detect spectral extrema -> track features
over time -> correlate wavelength shift with temperature -> plot.
"""

from .io import SpectraSet, load_sms_csv
from .peaks import detect_extrema, parabolic_refine
from .tracking import track_feature, track_all_extrema
from .analysis import fit_sensitivity, sensitivity_map

__all__ = [
    "SpectraSet",
    "load_sms_csv",
    "detect_extrema",
    "parabolic_refine",
    "track_feature",
    "track_all_extrema",
    "fit_sensitivity",
    "sensitivity_map",
]
