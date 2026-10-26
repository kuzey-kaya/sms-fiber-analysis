"""The automatic choices (tracker, reference fringe, load steps) agree with the app on both runs."""

from pathlib import Path

import numpy as np
import pytest

from sms_analysis import load_sms_csv
from sms_analysis.auto import choose, spectral_noise

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "data.csv").exists(), reason="example run missing")
def test_cooldown_is_a_continuous_run_on_the_standard_tracker():
    data = load_sms_csv(ROOT / "data" / "data.csv")
    assert spectral_noise(data) < 0.004
    c = choose(data)
    assert not c.width_aware and not c.is_steps
    assert c.fringe == "dip@1542.3"                      # the published reference fringe
    assert c.table.loc[c.table["fringe"] == c.fringe, "locked_pct"].item() == 100.0


@pytest.mark.skipif(not (ROOT / "data" / "strain.csv").exists(), reason="strain run missing")
def test_strain_is_a_load_step_run_on_the_width_fit():
    data = load_sms_csv(ROOT / "data" / "strain.csv")
    assert spectral_noise(data) > 0.004
    c = choose(data)
    assert c.width_aware and c.is_steps
    assert c.fringe == "dip@1552.8"                      # the advisor's "min1555"
    assert len(c.plateaus) == 17 and int((c.step_values > 0).sum()) == 8
    assert np.nanmax(c.step_cond) == 400.0                # 50 g steps assumed
    assert np.isnan(c.step_cond[0]) and np.isfinite(c.step_cond).sum() > 800   # settle rows blanked
