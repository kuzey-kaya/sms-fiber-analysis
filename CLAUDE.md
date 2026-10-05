# CLAUDE.md — project context for Claude Code

This file is read automatically when Claude Code starts in this folder. It carries
everything from the earlier Cowork chat in which this project was built, so a new
session can continue without that chat.

## What this project is

Python replacement for an Igor Pro workflow used by Kuzey Kaya's advisor's group to
analyse **SMS (single-mode–multimode–single-mode) fiber-optic sensor spectra**.
A run is a CSV of transmission spectra recorded over time while temperature (or
strain) changes. The pipeline detects every peak/dip in every spectrum, tracks each
interference fringe over time with guards against fringe-hopping, fits wavelength
vs temperature per fringe (sensitivity in pm/°C), and renders the whole run as a
wavelength–time map plus 3-D views. Advisor has seen the results, liked them
("great analysis"), and asked for the generic data name (`data/data.csv`) and the
all-fringes export now in place.

Working language: code, comments, docs, figures and the report are **English**;
conversation with the owner is usually Turkish.

## Quick start

```bash
pip3 install -r requirements.txt
# put the CSV to analyse at data/data.csv (the 2024-10-17 cooldown run ships there)
python3 scripts/inspect_csv.py                     # how the file is interpreted
python3 scripts/run_analysis.py                    # fig1-7, tracked_all_*.csv, sensitivity_map.csv
python3 scripts/peak_inventory.py --interval 1200  # all peaks, trajectories, interval staircase
python3 scripts/staircase_views.py                 # fig9-12 staircase / fringe-hop views (Igor style)
python3 scripts/hero_figures.py                    # fig15-17 annotated maps
python3 scripts/spectral_3d.py                     # fig18-19 3-D views (+ HTML if plotly installed)
```
Interactive app (FringeLab, draft): `python3 launch.py`, or double-click `run_app.bat` (Windows) /
`run_app.command` (macOS); development: `streamlit run app.py`.

Every script takes an optional CSV path as first argument and `--outdir` (default
`figures/`). Scripts must run from this folder (they import `sms_analysis/` next to them).

## Data format and loader (`sms_analysis/io.py`)

Standard layout: col 0 temperature (°C), col 1 time (s), remaining columns
transmittance with the **wavelength in nm as the header**; one row per spectrum.
`load_sms_csv` also handles: extra/index columns, temp/time columns anywhere (found
by header regex `temp|sicak|°c` and `time|zaman|sec`), missing temperature
(`SpectraSet.has_temperature == False` → calibration steps are skipped, maps/tracking
still run), **transposed files** (rows = wavelengths, columns = spectra; detected by
`_looks_like_wavelength_axis` on the first column), `;`/tab separators and decimal
commas. Tested on six layout variants on 2026-10-04. If a new file fails, run
`scripts/inspect_csv.py <file>` first; the explicit override is
`load_sms_csv(path, temp_col=..., time_col=...)`.

The shipped run `data/data.csv` (originally `smsdata20241017.csv`): 730 spectra,
1475–1575 nm at 0.1 nm, cooldown 44.1 → 22.8 °C over 4.8 h, one spectrum per ~23 s.
It is a continuous cooldown — no setpoint steps.

## Package map

- `sms_analysis/io.py` — `load_sms_csv`, `SpectraSet` (wavelengths, time, temperature, spectra).
- `sms_analysis/peaks.py` — `detect_extrema` (Savitzky–Golay window 11/order 3, `find_peaks`
  prominence 0.02, distance 5) and `parabolic_refine` (3-point parabola, guarded: |offset| > 1 → raw sample).
- `sms_analysis/tracking.py` — `track_feature` (±1.5 nm window, rejects window-edge hits and
  jumps > 0.5 nm as lost lock, refines on the smoothed curve; optional `smoothed=` pre-smoothed
  spectra), `track_all_extrema` (all 24 fringes of frame 0; smooths each spectrum once and passes
  it down — ~7 s instead of ~18 s, outputs byte-identical; takes `max_step`), `track_feature_legacy` (exact reproduction of the old Colab tracker:
  4 nm window, unguarded refine on raw data → fringe-hops; optional median filter),
  `track_band_extremum` (strongest extremum in a fixed band → staircase from identity hops).
- `sms_analysis/analysis.py` — `fit_sensitivity` (linregress λ vs T), `sensitivity_map` (table; returns
  empty frame when no temperature).
- `sms_analysis/plotting.py`, `igor_style.py` (Igor Pro look: red λ left axis, blue T right axis),
  `staircase.py` (temperature-bin plateau averages).
- `scripts/run_analysis.py` — main pipeline; writes `tracked_all_long.csv` (fringe, time, temperature,
  wavelength, value) and `tracked_all_wide.csv` (one column per fringe) for ALL tracked fringes,
  `tracked_feature.csv` for the detailed one, `sensitivity_map.csv`, fig1–7.
- `scripts/peak_inventory.py` — `all_peaks_long.csv` (every extremum of every spectrum),
  `all_peaks_tracked_wide.csv`, `fringe_directions.csv`, fig13 (trajectory map), fig14 (interval-sampled staircase).
- `scripts/staircase_views.py` — regenerates fig9–12 in the Igor style: `fig9_staircase_3C.png`
  (`staircase_bin` + `plot_staircase`, 3 °C bins on the dip starting at 1542.3 nm; skipped without
  temperature), `fig10_legacy_igor.png` (`track_feature_legacy` defaults), `fig11_multistep_igor.png`
  (`search_window=8`, `median_kernel=15`), `fig12_desc_med7.png` (`track_band_extremum`, band
  1475–1535, `mode="peak"`, `median_kernel=7`). The other fig9–12 variants and their CSVs in
  `figures/` (`fig9_staircase.png`, `fig10_legacy_tracking.png`, `fig11_multistep_tracking.png`,
  `fig12_staircase_*.png`, `staircase_plateaus.csv`, `band_*_staircase.csv`,
  `tracked_feature_legacy*.csv`) are from the Cowork session and have no generating script.
- `scripts/hero_figures.py` — fig15 (map + tracked fringes + λc), fig16 (map indexed by temperature:
  each fringe a straight calibration line), fig17 (three zooms).
- `scripts/spectral_3d.py` — fig18 (floor map + fringe curves rising with temperature), fig19
  (transmittance as height over wavelength × temperature — the advisor's favourite), optional plotly HTML.
- `scripts/simulate_staircase.py` — clearly labelled SIMULATED stepped-temperature demo; never present as data.
- `scripts/inspect_csv.py` — loader diagnosis.
- `app.py` — **FringeLab**, the Streamlit + Plotly interface (draft, 2026-10-04). Sidebar: example
  run or uploaded file, condition name/unit + column overrides (strain/load runs), wavelength crop,
  detection and tracking settings (defaults = the script defaults). Tabs: spectral map (time /
  spectrum number / condition axis; transmittance, change from first spectrum, per-spectrum 0–1;
  tracked-fringe overlay), spectrum & peaks, fringe tracking (all shifts + one fringe Igor-style +
  calibration), sensitivity, 3-D surface, export (CSV/JSON). Uses only package functions; results
  are cached per file + settings. "Tracked" in the app = locked ≥ 90 % of frames (21/24 on
  data/data.csv; the scripts' 22/24 counts R² ≥ 0.90).
- `launch.py` — starts Streamlit headless on a free port and opens the browser (avoids Streamlit's
  first-run e-mail prompt). `run_app.bat` / `run_app.command` — create `.venv`, install
  `requirements.txt` once (marker `.venv/installed.txt`), then call `launch.py`.
  `launch.py` also prints the LAN address (`lan_address()`) for opening the app from a tablet on
  the same Wi-Fi (Streamlit listens on all interfaces; no password). `run_app.command` is stored
  executable in git (mode 100755) and pinned to LF by `.gitattributes`.
  `.claude/launch.json` — preview config for Claude Code (port 8765).
- `HANDBOOK.md` — how to get and run the project per device: Windows, macOS/Linux, tablets
  (same-Wi-Fi address, Streamlit Community Cloud, GitHub Codespaces), own data, scripts, updating,
  troubleshooting. Its status table says what was actually tried; keep it honest when that changes.
- Hosted app: https://fringelab.streamlit.app (Streamlit Community Cloud, deployed by the owner
  2026-10-05 from `main`/`app.py`; public, redeploys on every push to `main`).
- Repository: https://github.com/kuzey-kaya/sms-fiber-analysis (`main`; git set up 2026-10-05).
  Commit and push after each change.

## Physics the code relies on

SMS sensor = modal Mach–Zehnder interferometer, T(λ) = t₁ + t₂ + 2√(t₁t₂)cos Δφ(λ).
Critical wavelength (dispersion turning point) λc ≈ 1525 nm in this sensor. Fringes
below λc shift to longer λ on cooling (negative sensitivity, −88…−320 pm/°C), fringes
above λc to shorter λ (+115…+403 pm/°C); magnitude grows toward λc; fringes within
~1515–1535 nm broaden/merge and are lost after ~40 min. A real fringe moves < 0.1 nm per
frame, so larger jumps are tracker artefacts.

## Key results on data/data.csv (keep consistent if quoted)

22 of 24 fringes tracked over all 730 frames with R² ≥ 0.90 (most ≥ 0.997).
Fringe near 1493 nm: −141 pm/°C vs the group's published −142 pm/°C near 1495 nm
(Chen et al., Optica Sensing Congress JM4A.8, 2025) — the validation against Igor.
Most sensitive trackable fringe: dip starting at 1542.3 nm, +359.3 ± 0.7 pm/°C, R² 0.997.

## Decisions and their reasons

- Hop rejection in `track_feature` exists because the old notebook's 11 nm "step" at
  t ≈ 7660 s was the tracker sliding into the λc trough (unguarded parabolic refine on raw
  data threw the position, the 4 nm window followed). Step height = fringe spacing, not physics.
- The advisor's staircase slide (peak λ 1545→1560 nm, labels 0 g…300 g, flat ~20 °C) is the
  **strain** experiment (50 g every 5 min; Salik et al. SM1E.2 2025). The cooldown CSV cannot
  produce it; step-like traces from this data come only from sampling/binning (fig9, fig14) or
  fringe hops (fig10–12). Given the strain CSV, the pipeline reproduces the slide unchanged.
- Peak definition: local extremum of the smoothed curve + 3-point parabolic refinement.
- Linear λ(T) fit per fringe; slight curvature at the ends is physical (sensitivity depends on
  distance from λc). Piecewise/quadratic fit is an easy extension if asked.

## Literature anchors (already cited in the report and deck)

Salik 2012 IEEE PTL 24(7) 593 (critical wavelength); Zhou 2014 IEEE PTL 26(21) 2185;
Lu 2019 IEEE Sensors J 19(5); Xu 2023 Sensors 23, 1725 (DTP review, doi 10.3390/s23031725);
Hu 2025 Light Sci Appl 14, 392 (LSTM whole-spectrum demodulation); Salik 2025 Optica Sensing
SM1E.2 (phase unwrapping, wide dynamic range); Chen 2025 Optica Sensing JM4A.8 (probe sensor,
−0.142 nm/°C).

## Deliverables that exist outside this folder

- Slide deck "SMS Fiber Sensor Analysis in Python" (15 slides, Claude artifact, owner's account).
- `SMS_Fiber_Sensor_Analysis_Report.pdf` (in this folder, 14 pages; its reproducibility
  section still names the original file `smsdata20241017.csv` — now `data/data.csv`).
- A Turkish/English speaker script (markdown) kept by the owner.

## Open questions for the advisor (unanswered as of 2026-10-04)

1. Which dataset made the staircase slide; can the strain CSV be shared?
2. Exact meaning of "tracking at intervals" (time, temperature, or Igor search window)?
3. Reference fringe — the one near 1546 nm from the Igor graph title "peak1546"?
4. How Igor defined a peak (raw extremum vs smoothed/fitted)?

## Likely next work

- FringeLab (`app.py`) is a first draft: the macOS launcher and the double-click launchers'
  first-run install have not been tried on a real second machine yet; no app icon/name decided
  ("FringeLab" is a placeholder, one constant `APP_NAME`); candidates to add: staircase/legacy
  views, choice of reference spectrum, side-by-side comparison of two files, packaged installer.
- Port the phase-unwrapping method (SM1E.2) onto `SpectraSet`/tracks.
- Run on new datasets: copy the file to `data/data.csv` (or pass its path), run
  `inspect_csv.py`, then the four scripts.
- Keep README's figure table and this file in sync when adding outputs.

## Testing checklist before handing anything to the advisor

`python3 scripts/inspect_csv.py` → all four scripts with no arguments → `ls figures` shows
fig1–fig19 and the CSVs → numbers above unchanged for data/data.csv.
