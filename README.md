# SMS Fiber Sensor Analysis

Python toolkit for analyzing single-mode–multimode–single-mode (SMS) fiber
optic sensor spectra: peak/dip detection, fringe tracking over time, and
temperature calibration. Replaces the earlier Igor Pro / Colab workflow.

## Physics background (why peaks move "irregularly")

An SMS sensor is a modal Mach-Zehnder interferometer. Its transmission
spectrum `T(λ) = t1 + t2 + 2√(t1·t2)·cos[Δφ(λ)]` is a fringe pattern with a
**critical wavelength λc** (dispersion turning point, ≈1525 nm in this data)
where fringes become very wide. When temperature changes:

- fringes **below** λc shift one way, fringes **above** λc shift the other way
  (opposite signs of sensitivity);
- sensitivity **magnitude grows** the closer a fringe is to λc
  (here: ~90 pm/°C at the band edges up to ~400 pm/°C near λc);
- fringes very close to λc deform and merge, so they cannot be tracked
  reliably over a wide temperature range.

So "the peaks" do not move as one: each fringe must be tracked individually,
and its own λ(T) line is the calibration. See Salik et al. and Chen et al.,
Optica Sensing Congress 2025 (papers SM1E.2 and JM4A.8).

## Data format

CSV, one row per acquired spectrum:

| column 0 | column 1 | columns 2… |
|---|---|---|
| Temperature (°C) | Time (s) | transmittance; header = wavelength in nm |

The loader also accepts extra/index columns, temperature and time columns in any
position (found by header name), files without a temperature column (maps and
tracking still run; calibration is skipped), transposed files (rows =
wavelengths), `;`/tab separators and decimal commas. `scripts/inspect_csv.py`
shows how a file is read.

`data/data.csv` (the 2024-10-17 run, originally `smsdata20241017.csv`): 730 spectra, 1475–1575 nm at 0.1 nm steps,
cooldown 44 °C → 22.8 °C over ~4.8 h, one spectrum every ~23 s.

## Interactive app: FringeLab (draft)

A browser-based interface on top of the same package: load a spectra file, adjust
the settings in the sidebar, and every view updates immediately. Nothing is sent
anywhere — the app runs on your own computer and opens in your browser (other
devices on the same network can open it too while it runs; see the handbook).

Step-by-step instructions for Windows, macOS, Linux and tablets (iPad), plus
updating and troubleshooting, are in [HANDBOOK.md](HANDBOOK.md).

**Install and start (Windows and macOS)**

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/)
   (Windows: tick "Add python.exe to PATH" in the installer).
2. Copy this folder to the computer.
3. Windows: double-click `run_app.bat`.
   macOS: double-click `run_app.command` (the first time, right-click → Open; if
   macOS says it cannot be run, do `chmod +x run_app.command` once in Terminal).

The first start creates a private environment in `.venv` and installs the
requirements (a few minutes, needs internet, only once). After that the app
opens in the browser within seconds. Close the launcher window to stop it.
Manual alternative: `pip install -r requirements.txt`, then `python launch.py`.

**What it shows**

| tab | content |
|---|---|
| Spectral map | the whole run as a wavelength × time heatmap (fig3), with the tracked fringes overlaid; vertical axis can be time, spectrum number or the condition; can also show the change from the first spectrum or each spectrum scaled 0–1 |
| Spectrum & peaks | any single spectrum with its detected peaks/dips; optional overlay of evenly spaced spectra |
| Fringe tracking | shift of every tracked fringe; one fringe in detail in the Igor style with its calibration — linear and/or quadratic fit with the residuals, local sensitivity at both ends of the run |
| Sensitivity | sensitivity of every fringe vs wavelength, and the fringe table with the linear and quadratic RMSE and the local sensitivity at the coldest and warmest temperature |

The start page (and the *How the file was read* panel above the tabs) shows the
expected file layout with a downloadable template CSV and a report of how the
loaded file was interpreted.
| 3-D view | transmittance as height over wavelength × time/condition (rotatable) |
| Movie | the run animated frame by frame (spectrum + extrema, map cursor, tracked fringe) with play/pause and a time slider; GIF export for slides |
| Export | tracked fringes (wide/long CSV), fringe table, the settings used |

Sidebar settings: data source (the example run `data/data.csv` or an uploaded
file), condition name/unit and column overrides (for runs where strain, load, …
was varied instead of temperature), wavelength range, detection (prominence,
separation, smoothing) and tracking (search window, jump rejection). The
defaults are the values used by the scripts below.

## Install & run (scripts)

```bash
pip install -r requirements.txt
python scripts/inspect_csv.py                 # optional: shows how data/data.csv is interpreted
python scripts/run_analysis.py                # uses data/data.csv, writes to figures/
```

To analyse another run, copy it to `data/data.csv` (any layout the loader
understands — see "Data format") or pass its path:
`python scripts/run_analysis.py path/to/other.csv --outdir other_figures`.

Useful options:

```
--track 1542.3        track a specific fringe (nm in the first spectrum)
--track-kind dip      'dip' (minimum) or 'peak' (maximum)
--prominence 0.02     peak-detection prominence (transmittance units)
--search-window 1.5   tracking half-window in nm
```

With no `--track`, the script tracks *every* fringe, builds the sensitivity
map, and auto-selects the most sensitive well-tracked fringe for the
detailed plots.

### Peak inventory (advisor's procedure for slowly-varying data)

```bash
python scripts/peak_inventory.py --interval 1200
```

Exports *every* detected peak/dip of every spectrum (`all_peaks_long.csv`,
plus the identity-resolved `all_peaks_tracked_wide.csv`), tabulates the drift
direction and sensitivity of each fringe (`fringe_directions.csv`), draws the
trajectory map of all extrema vs time (`fig13`), and shows the tracked
positions sampled every `--interval` seconds as staircases (`fig14`).

### Staircase views (figs 9–12)

```bash
python scripts/staircase_views.py
```

Regenerates the Igor-style step-like views of the cooldown: temperature-bin
plateaus (`fig9`), the legacy Colab tracker with its fringe-hop (`fig10`,
`fig11`) and the strongest peak in a fixed band (`fig12`).

### Movie of the run (Igor-style movie export)

```bash
python scripts/make_movie.py                    # figures/movie_run.gif, every 5th spectrum, 12 fps
python scripts/make_movie.py --format mp4 --every 3 --track 1542.3   # MP4 needs ffmpeg on the PATH
```

Each frame shows the current spectrum with its detected peaks and dips (the
first spectrum as a grey ghost), the spectral map with a cursor at the current
time, and the most sensitive tracked fringe drawn up to that time in the Igor
colours, with time and temperature stamped. The same animation runs in the app's
*Movie* tab with play/pause and a time slider, plus a "Render as GIF" button.

### Presentation and 3-D figures

```bash
python scripts/hero_figures.py     # fig15-17: annotated map, map vs temperature, zooms
python scripts/spectral_3d.py      # fig18-19: 3-D views (+ interactive HTML with plotly)
```

## Outputs

| file | content |
|---|---|
| `fig1_spectrum_peaks.png` | first spectrum with detected peaks/dips |
| `fig2_spectra_overlay.png` | spectra at several temperatures |
| `fig3_spectral_map.png` | whole run as a wavelength–time heatmap |
| `fig4_tracking_vs_time.png` | tracked fringe + temperature vs time (Igor-style) |
| `fig5_correlation.png` | wavelength vs temperature with linear fit |
| `fig6_sensitivity_map.png` | sensitivity of every fringe vs wavelength |
| `fig7_igor_style.png` | the tracking graph rendered in the Igor Pro house style |
| `fig9_staircase*.png` | tracked fringe averaged in 3 °C / 2 °C temperature bins, drawn as plateaus (`scripts/staircase_views.py` writes `fig9_staircase_3C.png`: dip starting at 1542.3 nm, 3 °C bins) |
| `fig10_legacy_*.png` | exact reproduction of the original Colab tracking (with its fringe-hop step at t ≈ 7660 s); `scripts/staircase_views.py` writes `fig10_legacy_igor.png` |
| `fig11_multistep_*.png` | legacy tracker with a wider window + median filter (several fringe-hop steps); `scripts/staircase_views.py` writes `fig11_multistep_igor.png` (8 nm window, median 15) |
| `fig12_*.png` | strongest extremum inside a fixed band (`track_band_extremum`): identity hops between fringes → staircase; `scripts/staircase_views.py` writes `fig12_desc_med7.png` (strongest peak in 1475–1535 nm, median 7) |
| `fig13_peak_trajectories.png` | every detected extremum of every spectrum vs time — each streak is one fringe drifting |
| `fig14_interval_staircase*.png` | tracked positions sampled every 1200 s and held → staircase view of the slow drift |
| `fig15_spectral_map_annotated.png` | presentation version of the spectral map: tracked fringes overlaid, λc marked, temperature axis (`scripts/hero_figures.py`) |
| `fig16_spectral_map_vs_temperature.png` | the same map indexed by temperature — each fringe becomes a straight calibration line |
| `fig17_spectral_map_zooms.png` | three zooms: below λc, the λc region, above λc |
| `fig18_3d_spectral_surface.png` | 3-D: spectral map on the floor, each tracked fringe rising with temperature, cooling curve on the back wall (`scripts/spectral_3d.py`) |
| `fig19_3d_fringe_landscape.png` | 3-D: transmittance as height over wavelength × temperature — ridges/valleys tilting across λc |
| `fig18_3d_interactive.html` | rotatable version of fig18; written only if `plotly` is installed (`pip3 install plotly`) |
| `movie_run.gif` / `.mp4` | the run animated: spectrum + extrema per frame, map with time cursor, tracked fringe growing (`scripts/make_movie.py`) |
| `fig20_calibration_models.png` | linear vs quadratic calibration of the detailed fringe with both residual series (`run_analysis.py`) |
| `tracked_all_long.csv` | every tracked fringe: fringe, time, T, wavelength, value (one row per fringe per frame) |
| `tracked_all_wide.csv` | every tracked fringe: one wavelength column per fringe, one row per frame |
| `tracked_feature.csv` | time, T, wavelength, value of the fringe used for the detailed plots |
| `sensitivity_map.csv` | per-fringe sensitivity table: linear fit (slope, R², std. error) plus the curved-calibration columns — RMSE of the linear and quadratic fits (pm), curvature (pm/°C²), local sensitivity at the coldest and warmest temperature, F-test p-value |
| `all_peaks_long.csv` | every detected peak/dip of every spectrum (row, time, T, kind, wavelength, value) |
| `all_peaks_tracked_wide.csv` | same, identity-resolved: one column per fringe |
| `fringe_directions.csv` | per fringe: start/end wavelength, drift direction on cooling, sensitivity, R² |
| `staircase_plateaus.csv` | plateau means behind fig9 (usable as calibration points) |

Note: figs 9–12 and 14 are different *representations* of the same
continuous cooldown. The steps in figs 10–12 are fringe-hops of the tracker
(step height = fringe spacing), not physical jumps of the sensor; the steps
in figs 9 and 14 are binning / sampling of the smooth drift. A physical
staircase like the strain experiment's requires a stepped input.

## Package layout

```
sms_analysis/
  io.py         load CSV -> SpectraSet (wavelengths, time, temperature, spectra)
  peaks.py      extrema detection + parabolic sub-sample refinement
  tracking.py   windowed fringe tracking with fringe-hop rejection
  analysis.py   linear and quadratic λ(T) fits, curvature test, per-fringe sensitivity table
  plotting.py   all figures
scripts/
  run_analysis.py   end-to-end pipeline (CLI)
```

## Key results on `data/data.csv` (2024-10-17 run)

- 22 of 24 fringes tracked over the full run with R² ≥ 0.90 (most ≥ 0.997).
- Sign flip across λc ≈ 1525 nm: −88…−320 pm/°C below, +115…+403 pm/°C above.
- Fringe near 1493 nm: **−141 pm/°C**, matching the −142 pm/°C published for
  ~1495 nm in the group's inline-SMS paper — a good cross-check that the
  Python pipeline reproduces the Igor analysis.
- Most sensitive reliably-tracked fringe: dip starting at 1542.3 nm,
  **+359 pm/°C**, R² = 0.997 over the full 21 °C range.
- The calibration of that fringe is visibly curved: a quadratic fit halves the
  residual (RMSE 107 → 49 pm) and gives a local sensitivity of +414 pm/°C at
  22.8 °C falling to +283 pm/°C at 44.1 °C — the fringe moves toward the
  critical wavelength as it cools, so its sensitivity grows. Fringes far from
  λc (e.g. the dip at 1567.7 nm) are linear to within the noise.
