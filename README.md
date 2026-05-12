## Citation
If you use this code in academic work, please cite the repository using the `CITATION.cff` file.

## AI assistance
Parts of this codebase were drafted with assistance from ChatGPT (OpenAI) and subsequently reviewed/edited by the author.

# Muon finder for Renishaw WDF Raman maps

Prototype tool to detect cosmic ray (muon) artifacts in Raman maps.
Loads `.wdf`, performs 1D grayscale morphology along spectral axis (opening + top-hat),
creates a score map, extracts narrow spike segments, and provides a hover viewer.

## Install
pip install -r requirements.txt

## Run
python muon_finder.py --input /path/to/map.wdf

### Run with config file
```bash
python muon_finder.py --config config.json
```

### Optional: target only selected spectra by coordinates
Prepare a CSV with headers `y,x`, e.g.:

```csv
y,x
108,144
112,87
34,201
```

Then run:

```bash
python muon_finder.py --input /path/to/map.wdf --coords-csv /path/to/targets.csv
```

This mode extracts/despikes only specified coordinates and can render a compact (near-square) viewer layout
from selected points. 

### Outputs and debug report
Use config keys:
- `save_npz_path`
- `save_spikes_csv_path`
- `debug_report_path`

The debug report includes aggregate counts and (optionally) per-spectrum details for selected coordinates.
Per-spike debug fields include shape descriptors computed primarily from morphological gradient
(e.g. `rise_slope_z`, `fall_slope_z`, `plateau_width_90`, `gradient_max_z`), plus `muon_score` and `peak_position_cm1`.
`rise_slope`/`fall_slope` are computed from first-difference of the gradient segment around the candidate.
`top_pixels` and `per_spectrum[*].n_spikes` are now based on the same de-duplicated spike view (`start/end` duplicates merged). 
`n_spikes_total_raw` is also reported for reference.

Additional anti-false-positive descriptors are available per spike (focused on distinguishing narrow true Raman peaks vs impulsive artifacts):
- support after core removal: `support_after_core_removal_ratio`, `residual_area_after_core_removal`, `residual_component_count_after_core`
- curvature support: `curv2_support_width`, `curv2_abs_integral`, `curv2_apex_to_surround_ratio`, `curv2_component_count`
- multi-scale top-hat: `multiscale_tophat_area_small`, `multiscale_tophat_area_medium`, `multiscale_tophat_area_large`, `multiscale_tophat_decay_ratio`, `multiscale_tophat_persistence`
- residual/component: `residual_component_count`, `residual_component_spacing`, `gradient_component_count`, `gradient_support_width`
- optional foot-shape ratios: `foot_width_ratio_left_30_70`, `foot_width_ratio_right_30_70`

You can switch feature extraction signal source for debug report via:
- `debug_feature_signal_source`: `"gradient"` (default) or  `"raw"`
- `merge_duplicate_segments`: `true/false` (applies to both debug report and viewer spike overlay merge)
- `feature_expand_to_gradient_foot`: `true/false` (expand feature interval from anchors toward local gradient "foot")
- `feature_foot_k_mad`: baseline threshold multiplier for foot detection (default `2.0`)
- `feature_foot_min_run`: minimum consecutive points under threshold to accept as foot (default `2`)
- `feature_window_method`: `"mad_run"` (default) or `"erosion_touch"` mode (default `5`).
- `boundary_minimum_source` `"raw"` (default) or `"gradient"` for splitting overlaps between neighboring peaks

When set to `raw`,slope/asymmetry/plateau/`gradient_max*`-named fields are computed from the raw spectrum in the candidate interval; `feature_source` in JSON reflects the selected source.
When `feature_expand_to_gradient_foot=true`, feature metrics are computed on the expanded interval and JSON includes
`feature_window_start`/`feature_window_end` fields so you can verify what was used.
The same expanded interval is used for viewer spike egde/band overlays, so green bands reflect the effective feature window.
If neighboring expanded windows overlap, boundaries are split at local minima of the selected `boundary_minimum_source`

### Interactive labeling helper (click-to-label)
You can label candidates directly in a raw-spectrum GUI (separate script):

```bash
python label_candidates_gui.py --report path/to/debug_report.json --input path/to/map.wdf  --out-csv labels.csv --start-index 0 --max-spectra 100
```

Controls:
- mouse click near dashed candidate line = toggle `is_muon` (red = selected as muon)
- `n` / Right Arrow = next spectrum
- `p` / Left Arrow = previous spectrum
- `s` = save labels CSV
- `q` = save and quit

### Quick helper: one spectrum by coordinates
```bash
python debug_explorer.py --report path/to/debug_report.json --mode candidates_in_spectrum --y 62 --x 15 --param muon_score --x-axis candidate_index
```

Many spectra as panels (each panel shows all candidates of one spectrum):
```bash
python debug_explorer.py --report path/to/debug_report.json --mode multi_spectra_panels --param muon_score --x-axis candidate_index
```

```bash
python debug_explorer.py --report path/to/debug_report.json --mode multi_spectra_panels --params rise_slope_z,fall_slope_z,prominence_slope_z --x-axis candidate-index
```

To chunk panel visualization:
```bash
python debug_explorer.py --report path/to/debug_report.json --mode multi_spectra_panels --param muon_score --max-spectra 25 --start-index 50
```

Highlight points above threshold (e.g. show values > 1.2 in red)
```bash
python debug_explorer.py --report path/to/debug_report.json --mode multi_spectra_panels --param muon_score --max_spectra 25 --threshold value 1.2 --threshold-mode above
```

### Optional preprocessing: spectral resampling
Config keys:
- `resample_enabled` (bool)
- `resample_factor` (int, default 2)

This linearly upsamples each spectrum before morphology extraction.

Viewer spike overlays
- peak markers (`spike_peaks`) are dashed red vertical lines
- segment edges (`spike_edges`) are dashed green vertical lines
- segment span (`spike_band`) is a light green background band rendered behind curves
- all of the above can be toggled via checkboxes in viewer
- optional overlay `interest_metrics` prints per-spike labels at peak positions for:
- `pcse` = `peak_curvature_extreme`
- `mtdr` = `multiscale tophat_decay_ratio`
- `c2w` = `curv2_support_width`
- viewer now also supports `gradient_d1` and `gradient_d2` curves (first/second derivative of gradient), `raw_d2` (second derivative of raw), `raw_d3` (third derivative of raw), and the curvature overlays `pcs_d2` and `pcse`, all off by default

### Correlation helper for labeled candidates
Prepare labels CSV columns: `y,x,peak_index,is_muon`

You can auto-generate a template from `debug_report.json` and only fill the last column manually:

```bash
python make_labels_csv.py --report out/debug_report.json --out-csv labels.csv
```

Example `labels.csv`:

```csv
y,x,peak_index,is_muon
62,15,431,1
62,15,517,0
63,15,408,1
80,104,221,0
```

Notes
- `y,x` must match coordinates present in `debug_report.json`
- `peak_index` must match candidate `peak_index` exactly (same spectral sample index)
- `is_muon`: `1` = muon/cosmic spike, `0` = not muon

```bash
python debug_stats.py --report debug_report.json --labels-csv labels.csv --out-json corr.json
```

Optional modeling flags:
- `--logreg-quadratic` adds univariate quadratic logistic fit (`z` and `z^2`)
- `--mi-bins 10` sets quantile-bin count for mutual information estimate
- `--plots-dir out/plots --top-k 12` writes ranking plot (`feature_auc_ranking.png`)
- `--bootstrap-iters 1000 --bootstrap-seed 42` adds bootstrap AUC CI and plot (`feature_auc_bootstrap_ci.png`)
- `--plots-dir ...` also writes `features_auc_mi_scatter.png` (2D map: AUC vs mutual information)
- `--interactive-scatter` opens interactive hover scatter with highlighted hovered feature
- `--plot-features rise_slope_z,fall_slope_z` writes logistic regression curve PNG files for selected features
- by default weak legacy features are hidden from ranking; use `--include-all-features` to include all numeric features
- `--plots-dir ...` also writes feature redundancy heatmaps: `feature_corr_pearson.png`, `feature_corr_spearman.png`

`debug_stats.py` now reports for each feature:
- Pearson and Spearman correlation to `is_muon`
- `transform` (e.g., `fall_slope` is evaluated as `abs(fall_slope)`)
- `auc_feature_raw` (ROC-AUC using transformed feature as score)
- `auc_feature_oriented` + `auc_direction` (same discrimination with the best direction handling)
- `mutual_info_bits` (discretized mutual information estimate)
- `logreg` block with univariate logistic regression coefficients and diagnostics (`auc_logreg`, `mcfadden_r2`, `status`)
- bootstrap AUC interval fields (`auc_boot_mean`, `auc_boot_ci_lo`, `auc_boot_ci_hi`)
- `feature_correlation` block with pairwise Pearson/Spearman matrices for naked features

`debug_report` now also includes versioned score:
- `spike_score_v1` from weighted normalized components
  - `rise_slope_z` (0.36), `gradient_max_z` (0.18), `area_z` (0.10)

### Interactive correlation hover maps (Pearson + Spearman)
For interactive inspection of feature-feature correlation matrices from `corr.json`:

```bash
python correlation_hover_gui.py --corr-json corr.json --layout two-panels
```

Behavior:
- two panels: Pearson a Spearman side by side
- axis ticks show metric/feature names on both axes for direct pair comparison
- hover mouse over any matrix cell to see a tooltip with:
  - matrix type
  - row/column feature name + feature `AUC_oriented` (if present in `corr.json`)
  - correlation coefficient and correlation-strength category
- tooltip color if strength-coded by `|corr|`:
  - 0.0-0.2 orthogonal (green)
  - 0.2-0.4 weak (blue)
  - 0.4-0.6 medium (orange)
  - 0.6-0.8 strong (light purple)
  - 0.8-1.0 redundant (red)

Optional:
- `--layout single --single-matrix pearson` opens one panel
- in single mode press `t` to toggle Pearson/Spearman
- right mouse click on a cell toggles a persistent pinned note (click same cell again to remove)
- each right-click selection is also printed to console
- `--zero-emphasis` switches coloring to `1-|corr|` so near-zero correlations are visually strongest
- `--annotate-diag` draws `1.00` on diagonal (small matrices)

### Interactive metric threshold viewer (no/maybe/ok)
Inspect one metric directly on raw spectra for coordinates present in `labels.csv`

```bash
python metric_threshold_viewer.py --input map.wdf --report path/to/debug_report.json --labels-csv labels.csv --metric spike_score_v1 --corr-json corr.json
```

What it does:
- loads spectra from `.wdf`/`.npz` for all `(y,x)` present in `labels.csv` (including spectra with zero labeled candidates)
- classifies each candidate peak by two thresholds of selected metric:
  - `< low` = `no-muon` (blue)
  - `low..high` = `maybe-muon` (orange)
  - `>= high` = `ok-muon` (red)
- prints thresholds to console
- overlays colored peak lines + per-peak metric value labels
- histogram is split by labels (`muon` vs `non-muon`) to show overlap
- Right/Left arrows (or `n`/`p`) browse spectra

Thresholds:
- optional manual: `--threshold-low ... --threshold-high ...`
- if omitted, thresholds are auto-estimated from metric distribution using 1D k-means (`k=3`)

### Interactive spike score tuner GUI
Tune a weighted test score against labels interactively:
```bash
python score_tuner_gui.py --report path/to/debug_report.json --labels-csv labels.csv --features rise_slope_z,fall_slope_z,gradient_max_z,prominence_local_z,area_z
```
The GUI shows:
- ROC curve and weighted-score AUC
- histogram split by labels (`is_muon`)
- live Pearson/Spearman/logreg-AUC readout
- per-feature sliders + numeric boxes (0..slider-max) with normalized internal weights
