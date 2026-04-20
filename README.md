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

You can switch feature extraction signal source for debug report via:
- `debug_feature_signal_source`: `"gradient"` (default) or  `"raw"`
- `merge_duplicate_segments`: `true/false` (applies to both debug report and viewer spike overlay merge)

When set to `raw`,slope/asymmetry/plateau/`gradient_max*`-named fields are computed from the raw spectrum in the candidate interval; `feature_source` in JSON reflects the selected source.

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

`debug_stats.py` now reports for each feature:
- Pearson and Spearman correlation to `is_muon`
- `transform` (e.g., `fall_slope` is evaluated as `abs(fall_slope)`)
- `auc_feature_raw` (ROC-AUC using transformed feature as score)
- `auc_feature_oriented` + `auc_direction` (same discrimination with the best direction handling)
- `mutual_info_bits` (discretized mutual information estimate)
- `logreg` block with univariate logistic regression coefficients and diagnostics (`auc_logreg`, `mcfadden_r2`, `status`)