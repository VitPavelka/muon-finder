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

### Quick helper: plot one debug parameter vs spectrum index
```bash
python debug_explorer.py --report path/to/debug_report.json --y 62 --x 15 --param muon_score --x-axis candidate_index
```

Across many spectra (one reduced value per spectrum):
```bash
python debug_explorer.py -- report path/to/debug_report.json --mode spectra_overview --param muon_score --reduce max --max-spectra 500
```