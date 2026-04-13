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

This mode extracts/despikes only specified coordinates and can render a compact (near-square) viewer layout from selected points.
