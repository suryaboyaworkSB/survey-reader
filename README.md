# Survey Reader

Reads scanned paper survey forms (bubble/checkbox answers) and exports results to Excel, using image alignment against a blank template rather than a trained model.

## Key files

- `survey_ocr.py` — CLI entry point / main pipeline
- `grid_pipeline.py` — the accurate reading pipeline (ORB homography + template subtraction); the most actively developed file
- `grid_detect.py` — bubble detection + ink scoring
- `chunk_scan.py` — resumable per-survey batch runner, best for large itineraries (can be interrupted and resumed)
- `batch_scan.py` — simpler whole-folder batch runner, one-shot (fine to run locally with no time limit)
- `make_dataentry_csv.py` — builds a data-entry CSV from a results workbook

## Required data files

These must sit alongside the scripts in the repo root:

- `form_config.json` — field/bubble layout configuration
- `calibration_overrides.json` — per-form calibration adjustments
- `BRNB422005F2960_004268.pdf` — the blank form template used as the alignment reference

## Usage

Process a folder of scanned forms:

```bash
# Resumable — best for large batches, can be safely interrupted and re-run
python3 chunk_scan.py INPUT_FOLDER

# One-shot — simpler, fine for smaller batches with no time limit
python3 batch_scan.py INPUT_FOLDER
```

## Setup

```bash
pip install -r requirements.txt
```

(See individual scripts for additional dependencies such as `opencv-python`, `pymupdf`, and `openpyxl`.)

## Notes

Scanned inputs, debug/calibration images, and generated result spreadsheets are excluded from version control via `.gitignore` — only source scripts and the three required config/template files above are tracked. Several old `form_config*.json` backup variants (`.bak`, `.bak2`, `_original`, `_calibrated`, etc.) are also still tracked from earlier iterations and can likely be cleaned up.
