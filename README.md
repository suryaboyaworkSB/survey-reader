# Survey Form OCR Processor

Reads **handwritten text** and **checkboxes** from scanned PDF survey forms and exports all answers to an **Excel spreadsheet** — no AI training required.

---

## How it works

1. You provide the **blank original form** as the reference template.
2. The script aligns each scanned filled-in form to that template.
3. It crops out each answer field, runs OCR on text fields, and detects tick marks on checkbox fields.
4. All answers are written to `survey_results.xlsx`, one row per form.

---

## Setup (one-time)

You need Python 3.8+ installed.  Then run:

```bash
pip install pymupdf opencv-python-headless pillow easyocr numpy openpyxl
```

> The first time you run the script, EasyOCR will download a small English language model (~50 MB). After that it works offline.

---

## Step 1 — Configure your form fields

Open **`form_config.json`** and define where each answer field appears on your form.

Coordinates use **fractions of the page size** (0.0–1.0), so they work regardless of scan resolution.

### How to find coordinates

The easiest way:

1. Open your blank PDF in a viewer and note roughly where each field is.
2. Estimate: if a field starts halfway across and a quarter down the page, that's `x: 0.5, y: 0.25`.
3. Run the script once on a test scan and check the output — adjust coordinates if the wrong area was captured.

### Field types

| `type` | Use for |
|--------|---------|
| `"text"` | Any handwritten answer box |
| `"checkbox"` | A tick box / bubble (returns **Yes** or **No**) |

### Example entry

```json
{
  "name": "Q1 - Satisfaction rating",
  "type": "text",
  "page": 0,
  "x": 0.10,
  "y": 0.20,
  "w": 0.60,
  "h": 0.05
}
```

---

## Step 2 — Organise your files

```
my_project/
├── blank_form.pdf          ← your original unwritten form
├── scans/
│   ├── form_001.pdf        ← each completed scan is one PDF
│   ├── form_002.pdf
│   └── ...
├── form_config.json        ← field definitions
└── survey_ocr.py           ← the script
```

---

## Step 3 — Run the script

```bash
python survey_ocr.py \
  --template blank_form.pdf \
  --scans    scans/ \
  --config   form_config.json \
  --output   survey_results.xlsx
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--template` | *(required)* | Path to blank form PDF |
| `--scans` | *(required)* | Folder of completed scan PDFs |
| `--config` | `form_config.json` | Field layout config |
| `--output` | `survey_results.xlsx` | Output spreadsheet |
| `--dpi` | `200` | Rendering quality — raise to `300` for better accuracy on small handwriting |

---

## Tips for best results

- **Scan quality matters** — 200 DPI minimum; 300 DPI recommended for handwriting.
- **Consistent scanning** — try to scan all forms at the same orientation and size.
- **Checkbox sensitivity** — if checkboxes are being misread, adjust the `threshold` value inside `read_checkbox_field()` in the script (lower = more sensitive).
- **Multi-page forms** — set `"pages_per_form"` in the config and specify `"page": 1` (etc.) for fields on later pages.

---

## Output format

`survey_results.xlsx` will have:
- **Row 1**: Header with field names
- **Row 2+**: One row per scanned form
- **Column A**: The filename of the scanned PDF

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OCR reads wrong text | Re-check x/y/w/h coordinates; increase `--dpi` |
| All checkboxes show "No" | Lower the threshold in `read_checkbox_field()` |
| "No PDF files found" | Make sure your scan files end in `.pdf` |
| Very slow processing | Normal for first run (model download); subsequent runs are faster |
