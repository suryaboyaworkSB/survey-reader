#!/usr/bin/env python3
"""
Batch-scan a folder of survey PDFs. Each PDF (one itinerary/mailback run) is
processed on its own and produces, under an output folder:

    <name>_results.xlsx           — answers (Survey Results + Needs Review tabs)
    <name>_results_dataentry.csv  — wide data-entry sheet (questions x surveys)
    overlays_<name>/              — per-page alignment images (only with --overlays)

Runs in ONE process (calibrates the blank once) and is RESUMABLE — re-running
skips files whose _results.xlsx already exists, so a big folder can be finished
over several runs.

Usage:
    python3 batch_scan.py  INPUT_FOLDER  [OUTPUT_FOLDER]  [--overlays]
"""
import sys, os, glob, json, traceback

BLANK      = "BRNB422005F2960_004268.pdf"
CONFIG     = "form_config.json"
SKIP_PAGES = 1
SCAN_ROT   = 90
EXCLUDE    = ["ripped test", "blank", "brnb422005", "headway", "paddle"]
HERE       = os.path.dirname(os.path.abspath(__file__))


def main():
    argv = [a for a in sys.argv[1:] if a != "--overlays"]
    overlays = "--overlays" in sys.argv
    if not argv:
        print("usage: python3 batch_scan.py INPUT_FOLDER [OUTPUT_FOLDER] [--overlays]")
        sys.exit(1)
    inp = argv[0]
    out = argv[1] if len(argv) > 1 else os.path.join(inp, "Results")
    os.makedirs(out, exist_ok=True)
    os.chdir(HERE)

    import survey_ocr as so
    import grid_pipeline as gp
    import make_dataentry_csv as mk

    config = json.load(open(CONFIG))
    fields = config["fields"]
    ov = gp.load_overrides("calibration_overrides.json")
    print(f"Calibrating from {BLANK} ...")
    cal = gp.calibrate_blank(BLANK, fields, scan_rotation=SCAN_ROT, overrides=ov)

    pdfs = [p for p in sorted(glob.glob(os.path.join(inp, "*.pdf")))
            if not any(x in os.path.basename(p).lower() for x in EXCLUDE)]
    print(f"{len(pdfs)} survey batch file(s) in folder.\n")

    done = todo = failed = 0
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0].replace(" ", "_")
        xlsx = os.path.join(out, f"{name}_results.xlsx")
        csvf = os.path.join(out, f"{name}_results_dataentry.csv")
        if os.path.isfile(xlsx):
            done += 1; continue
        todo += 1
        print(f"-> {os.path.basename(pdf)}", flush=True)
        try:
            so.process_forms(None, pdf, config, xlsx, dpi=300,
                             scan_rotation=SCAN_ROT, skip_pages=SKIP_PAGES,
                             annotate_dir=(os.path.join(out, f"overlays_{name}")
                                           if overlays else None),
                             precomputed_cal=cal)
            mk_argv = ["make_dataentry_csv.py", xlsx, csvf]
            _old = sys.argv; sys.argv = mk_argv
            try: mk.main()
            finally: sys.argv = _old
            done += 1
        except Exception:
            failed += 1
            print(f"   !! FAILED: {os.path.basename(pdf)}")
            traceback.print_exc()
    print(f"\n{done}/{len(pdfs)} complete  ({todo} this run, {failed} failed) -> {out}")


if __name__ == "__main__":
    main()
