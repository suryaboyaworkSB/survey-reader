#!/usr/bin/env python3
"""
Resumable, per-survey chunked batch runner.

Same result as batch_scan.py, but processes surveys ONE AT A TIME and writes a
checkpoint after each. This lets a single big itinerary (30-40 surveys) finish
across MANY short runs — every run picks up exactly where the last stopped.
Designed for time-limited environments (e.g. a 45s call budget).

Per file it keeps  <out>/.progress/<name>.json  = {"surveys":[[label,offset],...],
"done":{label:result,...}}. When every survey in a file is done it writes
<name>_results.xlsx + <name>_results_dataentry.csv and deletes the checkpoint.

Usage:
    python3 chunk_scan.py INPUT_FOLDER [OUTPUT_FOLDER] [--budget SECONDS]
"""
import sys, os, glob, json, time, traceback

BLANK      = "BRNB422005F2960_004268.pdf"
CONFIG     = "form_config.json"
SKIP_PAGES = 1
SCAN_ROT   = 90
DPI        = 300
EXCLUDE    = ["ripped test", "blank", "brnb422005", "headway", "paddle"]
HERE       = os.path.dirname(os.path.abspath(__file__))


def main():
    args = [a for a in sys.argv[1:]]
    budget = 35.0
    if "--budget" in args:
        i = args.index("--budget"); budget = float(args[i + 1]); del args[i:i + 2]
    if not args:
        print("usage: python3 chunk_scan.py INPUT_FOLDER [OUTPUT_FOLDER] [--budget S]")
        sys.exit(1)
    inp = args[0]
    out = args[1] if len(args) > 1 else os.path.join(inp, "Results")
    prog_dir = os.path.join(out, ".progress")
    os.makedirs(prog_dir, exist_ok=True)
    os.chdir(HERE)

    import survey_ocr as so
    import grid_pipeline as gp
    import make_dataentry_csv as mk
    from pathlib import Path

    config = json.load(open(CONFIG))
    fields = config["fields"]
    ppf    = config.get("pages_per_form", 2)
    gcfg   = config.get("groups", {})
    ov = gp.load_overrides("calibration_overrides.json")
    blank_cal, blank_meta = gp.calibrate_blank(BLANK, fields,
                                               scan_rotation=SCAN_ROT, overrides=ov)

    pdfs = [p for p in sorted(glob.glob(os.path.join(inp, "*.pdf")))
            if not any(x in os.path.basename(p).lower() for x in EXCLUDE)]

    t0 = time.time()
    processed_here = 0
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0].replace(" ", "_")
        xlsx = os.path.join(out, f"{name}_results.xlsx")
        csvf = os.path.join(out, f"{name}_results_dataentry.csv")
        if os.path.isfile(xlsx):
            continue
        pj = os.path.join(prog_dir, f"{name}.json")

        # Load or build the survey list (cached so serials aren't re-read).
        if os.path.isfile(pj):
            ck = json.load(open(pj))
        else:
            surveys = [[lab, off] for (lab, _p, off) in
                       so.iter_surveys([Path(pdf)], ppf, True, DPI, SCAN_ROT,
                                       skip_pages=SKIP_PAGES)]
            ck = {"surveys": surveys, "done": {}}
            json.dump(ck, open(pj, "w"))

        surveys = ck["surveys"]; done = ck["done"]
        # 0-survey files (e.g. "NO RETURNS" single page): finish immediately.
        if not surveys:
            _finish(so, mk, fields, [], xlsx, csvf, pj);
            print(f"== {name}: 0 surveys (empty output)"); continue

        remaining = [(lab, off) for lab, off in surveys if lab not in done]
        if not remaining:
            _finish(so, mk, fields, [done[l] for l, _ in surveys], xlsx, csvf, pj)
            print(f"== {name}: complete ({len(surveys)} surveys)"); continue

        print(f"-> {name}: {len(done)}/{len(surveys)} done, processing…", flush=True)
        for lab, off in remaining:
            if time.time() - t0 > budget:
                print(f"   [budget hit] {name}: {len(done)}/{len(surveys)}")
                json.dump(ck, open(pj, "w")); _report(prog_dir, out); return
            try:
                res = so.process_one_survey(lab, pdf, off, [], fields, None, DPI,
                                            scan_rotation=SCAN_ROT,
                                            group_config=gcfg,
                                            blank_cal=blank_cal, blank_meta=blank_meta)
            except Exception:
                print(f"   !! {name} survey {lab} FAILED"); traceback.print_exc()
                res = {"Survey": lab}
            done[lab] = res
            processed_here += 1
            json.dump(ck, open(pj, "w"))
        # all surveys done for this file
        _finish(so, mk, fields, [done[l] for l, _ in surveys], xlsx, csvf, pj)
        print(f"== {name}: complete ({len(surveys)} surveys)")

    print(f"\nProcessed {processed_here} survey(s) this run.")
    _report(prog_dir, out)


def _finish(so, mk, fields, results, xlsx, csvf, pj):
    if results:
        so.save_transposed_excel(results, fields, xlsx)
        _old = sys.argv; sys.argv = ["make_dataentry_csv.py", xlsx, csvf]
        try: mk.main()
        except Exception: traceback.print_exc()
        finally: sys.argv = _old
    else:
        # write a tiny placeholder so the file is marked done and skipped
        from openpyxl import Workbook
        wb = Workbook(); wb.active["A1"] = "No returns / 0 surveys"; wb.save(xlsx)
    if os.path.isfile(pj):
        try: os.remove(pj)
        except OSError: pass   # mounted folders may block delete; harmless


def _report(prog_dir, out):
    n_done = len(glob.glob(os.path.join(out, "*_results.xlsx")))
    pend = []
    for pj in sorted(glob.glob(os.path.join(prog_dir, "*.json"))):
        nm = os.path.basename(pj)[:-5]
        if os.path.isfile(os.path.join(out, f"{nm}_results.xlsx")):
            continue   # done; leftover checkpoint the mount wouldn't let us delete
        ck = json.load(open(pj))
        pend.append(f"{nm} {len(ck['done'])}/{len(ck['surveys'])}")
    print(f"[STATUS] files complete: {n_done}")
    if pend: print("[STATUS] in progress: " + "; ".join(pend))


if __name__ == "__main__":
    main()
