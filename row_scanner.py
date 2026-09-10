"""
row_scanner.py
==============
For each checkbox group, determine which option is selected by:
1. Finding the correct y-band for each row of options (scan vertically near config y)
2. Scanning horizontally to find ink-density peaks (= bubble positions)
3. Matching peaks to option names by x-position order
4. The peak with highest ink density = selected option

This completely replaces Hough-based detection.
"""
import json, sys, pathlib
import numpy as np
import cv2
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from survey_ocr import pdf_page_to_image, estimate_page_correction

FILL_RADIUS = 11   # px — radius used to measure ink density
SEARCH_Y    = 25   # ±px to search for actual bubble row
SEARCH_X    = 30   # ±px to locate x peak within each column

def ink_at(img, cx, cy, r=FILL_RADIUS):
    """Interior ink density in [0,1]."""
    h, w = img.shape[:2]
    x1, y1 = max(0, cx-r), max(0, cy-r)
    x2, y2 = min(w, cx+r), min(h, cy+r)
    patch = img[y1:y2, x1:x2].astype(np.float32)
    lcx, lcy = cx-x1, cy-y1
    Y, X = np.ogrid[-lcy:patch.shape[0]-lcy, -lcx:patch.shape[1]-lcx]
    mask = X*X + Y*Y < (r-3)**2
    if mask.sum() < 4:
        return 0.0
    return float((255 - patch[mask]).mean() / 255.0)


def find_best_y(img, cx, config_cy, search=SEARCH_Y, r=FILL_RADIUS):
    """Find y in [config_cy-search, config_cy+search] that maximises ink at cx."""
    ih = img.shape[0]
    best_y, best_score = config_cy, ink_at(img, cx, config_cy, r)
    for dy in range(-search, search+1, 2):
        cy = config_cy + dy
        if cy < r or cy >= ih - r:
            continue
        s = ink_at(img, cx, cy, r)
        if s > best_score:
            best_score = s
            best_y = cy
    return best_y, best_score


def find_best_x(img, config_cx, cy, search=SEARCH_X, r=FILL_RADIUS):
    """Find x in [config_cx-search, config_cx+search] that maximises ink at cy."""
    iw = img.shape[1]
    best_x, best_score = config_cx, ink_at(img, config_cx, cy, r)
    for dx in range(-search, search+1, 2):
        cx = config_cx + dx
        if cx < r or cx >= iw - r:
            continue
        s = ink_at(img, cx, cy, r)
        if s > best_score:
            best_score = s
            best_x = cx
    return best_x, best_score


def detect_page(scan, fields, corr, min_floor=0.22, verbose=False):
    """
    Detect checkbox answers for one page.
    Returns dict: group_prefix -> selected_option_name (or None)
    """
    ih, iw = scan.shape[:2]
    ax, bx, ay, by_ = corr

    def aff(ex, ey):
        cx = max(FILL_RADIUS, min(iw-FILL_RADIUS, int(ax*ex + bx)))
        cy = max(FILL_RADIUS, min(ih-FILL_RADIUS, int(ay*ey + by_)))
        return cx, cy

    # Group fields by prefix
    groups = {}
    for f in fields:
        if f.get("type") != "checkbox":
            continue
        name = f["name"]
        prefix = name.split(": ", 1)[0] if ": " in name else name
        groups.setdefault(prefix, []).append(f)

    results = {}

    for prefix, opts in groups.items():
        if len(opts) < 2:
            continue

        # ── Cluster options by their config y (= same row) ───────────────
        y_vals = {}   # y_key → list of fields
        for f in opts:
            ey_raw = int(f["y"] * ih)
            yk = round(ey_raw / 20) * 20   # bucket to 20px rows
            y_vals.setdefault(yk, []).append(f)

        scores = {}

        for yk, row_fields in y_vals.items():
            # For each field in this row, compute affine-corrected expected center
            # then search locally for the true bubble position
            for f in row_fields:
                ex = int(f["x"] * iw)
                ey = int(f["y"] * ih)
                ecx, ecy = aff(ex, ey)

                # Find the best y for this column (±SEARCH_Y)
                best_cy, _ = find_best_y(scan, ecx, ecy, SEARCH_Y)
                # Find the best x for this row (±SEARCH_X)
                best_cx, best_score = find_best_x(scan, ecx, best_cy, SEARCH_X)

                scores[f["name"]] = best_score

                if verbose:
                    opt = f["name"].split(": ", 1)[-1]
                    print(f"  {opt:30s} config=({ecx},{ecy}) "
                          f"best=({best_cx},{best_cy}) score={best_score:.3f}")

        # Winner-takes-all
        if not scores:
            results[prefix] = None
            continue

        best_name = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]

        if best_score < min_floor:
            results[prefix] = None
        else:
            results[prefix] = best_name.split(": ", 1)[-1] if ": " in best_name else best_name

        if verbose:
            print(f"  → {prefix}: {results[prefix]} ({best_score:.3f})")

    return results


def run_survey(pdf_path, config, survey_idx=0, verbose=False):
    fields_all = config["fields"]
    all_results = {}

    for page_idx in range(2):
        scan = pdf_page_to_image(pdf_path, page_num=survey_idx*2 + page_idx,
                                  dpi=300, scan_rotation=90)
        ih, iw = scan.shape[:2]
        page_fields = [f for f in fields_all if f.get("page", 0) == page_idx]
        corr = estimate_page_correction(scan, page_fields)

        if verbose:
            print(f"\n=== Page {page_idx+1} affine: {corr} ===")

        res = detect_page(scan, page_fields, corr, verbose=verbose)
        all_results.update(res)

    return all_results


if __name__ == "__main__":
    with open("form_config.json") as f:
        config = json.load(f)

    pdf_path = str(sorted(pathlib.Path("scans").glob("*.pdf"))[0])
    print(f"Scan: {pdf_path}")

    results = run_survey(pdf_path, config, survey_idx=0, verbose=True)
    print("\n\n=== RESULTS ===")
    for prefix, ans in sorted(results.items()):
        print(f"  {prefix}: {ans}")
