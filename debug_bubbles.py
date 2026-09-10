"""
debug_bubbles.py
================
Generates annotated PNG images showing exactly where the OCR code looks
for each checkbox bubble on each survey page, overlaid on the actual scan.

For every bubble the code checks it draws:
  - GREEN circle  = detected circle location (where Hough found the ring)
  - RED    circle = expected location from form_config.json (before Hough correction)
  - Score value next to each circle (0.00–1.00)
  - CYAN label on the option that was selected (highest score & above floor)
  - YELLOW label on the question group name

Run:
    cd "WORK PROJECTS/Survey Reader"
    python3 debug_bubbles.py --scans scans/ --config form_config.json --survey 1
       --survey N  : which survey to debug (1-based, default=1)
       --page N    : which page of that survey (1-based, default=all)
       --group STR : only show one group (e.g. "Q11 - Frequency")
       --out DIR   : output folder (default: debug_out/)

Output: debug_out/survey_1_page_1.png, survey_1_page_2.png, ...
"""

import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np
try:
    import fitz  # noqa — pymupdf optional
except ImportError:
    pass  # using pypdfium2 instead

# ── Reuse core helpers from survey_ocr ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from survey_ocr import (
    pdf_page_to_image,
    pdf_page_count,
    estimate_page_correction,
    _locate_bubble,
    _interior_ink_density,
)

# ── Ground-truth table (what the correct answer should be) ───────────────────
# Fill this in to see CORRECT vs DETECTED labels side-by-side.
# Format: {survey_label: {group_name: correct_option_name}}
GROUND_TRUTH = {
    "0001": {
        "Q2 - Board Time":           "AM",
        "Q3 - Came From":            "School",
        "Q5 - To Bus":               "Bike",
        "Q8 - After Bus":            "Another Bus",
        "Q10 - Going To":            "School",
        "Q11 - Frequency":           "1-2 days per week",
        "Q12 - Ticket":              "Bus Monthly Pass",
        "Q13 - Pay":                 "Credit Card",
        "Q14 - Buy Ticket":          "NJ Transit Mobile App",
        "Q15 - Return":              "Same Bus Route 4pm to 6:59pm",
        "Q16 - Trip Purpose":        "Work",
        "Q17 - Alt Mode":            "Carpooled/Dropped Off",
        "Q18 - Gender":              "Non-Binary/Gender Fluid",
        "Q20 - Traveling With Child":"No",
        "Q21 - English":             "Well",
        "Q22 - Other Language":      "No",
        "Q23 - Hispanic/Latino":     "No",
        "Q24 - Race":                "Asian/Pacific Islander",
        "Q27 - Disability":          "Yes - Vision (blind or low vision)",
        "Q28 - Income":              "$25,000-$34,999",
        "Q29 - Most Important Bus Improvement": "",
    },
    "0002": {
        "Q2 - Board Time":           "AM",
        "Q3 - Came From":            "Work",
        "Q5 - To Bus":               "Walked Only",
        "Q8 - After Bus":            "",
        "Q10 - Going To":            "Other (Please specify)",
        "Q11 - Frequency":           "less than once a year",
        "Q12 - Ticket":              "Bus Monthly Pass",
        "Q13 - Pay":                 "Paypal",
        "Q14 - Buy Ticket":          "Other (Please specify)",
        "Q15 - Return":              "Other (Please specify)",
        "Q16 - Trip Purpose":        "Personal Business",
        "Q17 - Alt Mode":            "Would Not Make Trip",
        "Q18 - Gender":              "Non-Binary/Gender Fluid",
        "Q20 - Traveling With Child":"No",
        "Q21 - English":             "Not well",
        "Q22 - Other Language":      "Yes",
        "Q23 - Hispanic/Latino":     "Yes",
        "Q24 - Race":                "White/Caucasian",
        "Q27 - Disability":          "No",
        "Q28 - Income":              "",
    },
    "0003": {
        "Q2 - Board Time":           "AM",
        "Q3 - Came From":            "Home",
        "Q5 - To Bus":               "Walked Only",
        "Q8 - After Bus":            "Light Rail (Please specify Boarding Station)",
        "Q10 - Going To":            "School",
        "Q11 - Frequency":           "less than once a year",
        "Q12 - Ticket":              "One-way/Cash Fare/Transfer",
        "Q13 - Pay":                 "Google Pay",
        "Q14 - Buy Ticket":          "Ticket Vending Machine",
        "Q15 - Return":              "Other (Please specify)",
        "Q16 - Trip Purpose":        "Work",
        "Q17 - Alt Mode":            "Other (Please specify)",
        "Q18 - Gender":              "Female/Woman",
        "Q20 - Traveling With Child":"Yes",
        "Q21 - English":             "Not well",
        "Q22 - Other Language":      "Yes",
        "Q23 - Hispanic/Latino":     "No",
        "Q24 - Race":                "Mixed Race",
        "Q27 - Disability":          "No",
        "Q28 - Income":              "Under $15,000",
    },
    "0004": {
        "Q2 - Board Time":           "AM",
        "Q3 - Came From":            "School",
        "Q5 - To Bus":               "Bike",
        "Q8 - After Bus":            "Another Bus (Please specify route number)",
        "Q10 - Going To":            "",
        "Q11 - Frequency":           "2 days per week",
        "Q12 - Ticket":              "Rail Monthly Pass",
        "Q13 - Pay":                 "",
        "Q14 - Buy Ticket":          "NJ TRANSIT Ticket agent",
        "Q15 - Return":              "Same Bus Route 7pm to 9:59pm",
        "Q16 - Trip Purpose":        "School",
        "Q17 - Alt Mode":            "Taxi",
        "Q18 - Gender":              "Male/Man",
        "Q20 - Traveling With Child":"No",
        "Q21 - English":             "Not well",
        "Q22 - Other Language":      "No",
        "Q23 - Hispanic/Latino":     "Yes",
        "Q24 - Race":                "Mixed Race",
        "Q27 - Disability":          "Yes - Hearing (deaf or difficulty hearing)",
        "Q28 - Income":              "$150,000-$199,999",
    },
    "0005": {
        "Q2 - Board Time":           "AM",
        "Q3 - Came From":            "School",
        "Q5 - To Bus":               "Another Bus (Please specify route number)",
        "Q8 - After Bus":            "",
        "Q10 - Going To":            "School",
        "Q11 - Frequency":           "1-3 days a month",
        "Q12 - Ticket":              "Rail Monthly Pass",
        "Q13 - Pay":                 "Cash",
        "Q14 - Buy Ticket":          "NJ TRANSIT Ticket agent",
        "Q15 - Return":              "Same Bus Route 4pm to 6:59pm",
        "Q16 - Trip Purpose":        "Work",
        "Q17 - Alt Mode":            "Bike",
        "Q18 - Gender":              "Non-Binary/Gender Fluid",
        "Q20 - Traveling With Child":"No",
        "Q21 - English":             "Not well",
        "Q22 - Other Language":      "No",
        "Q23 - Hispanic/Latino":     "Yes",
        "Q24 - Race":                "Asian/Pacific Islander",
        "Q27 - Disability":          "Yes - Vision (blind or low vision)",
        "Q28 - Income":              "",
    },
}

MIN_INK_FLOOR = 0.25   # must match survey_ocr.py

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_EXPECTED  = (0,   0,   200)   # red    = config expected position
C_DETECTED  = (0,   200, 0  )   # green  = Hough-found circle
C_SELECTED  = (255, 200, 0  )   # cyan   = winner (selected)
C_CORRECT   = (0,   255, 0  )   # bright green = matches ground truth
C_WRONG     = (0,   0,   255)   # bright red   = wrong answer
C_BLANK_GT  = (200, 200, 0  )   # teal = correct blank


def annotate_page(scan_img, fields_on_page, correction, survey_label,
                  filter_group=None):
    """Return an annotated copy of scan_img."""
    vis = cv2.cvtColor(scan_img, cv2.COLOR_GRAY2BGR)
    h, w = scan_img.shape[:2]

    # Group fields
    groups: dict[str, list] = {}
    for f in fields_on_page:
        g = f.get("group", f["name"])
        groups.setdefault(g, []).append(f)

    gt_survey = GROUND_TRUTH.get(survey_label, {})

    for grp_name, members in groups.items():
        if filter_group and filter_group.lower() not in grp_name.lower():
            continue

        scores = {}
        locs   = {}
        exp_locs = {}

        for f in members:
            # Expected position (from config, after affine)
            ex = int(f["x"] * w)
            ey = int(f["y"] * h)
            if correction:
                ax, bx, ay, by_ = correction
                ex = int(ax * ex + bx)
                ey = int(ay * ey + by_)
            exp_locs[f["name"]] = (ex, ey)

            # Hough-located position
            cx, cy, radius = _locate_bubble(scan_img, f, correction=correction)
            locs[f["name"]]   = (cx, cy, radius)

            # Ink score
            scores[f["name"]] = _interior_ink_density(scan_img, cx, cy, radius)

        # Determine winner
        best_name  = max(scores, key=scores.__getitem__) if scores else None
        best_score = scores.get(best_name, 0.0)
        selected   = best_name if best_score >= MIN_INK_FLOOR else None

        # Ground truth for this group
        gt_answer = gt_survey.get(grp_name, None)

        # Draw
        for f in members:
            name = f["name"]
            cx, cy, r   = locs[name]
            ex, ey      = exp_locs[name]
            score        = scores[name]
            is_selected  = (name == selected)

            # Expected pos (red dot)
            cv2.circle(vis, (ex, ey), 4, C_EXPECTED, -1)

            # Detected circle (green ring, thicker if selected)
            ring_color = C_SELECTED if is_selected else C_DETECTED
            thickness  = 3 if is_selected else 1
            cv2.circle(vis, (cx, cy), r, ring_color, thickness)

            # Score label
            label_x = cx + r + 3
            label_y = cy + 5
            cv2.putText(vis, f"{score:.2f}", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ring_color, 1)

            # Option name (small, truncated)
            short = name[:30]
            cv2.putText(vis, short, (label_x + 40, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # Group header with correct/wrong indicator
        if members:
            gx = int(members[0]["x"] * w) - 15
            gy = int(members[0]["y"] * h) - 18
            if correction:
                ax, bx, ay, by_ = correction
                gx = int(ax * gx + bx)
                gy = int(ay * gy + by_)

            # Decide colour based on ground truth match
            if gt_answer is None:
                hdr_color = (200, 200, 200)
            elif gt_answer == "":
                # expect blank
                hdr_color = C_CORRECT if selected is None else C_WRONG
            else:
                # partial match: check if gt substring in selected name
                if selected and gt_answer.lower() in selected.lower():
                    hdr_color = C_CORRECT
                else:
                    hdr_color = C_WRONG

            header = f"{grp_name} -> OCR:{selected or 'BLANK'} | GT:{gt_answer or 'BLANK'}"
            cv2.putText(vis, header, (max(0, gx), max(15, gy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hdr_color, 1)

    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans",   required=True)
    ap.add_argument("--config",  default="form_config.json")
    ap.add_argument("--survey",  type=int, default=0,
                    help="1-based survey index to debug (0=all)")
    ap.add_argument("--page",    type=int, default=0,
                    help="1-based page within survey (0=all)")
    ap.add_argument("--group",   default="",
                    help="substring of group name to highlight only that group")
    ap.add_argument("--out",     default="debug_out")
    ap.add_argument("--dpi",     type=int, default=300)
    ap.add_argument("--scan-rotation", type=int, default=90)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    pages_per_form = config.get("pages_per_form", 2)
    scan_rotation  = args.scan_rotation
    fields_all     = config.get("fields", [])

    scans_dir = Path(args.scans)
    pdfs = sorted(scans_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs found in {scans_dir}")
        sys.exit(1)

    for pdf_path in pdfs:
        total_pages   = pdf_page_count(str(pdf_path))
        total_surveys = total_pages // pages_per_form
        print(f"{pdf_path.name}: {total_pages} pages → {total_surveys} surveys")

        for survey_idx in range(total_surveys):
            s_num = survey_idx + 1
            if args.survey and s_num != args.survey:
                continue

            label = f"{(survey_idx + 1):04d}"

            for local_page in range(pages_per_form):
                p_num = local_page + 1
                if args.page and p_num != args.page:
                    continue

                abs_page = survey_idx * pages_per_form + local_page
                scan_img = pdf_page_to_image(str(pdf_path),
                                             page_num=abs_page,
                                             dpi=args.dpi,
                                             scan_rotation=scan_rotation)
                h, w = scan_img.shape[:2]

                # Fields for this page (form_config uses 0-indexed pages)
                page_fields = [f for f in fields_all
                                if f.get("page", 0) == local_page]

                # Affine correction
                try:
                    corr = estimate_page_correction(scan_img, page_fields)
                except Exception:
                    corr = (1.0, 0.0, 1.0, 0.0)

                print(f"  Survey {s_num} page {p_num}: affine correction = {corr}")

                vis = annotate_page(scan_img, page_fields, corr, label,
                                    filter_group=args.group or None)

                out_path = os.path.join(args.out,
                    f"survey_{s_num:02d}_page_{p_num}.png")
                cv2.imwrite(out_path, vis)
                print(f"  Saved → {out_path}")

    print(f"\nDone. Open the images in {args.out}/ to inspect bubble detection.")


if __name__ == "__main__":
    main()
