#!/usr/bin/env python3
"""
calibrate_config.py  —  Auto-calibrate form_config.json from the blank template PDF
======================================================================================

The blank template has clean, unfilled checkbox circles.  This script renders
each template page, runs Hough Circle detection to find every printed circle,
then matches each config field to its nearest detected circle.  Fields whose
config position lands on a printed element (not on an actual checkbox) get their
x/y corrected.

Usage
-----
    python3 calibrate_config.py \
        --template NJTransitSurvey2026_BLANK.pdf \
        --config   form_config.json \
        --output   form_config_calibrated.json \
        --dpi      300

The script writes a new calibrated JSON and also a debug PNG for each page
showing: red dot = original config position, green circle = circle found in
template, cyan dot = corrected position (only when moved > 5px).

Requirements: same as survey_ocr.py (pymupdf, opencv-python-headless, numpy)
"""

import argparse
import json
import sys
import os
import math
import copy
import numpy as np
import cv2


# ─── PDF rendering ────────────────────────────────────────────────────────────

def render_page(pdf_path: str, page_num: int, dpi: int = 300) -> np.ndarray:
    """Render a template page to a grayscale numpy array (no rotation)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        doc.close()
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) not installed. Run: pip install pymupdf")


# ─── Circle detection ─────────────────────────────────────────────────────────

def find_all_circles(img: np.ndarray, er: int = 22) -> np.ndarray:
    """
    Find all circle candidates on a template page.
    Returns array of shape (N, 3): [[cx, cy, r], ...]
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=er * 2,
        param1=50,          # slightly more permissive on template (clean image)
        param2=12,
        minRadius=max(4, er - 12),
        maxRadius=er + 14,
    )
    if circles is None:
        return np.empty((0, 3), dtype=float)
    return np.round(circles[0]).astype(float)


def match_field_to_circle(circles: np.ndarray,
                          ex: float, ey: float,
                          search_px: int = 50) -> tuple | None:
    """
    Return (cx, cy, r) of the closest circle to expected position (ex, ey)
    within search_px, or None if no circle is close enough.
    """
    if len(circles) == 0:
        return None
    dists = np.sqrt((circles[:, 0] - ex) ** 2 + (circles[:, 1] - ey) ** 2)
    idx = int(np.argmin(dists))
    if dists[idx] <= search_px:
        return tuple(circles[idx])
    return None


# ─── Calibration ──────────────────────────────────────────────────────────────

def calibrate(template_path: str,
              config: dict,
              dpi: int = 300,
              search_px: int = 50,
              move_threshold: float = 3.0,
              debug_dir: str | None = None) -> dict:
    """
    Return a deep copy of config with corrected x/y for checkbox fields whose
    nearest detected circle in the template is more than `move_threshold` pixels
    away from the raw config position.

    Parameters
    ----------
    move_threshold : only update a field's position if the closest template
                     circle is more than this many pixels from the raw position
                     (avoids micro-adjustments from Hough subpixel jitter).
    """
    calibrated = copy.deepcopy(config)
    fields = calibrated["fields"]

    # Determine how many pages are needed
    page_indices = sorted({f.get("page", 0) for f in fields if f.get("type") == "checkbox"})

    stats = {"total": 0, "corrected": 0, "not_found": 0}

    for page_idx in page_indices:
        print(f"\n[PAGE {page_idx}] Rendering template page {page_idx} at {dpi} DPI …")
        try:
            img = render_page(template_path, page_idx, dpi)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        H, W = img.shape[:2]
        print(f"  Template size: {W}×{H} px")

        circles = find_all_circles(img, er=22)
        print(f"  Hough found {len(circles)} circle candidates")

        # Optional debug image
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for cx, cy, r in circles:
                cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 200, 0), 1)

        page_fields = [f for f in fields
                       if f.get("page", 0) == page_idx
                       and f.get("type") == "checkbox"]

        for f in page_fields:
            stats["total"] += 1
            # Raw expected center from config
            ex = (f["x"] + f["w"] / 2) * W
            ey = (f["y"] + f["h"] / 2) * H

            match = match_field_to_circle(circles, ex, ey, search_px=search_px)

            if match is None:
                print(f"  ⚠  {f['name'][:50]:50s}  NO circle within {search_px}px of ({ex:.0f},{ey:.0f})")
                stats["not_found"] += 1
                if debug_dir:
                    cv2.circle(vis, (int(ex), int(ey)), 5, (0, 0, 255), -1)  # red = not found
                continue

            cx, cy, r = match
            dist = math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)

            if dist > move_threshold:
                # Compute new fractions from circle center
                # (keep w and h unchanged — they define the expected radius)
                new_x = cx / W - f["w"] / 2
                new_y = cy / H - f["h"] / 2
                new_x = round(max(0.0, min(1.0, new_x)), 4)
                new_y = round(max(0.0, min(1.0, new_y)), 4)

                print(f"  ✎  {f['name'][:50]:50s}  "
                      f"({f['x']:.4f},{f['y']:.4f}) → ({new_x:.4f},{new_y:.4f})  "
                      f"[Δ={dist:.1f}px]")
                f["x"] = new_x
                f["y"] = new_y
                stats["corrected"] += 1

                if debug_dir:
                    # Old position: red
                    cv2.circle(vis, (int(ex), int(ey)), 4, (0, 0, 255), -1)
                    # New circle: green
                    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                    # Cyan dot at corrected center
                    ncx = int((new_x + f["w"] / 2) * W)
                    ncy = int((new_y + f["h"] / 2) * H)
                    cv2.circle(vis, (ncx, ncy), 4, (255, 255, 0), -1)
            else:
                if debug_dir:
                    # Unchanged: small white dot
                    cv2.circle(vis, (int(ex), int(ey)), 3, (255, 255, 255), -1)

        if debug_dir:
            out = os.path.join(debug_dir, f"calibration_page{page_idx}.png")
            cv2.imwrite(out, vis)
            print(f"  [DEBUG] Saved calibration overlay: {out}")

    print(f"\nCalibration summary")
    print(f"  Fields checked : {stats['total']}")
    print(f"  Positions moved: {stats['corrected']}")
    print(f"  Not found      : {stats['not_found']}")
    return calibrated


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-calibrate form_config.json bubble positions from the blank template PDF."
    )
    parser.add_argument("--template", required=True,
                        help="Path to the blank form template PDF")
    parser.add_argument("--config",   default="form_config.json",
                        help="Input field config JSON (default: form_config.json)")
    parser.add_argument("--output",   default="form_config_calibrated.json",
                        help="Output calibrated JSON (default: form_config_calibrated.json)")
    parser.add_argument("--dpi",      type=int, default=300,
                        help="Render DPI (default: 300 — must match survey_ocr.py DPI)")
    parser.add_argument("--search-px", type=int, default=50, dest="search_px",
                        help="Max pixel distance to search for a matching circle (default: 50)")
    parser.add_argument("--move-threshold", type=float, default=3.0, dest="move_threshold",
                        help="Only update position if circle is more than N px from config (default: 3)")
    parser.add_argument("--debug-dir", default=None, dest="debug_dir",
                        help="Folder to save calibration overlay images")
    args = parser.parse_args()

    if not os.path.isfile(args.template):
        print(f"ERROR: Template not found: {args.template}")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    print(f"Template : {args.template}")
    print(f"Config   : {args.config}  ({len(config['fields'])} fields)")
    print(f"DPI      : {args.dpi}")
    print(f"Search   : ±{args.search_px}px")

    calibrated = calibrate(
        template_path=args.template,
        config=config,
        dpi=args.dpi,
        search_px=args.search_px,
        move_threshold=args.move_threshold,
        debug_dir=args.debug_dir,
    )

    with open(args.output, "w") as f:
        json.dump(calibrated, f, indent=2)
    print(f"\nCalibrated config written → {args.output}")
    print("Review the changes and rename to form_config.json when satisfied.")
    if args.debug_dir:
        print(f"Overlay images → {args.debug_dir}/calibration_page*.png")


if __name__ == "__main__":
    main()
