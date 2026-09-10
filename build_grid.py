"""
build_grid.py  v3
=================
Build a calibrated bubble grid using Hough + affine-corrected matching.

Algorithm
---------
1. Detect Hough circles on reference scan.
2. Estimate page affine (same as survey_ocr.py uses per scan).
3. Apply affine correction to each config field's expected position.
4. Match config fields → nearest unassigned Hough circle (1-to-1, within SEARCH_PX).
5. Store the MATCHED CIRCLE position (not config position) in grid.
   Fall back to affine-corrected config position if no match found.
6. Save as calibrated_grid.json with normalised (nx, ny) coords.

Usage:  python3 build_grid.py
"""
import json, cv2, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from survey_ocr import pdf_page_to_image, estimate_page_correction

SEARCH_PX = 40      # max distance to match hough → config field
DPI       = 300

# ── helpers ──────────────────────────────────────────────────────────────────

def detect_hough(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=20,
        param1=50, param2=14,   # lower param2 → more circles found
        minRadius=10, maxRadius=22,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.int32)
    c = np.round(circles[0]).astype(np.int32)
    return c

def ring_contrast(img, cx, cy, r):
    """Returns (ring_darkness, outer_lightness, inner_fill)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, cx-r-6), max(0, cy-r-6)
    x2, y2 = min(w, cx+r+6), min(h, cy+r+6)
    p = img[y1:y2, x1:x2].astype(np.float32)
    lcx, lcy = cx-x1, cy-y1
    Y, X = np.ogrid[-lcy:p.shape[0]-lcy, -lcx:p.shape[1]-lcx]
    d = np.sqrt(X*X + Y*Y)
    ring   = (d > r-4) & (d < r+4)
    inner  = d < r-5
    outer  = d > r+5
    if ring.sum() < 8:
        return 0., 1., 0.
    rd = (255 - p[ring].mean())  / 255.
    fi = (255 - p[inner].mean()) / 255. if inner.sum() > 4 else 0.
    oi = p[outer].mean() / 255.         if outer.sum() > 4 else 1.
    return float(rd), float(oi), float(fi)

# ── main ─────────────────────────────────────────────────────────────────────

def build_grid(scan_pdf, config_path="form_config.json",
               out_path="calibrated_grid.json"):
    with open(config_path) as f:
        config = json.load(f)
    fields_all = config["fields"]
    grid = {}

    for page_idx in range(2):
        print(f"\n=== Page {page_idx+1} ===")
        scan = pdf_page_to_image(scan_pdf, page_num=page_idx,
                                  dpi=DPI, scan_rotation=90)
        ih, iw = scan.shape[:2]

        # Estimate affine correction for THIS page
        pf = [f for f in fields_all if f.get("page",0)==page_idx]
        corr = estimate_page_correction(scan, pf)
        ax, bx, ay, by_ = corr
        print(f"  Affine: ax={ax:.4f} bx={bx:.1f} ay={ay:.4f} by={by_:.1f}")

        # Hough circles
        circles = detect_hough(scan)
        # Filter: keep only decent ring contrast
        valid = []
        for cx, cy, r in circles:
            rd, oi, fi = ring_contrast(scan, cx, cy, r)
            if rd > 0.10 and oi > 0.55:
                valid.append((int(cx), int(cy), int(r), rd, fi))
        print(f"  Hough total: {len(circles)}  Valid: {len(valid)}")

        # For matching, index valid circles by (cx,cy)
        used = [False] * len(valid)

        page_fields = [f for f in fields_all
                       if f.get("page",0)==page_idx and f.get("type")=="checkbox"]

        n_snapped = 0
        for field in page_fields:
            name = field["name"]

            # Expected center — try raw x, raw x+w/2, raw x-offset variations
            raw_ex  = int(field["x"] * iw)
            raw_ey  = int(field["y"] * ih)
            cx_ex   = int((field["x"] + field.get("w",0.014)/2) * iw)
            cy_ex   = int((field["y"] + field.get("h",0.018)/2) * ih)

            # Apply affine correction to both
            def aff(ex, ey):
                return int(ax*ex + bx), int(ay*ey + by_)

            cands = [aff(raw_ex, raw_ey), aff(cx_ex, cy_ex)]

            # Also try clamped positions
            for ex_try in [raw_ex, cx_ex]:
                for ey_try in [raw_ey, cy_ex]:
                    cands.append(aff(ex_try, ey_try))

            # Find nearest unassigned valid circle to any candidate
            r_exp = max(8, int(min(field.get("w",0.014)*iw,
                                   field.get("h",0.018)*ih) / 2))
            best_i, best_d = None, SEARCH_PX + 1
            for i, (cx, cy, r, rd, fi) in enumerate(valid):
                if used[i]:
                    continue
                for (ex, ey) in cands:
                    d = np.hypot(cx - ex, cy - ey)
                    if d < best_d:
                        best_d = d
                        best_i = i

            if best_i is not None:
                cx, cy, r, rd, fi = valid[best_i]
                used[best_i] = True
                n_snapped += 1
                grid[name] = {
                    "page": page_idx,
                    "cx": cx, "cy": cy, "r": r,
                    "ring_dark":  round(rd, 3),
                    "inner_fill": round(fi, 3),
                    "nx": round(cx / iw, 5),
                    "ny": round(cy / ih, 5),
                    "snap_dist": round(float(best_d), 1),
                }
            else:
                # Fallback: use affine-corrected config position
                ex, ey = aff(raw_ex, raw_ey)
                ex = max(r_exp, min(iw-r_exp, ex))
                ey = max(r_exp, min(ih-r_exp, ey))
                rd2, oi2, fi2 = ring_contrast(scan, ex, ey, r_exp)
                grid[name] = {
                    "page": page_idx,
                    "cx": ex, "cy": ey, "r": r_exp,
                    "ring_dark":  round(rd2, 3),
                    "inner_fill": round(fi2, 3),
                    "nx": round(ex / iw, 5),
                    "ny": round(ey / ih, 5),
                    "fallback": True,
                }
                print(f"    FALLBACK [{field['x']:.3f},{field['y']:.3f}] → ({ex},{ey}): {name}")

        print(f"  Snapped: {n_snapped}  Fallback: {len(page_fields)-n_snapped}")

        # Visualise
        vis = cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR)
        for cx, cy, r, rd, fi in valid:
            cv2.circle(vis, (cx,cy), r, (60,60,60), 1)
        for name2, g in grid.items():
            if g.get("page",0) != page_idx:
                continue
            cx, cy, r = g["cx"], g["cy"], g["r"]
            fi = g["inner_fill"]
            fb = g.get("fallback", False)
            if fb:
                color = (0, 0, 128)   # dark red = fallback
            elif fi >= 0.35:
                color = (0, 200, 0)   # green = likely checked
            elif fi >= 0.20:
                color = (0, 150, 255) # orange = maybe
            else:
                color = (200, 200, 0) # cyan = likely empty
            cv2.circle(vis, (cx,cy), r, color, 2)
        out_img = f"/sessions/laughing-gracious-wozniak/mnt/Downloads/grid_page{page_idx+1}_v3.png"
        cv2.imwrite(out_img, vis)
        print(f"  Saved → {out_img}")

    with open(out_path, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"\nSaved → {out_path}  ({len(grid)} fields)")
    return grid

if __name__ == "__main__":
    pdf_path = sorted(pathlib.Path("scans").glob("*.pdf"))[0]
    print(f"Reference scan: {pdf_path}")
    build_grid(str(pdf_path))
