"""
Survey Form OCR Processor
=========================
Reads handwritten text and checkbox answers from scanned PDF survey forms,
then exports all answers to a transposed Excel spreadsheet
(questions as rows, each survey as a column).

Supports:
  - One PDF per survey  (default)
  - One PDF containing many surveys back-to-back ("multi_survey_pdf": true in config)
    e.g. 10 pages + pages_per_form=2  →  5 surveys
  - Portrait scans of landscape forms (auto-rotated before processing)
  - Automatic serial number detection from printed paper number on left strip

Detection strategy (v2 — ink density)
--------------------------------------
Instead of aligning the scan to a template with ORB homography and then
comparing cross-correlation scores, v2 works directly on the raw (rotated)
scan:

  1. Render each scan page, rotate so it matches the form orientation.
  2. For each bubble, use the fractional coordinates from form_config.json
     to compute the expected pixel center.
  3. Run a LOCAL Hough Circle Transform in a small search window around
     that expected center to find the actual printed circle (handles minor
     scan registration offsets of ±20–30 px).
  4. Measure interior ink density: count dark pixels inside the circle,
     excluding the outer ring border (so only respondent marks count).
  5. Within each option-group, compare relative ink densities:
       threshold = min_density_in_group + margin
     Options above the threshold are "selected".

This approach is robust to:
  - Any mark style (tick ✓, cross ✗, fill ●, backslash \\, dot ·)
  - Scanner registration offsets
  - Varying pencil darkness
  - No blank template required at all

Requirements:
    pip install pymupdf opencv-python-headless pillow numpy openpyxl pytesseract
    pip install transformers torch torchvision   # TrOCR handwriting model (preferred)
    pip install easyocr                          # fallback if transformers not available
    Also install Tesseract OCR binary: https://tesseract-ocr.github.io/tessdoc/Installation.html

Usage:
    python3 survey_ocr.py --scans scans/ [--template NJTransitSurvey2026_BLANK.pdf]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
try:
    import fitz  # PyMuPDF (preferred if available)
    _HAS_FITZ = True
except ImportError:
    import pypdfium2 as pdfium  # fallback
    _HAS_FITZ = False
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from PIL import Image


# ─────────────────────────────────────────────
# PDF → Image helpers
# ─────────────────────────────────────────────

def pdf_page_to_image(pdf_path: str, page_num: int = 0, dpi: int = 200,
                      scan_rotation: int = 0,
                      color: bool = False) -> np.ndarray:
    """
    Render one page of a PDF to a numpy array.

    scan_rotation : clockwise degrees to rotate AFTER rendering.
      Use 90 when scans were placed in the scanner rotated 90° to the left
      (counter-clockwise), which is the default for NJ Transit survey scans.
      0  = no rotation  (template / blank form)
      90 = rotate CW    (scanned forms tilted left in scanner)

    color : if True, return a 3-channel BGR image; otherwise grayscale (H×W).
    """
    if _HAS_FITZ:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        if color:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width)
        doc.close()
    else:
        doc = pdfium.PdfDocument(pdf_path)
        page = doc[page_num]
        bitmap = page.render(scale=dpi / 72)
        if color:
            pil = bitmap.to_pil().convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            pil = bitmap.to_pil().convert("L")
            img = np.array(pil)

    _rot_map = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if scan_rotation in _rot_map:
        img = cv2.rotate(img, _rot_map[scan_rotation])
    return img


def pdf_page_count(pdf_path: str) -> int:
    if _HAS_FITZ:
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
    else:
        doc = pdfium.PdfDocument(pdf_path)
        n = len(doc)
    return n


# ─────────────────────────────────────────────
# Field extraction helpers
# ─────────────────────────────────────────────

def extract_roi(img: np.ndarray, field: dict,
                x_extend: float = 0.0, y_extend: float = 0.0) -> np.ndarray:
    """
    Crop the region of interest for a field.
    Coords are 0–1 fractions of page size.
    """
    h, w = img.shape[:2]
    x1 = max(0, int(field["x"] * w))
    y1 = max(0, int(field["y"] * h))
    x2 = min(w, int((field["x"] + field["w"] * (1 + x_extend)) * w))
    y2 = min(h, int((field["y"] + field["h"] * (1 + y_extend)) * h))
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else np.zeros((4, 4), dtype=np.uint8)


def read_text_field(roi: np.ndarray, reader: "TextReader",
                    template_roi: np.ndarray = None) -> str:
    """
    Recognise handwritten text in a cropped field region.
    Returns the recognised text, or '' if the field is empty / unreadable.
    """
    if template_roi is not None:
        if template_roi.shape != roi.shape:
            template_roi = cv2.resize(template_roi, (roi.shape[1], roi.shape[0]))
        diff = np.clip(
            template_roi.astype(np.float32) - roi.astype(np.float32), 0, 255
        ).astype(np.uint8)
        blurred = cv2.GaussianBlur(diff, (3, 3), 0)
        _, clean = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    else:
        _, clean = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return reader.read(clean)


# ─────────────────────────────────────────────
# Handwriting reader  (TrOCR preferred, EasyOCR fallback)
# ─────────────────────────────────────────────

class TextReader:
    """
    Unified handwriting / text reader.
    Tries TrOCR first, falls back to EasyOCR, then returns '' if neither available.
    """

    def __init__(self):
        self._mode      = None
        self._processor = None
        self._model     = None
        self._easyocr   = None

    def load(self):
        if self._try_load_trocr():
            return
        self._load_easyocr()

    def _try_load_trocr(self) -> bool:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            print("[INFO] Loading TrOCR handwriting model …")
            self._processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-handwritten")
            self._model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten")
            self._model.eval()
            self._mode = "trocr"
            print("[INFO] TrOCR loaded")
            return True
        except Exception as exc:
            print(f"[WARN] TrOCR not available ({exc})")
            return False

    def _load_easyocr(self):
        try:
            import easyocr as _easyocr
            print("[INFO] Loading EasyOCR fallback …")
            self._easyocr = _easyocr.Reader(["en"], gpu=False, verbose=False)
            self._mode = "easyocr"
            print("[INFO] EasyOCR loaded")
        except ImportError:
            print("[WARN] Neither TrOCR nor EasyOCR available — text fields blank.")
            self._mode = "none"

    def read(self, image_np: np.ndarray) -> str:
        if self._mode == "trocr":
            return self._read_trocr(image_np)
        if self._mode == "easyocr":
            return self._read_easyocr(image_np)
        return ""

    def _read_trocr(self, image_np: np.ndarray) -> str:
        import torch
        from PIL import Image as PILImage

        img = image_np.astype(np.uint8)
        h, w = img.shape[:2]
        min_dim = 64
        if h < min_dim or w < min_dim:
            scale = max(min_dim / h, min_dim / w, 1.0)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        pil_img = PILImage.fromarray(img).convert("RGB")
        pixel_values = self._processor(pil_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values, max_new_tokens=64)

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def _read_easyocr(self, image_np: np.ndarray) -> str:
        results = self._easyocr.readtext(image_np, detail=0, paragraph=True)
        return " ".join(results).strip()


# ─────────────────────────────────────────────
# Page-level grid calibration
# ─────────────────────────────────────────────

def calibrate_page_grid(color_img_or_gray: np.ndarray,
                        fields: list,
                        color_img: np.ndarray = None) -> np.ndarray:
    """
    Compute a full 2×3 affine matrix M that maps config coordinates
    (fraction × scan_size) → actual scan pixel positions.

    The calibration captures three sources of variation per scan page:
    ┌─────────────────────────────────────────────────────────────────┐
    │  • Scale   : scan DPI vs template DPI (different dimensions)    │
    │  • Tilt    : form placed at an angle on the scanner bed         │
    │  • Offset  : form not perfectly aligned to scanner edge         │
    └─────────────────────────────────────────────────────────────────┘

    Algorithm (color path — preferred)
    -----------------------------------
    1. Detect all pre-printed orange circles from the COLOR scan.
       These are the ground-truth bubble positions at this particular tilt
       and scale — no template alignment needed.
    2. Compute raw expected pixel centers from config fractions × scan size.
    3. Match each expected center to the nearest orange circle within 70px.
    4. Fit a full 6-parameter affine transform via least-squares:
         scan_x = M[0,0]*exp_x + M[0,1]*exp_y + M[0,2]
         scan_y = M[1,0]*exp_x + M[1,1]*exp_y + M[1,2]
       This captures scale, rotation (tilt), and translation simultaneously.

    Algorithm (grayscale fallback)
    --------------------------------
    Same as before: Hough on grayscale + independent x/y axis fit.
    Returns a 2×3 matrix encoding the (ax, bx, ay, by) coefficients
    [ax 0 bx; 0 ay by].

    Returns
    -------
    M : 2×3 numpy float64 array
        Apply with:  scan_pos = M @ [exp_x, exp_y, 1]^T
    """
    H, W = color_img_or_gray.shape[:2]
    cb_fields = [f for f in fields if f.get("type") == "checkbox"]
    if not cb_fields:
        return np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64), float("inf")

    # Expected pixel centers from config fractions
    expected = np.array(
        [[(f["x"] + f["w"] / 2) * W, (f["y"] + f["h"] / 2) * H]
         for f in cb_fields],
        dtype=np.float64
    )

    detected = None

    # ── Color path: use orange circles ───────────────────────────────────────
    if color_img is not None or (len(color_img_or_gray.shape) == 3):
        ci = color_img if color_img is not None else color_img_or_gray
        orange_circles = detect_orange_bubbles(ci)
        if orange_circles.shape[0] >= 4:
            detected = orange_circles[:, :2].astype(np.float64)

    # ── Grayscale path: Hough on gray channel ─────────────────────────────
    if detected is None:
        gray = (cv2.cvtColor(color_img_or_gray, cv2.COLOR_BGR2GRAY)
                if len(color_img_or_gray.shape) == 3 else color_img_or_gray)
        er = max(6, int(np.median([min(f["w"] * W, f["h"] * H) / 2
                                   for f in cb_fields])))
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        all_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=er * 2,
            param1=60, param2=14,
            minRadius=max(4, er - 12), maxRadius=er + 14,
        )
        if all_circles is not None:
            detected = all_circles[0, :, :2].astype(np.float64)
            print(f"    [CORR] Page Hough detected {len(detected)} candidate circles"
                  f"  (nominal er={er})")

    if detected is None or len(detected) < 4:
        return np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64), float("inf")

    # ── Match expected → detected ─────────────────────────────────────────
    # For each expected position find the nearest UNUSED detected circle ≤ 70 px.
    # Deduplication is critical: if two adjacent expected positions both map to
    # the same detected circle (because config positions are not perfectly
    # calibrated), allowing duplicates would give contradictory least-squares
    # constraints and inflate residuals dramatically.
    matched_exp = []
    matched_det = []
    used_idx: set = set()
    for ex, ey in expected:
        dists = np.sqrt((detected[:, 0] - ex) ** 2 + (detected[:, 1] - ey) ** 2)
        order = np.argsort(dists)          # closest first
        for bi in order:
            if bi in used_idx:
                continue                   # already claimed — try next
            # This is the nearest unused circle
            if dists[bi] <= 70:
                matched_exp.append([ex, ey])
                matched_det.append([detected[bi, 0], detected[bi, 1]])
                used_idx.add(bi)
            break                          # stop after nearest unused (match or not)

    n = len(matched_exp)
    print(f"    [CALIB] Matched {n} bubbles for affine fit")
    if n < 4:
        return np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64), float("inf")

    src = np.array(matched_exp, dtype=np.float64)   # config positions (raw fractions × size)
    dst = np.array(matched_det, dtype=np.float64)   # actual orange-circle positions in scan

    # ── Fit independent per-axis scale + translate ────────────────────────
    #   dst_x = ax * src_x + bx   (scale and offset in x)
    #   dst_y = ay * src_y + by   (scale and offset in y)
    # ── Fit 4-param diagonal (scale + translate per axis) ────────────────
    Ax = np.column_stack([src[:, 0], np.ones(n)])
    Ay = np.column_stack([src[:, 1], np.ones(n)])
    (ax, bx), _, _, _ = np.linalg.lstsq(Ax, dst[:, 0], rcond=None)
    (ay_c, by_c), _, _, _ = np.linalg.lstsq(Ay, dst[:, 1], rcond=None)

    pred_x = ax * src[:, 0] + bx
    pred_y = ay_c * src[:, 1] + by_c
    res = np.sqrt((pred_x - dst[:, 0]) ** 2 + (pred_y - dst[:, 1]) ** 2)

    # Tilt angle from full 6-param affine (informational — logged but not used
    # for position correction, since the grayscale Hough provides a more stable
    # affine fit with ~10× more reference circles spread across the full page)
    A_full = np.hstack([src, np.ones((n, 1))])
    (a, b, c_), _, _, _ = np.linalg.lstsq(A_full, dst[:, 0], rcond=None)
    (d, e, f_), _, _, _ = np.linalg.lstsq(A_full, dst[:, 1], rcond=None)
    theta = float(np.arctan2(d, a) * 180 / np.pi)

    res_mean = float(np.mean(res))
    res_max  = float(np.max(res))
    print(f"    [CALIB] Grid: ax={ax:.4f} bx={bx:.1f}  ay={ay_c:.4f} by={by_c:.1f}  "
          f"tilt≈{theta:.3f}°  "
          f"residual mean={res_mean:.1f}px  max={res_max:.1f}px")

    M = np.array([[ax, 0., bx],
                  [0., ay_c, by_c]], dtype=np.float64)
    return M, res_mean


def affine_apply(M: np.ndarray, x: float, y: float) -> tuple:
    """Apply a 2×3 affine matrix: returns (mapped_x, mapped_y) as ints."""
    pt = M @ np.array([x, y, 1.0])
    return int(pt[0]), int(pt[1])


def estimate_page_correction(scan_img: np.ndarray, fields: list) -> tuple:
    """
    Legacy function — kept for backward compatibility with grayscale pipeline.

    Computes a simple per-axis affine correction (scale + translate) using
    Hough circles on a grayscale scan.

    Returns (ax, bx, ay, by) — the affine correction coefficients — or
    (1.0, 0.0, 1.0, 0.0) (identity) if too few circles are detected.
    """
    H, W = scan_img.shape[:2]

    # Nominal bubble radius from field list
    cb_fields = [f for f in fields if f.get("type") == "checkbox"]
    if not cb_fields:
        return (1.0, 0.0, 1.0, 0.0)
    er = max(6, int(np.median([min(f["w"] * W, f["h"] * H) / 2
                               for f in cb_fields])))

    # Full-page Hough with generous range
    blurred = cv2.GaussianBlur(scan_img, (5, 5), 1.5)
    all_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=er * 2,
        param1=60,
        param2=14,
        minRadius=max(4, er - 12),
        maxRadius=er + 14,
    )

    if all_circles is None:
        return (1.0, 0.0, 1.0, 0.0)

    detected = np.round(all_circles[0, :]).astype(int)  # (cx, cy, r)
    print(f"    [CORR] Page Hough detected {len(detected)} candidate circles  "
          f"(nominal er={er})")

    # Match expected → nearest detected
    matches_exp, matches_det = [], []
    used_idx = set()

    for field in cb_fields:
        ex = int((field["x"] + field["w"] / 2) * W)
        ey = int((field["y"] + field["h"] / 2) * H)

        best_dist, best_i = float("inf"), -1
        for i, (cx, cy, _r) in enumerate(detected):
            if i in used_idx:
                continue
            d = float(np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2))
            if d < best_dist:
                best_dist, best_i = d, i

        if best_dist <= 70 and best_i >= 0:
            matches_exp.append((ex, ey))
            matches_det.append((int(detected[best_i, 0]), int(detected[best_i, 1])))
            used_idx.add(best_i)

    n = len(matches_exp)
    print(f"    [CORR] Matched {n} bubbles for affine fit")
    if n < 4:
        return (1.0, 0.0, 1.0, 0.0)

    exp_arr = np.array(matches_exp, dtype=np.float64)
    det_arr = np.array(matches_det, dtype=np.float64)

    # Fit x: det_x = ax * exp_x + bx
    A = np.column_stack([exp_arr[:, 0], np.ones(n)])
    (ax, bx), _, _, _ = np.linalg.lstsq(A, det_arr[:, 0], rcond=None)

    # Fit y: det_y = ay * exp_y + by
    A = np.column_stack([exp_arr[:, 1], np.ones(n)])
    (ay, by_), _, _, _ = np.linalg.lstsq(A, det_arr[:, 1], rcond=None)

    # Compute residuals to report accuracy
    pred_x = ax * exp_arr[:, 0] + bx
    pred_y = ay * exp_arr[:, 1] + by_
    res = np.sqrt((pred_x - det_arr[:, 0]) ** 2 + (pred_y - det_arr[:, 1]) ** 2)
    print(f"    [CORR] Affine fit: ax={ax:.4f} bx={bx:.1f}  ay={ay:.4f} by={by_:.1f}  "
          f"residual mean={np.mean(res):.1f}px  max={np.max(res):.1f}px")

    return (float(ax), float(bx), float(ay), float(by_))


# ─────────────────────────────────────────────
# Bubble detection  (v2 — ink density)
# ─────────────────────────────────────────────

def _locate_bubble(scan_img: np.ndarray, field: dict,
                   search_px: int = 0,
                   correction: tuple = None) -> tuple:
    """
    Locate the actual circle center near the expected config position.

    Steps
    -----
    1. Compute expected center from config fractions.
    2. Apply affine correction (ax, bx, ay, by) if provided — this shifts the
       expected center to compensate for the systematic scan registration offset
       estimated by estimate_page_correction().
    3. Run local Hough Circle Transform within search_px of corrected expected.
    4. Return the Hough-found center; fall back to corrected expected if no match.

    Returns (cx, cy, radius) in pixel coordinates.
    """
    h, w = scan_img.shape[:2]

    # Raw expected center and radius from config fractions
    ex = int((field["x"] + field["w"] / 2) * w)
    ey = int((field["y"] + field["h"] / 2) * h)
    er = max(6, int(min(field["w"] * w, field["h"] * h) / 2))

    # Apply affine correction to get a better starting point
    if correction is not None:
        ax, bx, ay, by_ = correction
        ex = int(ax * ex + bx)
        ey = int(ay * ey + by_)

    # Clamp to image bounds
    ex = max(er, min(w - er, ex))
    ey = max(er, min(h - er, ey))

    # Search region: search_px beyond the corrected expected bubble boundary
    pad = search_px + er + 4
    x1 = max(0, ex - pad);  y1 = max(0, ey - pad)
    x2 = min(w, ex + pad);  y2 = min(h, ey + pad)

    roi = scan_img[y1:y2, x1:x2]
    if roi.shape[0] < er or roi.shape[1] < er:
        return ex, ey, er

    # Hough needs a blurred image to work reliably
    blurred = cv2.GaussianBlur(roi, (5, 5), 1.2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(er, 8),   # minimum distance between circle centres
        param1=60,            # Canny upper threshold
        param2=14,            # accumulator threshold (lower → more detections)
        minRadius=max(4, er - 12),
        maxRadius=er + 14,
    )

    if circles is not None:
        circles_int = np.round(circles[0, :]).astype(int)
        # Pick the circle whose centre is closest to our corrected expected position
        best = min(circles_int,
                   key=lambda c: (c[0] + x1 - ex) ** 2 + (c[1] + y1 - ey) ** 2)
        dist = float(np.sqrt((best[0] + x1 - ex) ** 2 + (best[1] + y1 - ey) ** 2))
        if dist <= search_px:          # only trust circles close enough
            return int(best[0] + x1), int(best[1] + y1), int(best[2])

    return ex, ey, er


def _interior_ink_density(scan_img: np.ndarray,
                          cx: int, cy: int, radius: int,
                          ring_frac: float = 0.22,
                          ink_threshold: int = 128) -> float:
    """
    Measure the fraction of pixels INSIDE a bubble that are ink/marks,
    excluding the outer ring border (the printed circle line itself).

    Uses a two-stage adaptive threshold:
      1. Sample the paper brightness from pixels outside the bubble.
      2. Set threshold = min(paper_brightness * 0.85, ink_threshold_cap).
    This avoids Otsu's false positives on uniform-white areas (empty bubbles)
    where Otsu would split the narrow 190-230 white range at ~210 and report
    ~50% density even for completely blank bubbles.

    Separately, inner_r is capped at MAX_INNER_R pixels so that measurement
    never overlaps the printed ring border ink (actual bubble radius ≈ 12-16 px
    while config-derived radius er ≈ 22 px, causing inner_r=17 to include ring).

    ring_frac       : fraction of radius to exclude as ring (default 22%)
    ink_threshold   : hard cap on the adaptive threshold (default 128)
                      — keeps dark-mark detection strict even on bright scans

    Returns 0.0 (blank interior) to 1.0 (fully inked).
    """
    # Cap inner radius: actual printed bubbles are ~12-16 px radius.
    # Config field half-widths give er≈22 px → inner_r=17 px which overlaps
    # the ring border and inflates scores.  Clamping at 9 px stays safely inside.
    MAX_INNER_R = 9
    inner_r = min(MAX_INNER_R, max(3, int(radius * (1.0 - ring_frac))))

    pad = 6
    x1 = max(0, cx - radius - pad);  y1 = max(0, cy - radius - pad)
    x2 = min(scan_img.shape[1], cx + radius + pad)
    y2 = min(scan_img.shape[0], cy + radius + pad)

    roi = scan_img[y1:y2, x1:x2]
    if roi.size < 4:
        return 0.0

    rh, rw = roi.shape[:2]
    lx = min(max(0, cx - x1), rw - 1)
    ly = min(max(0, cy - y1), rh - 1)

    # Interior mask: filled circle at inner_r (inside the ring)
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.circle(mask, (lx, ly), inner_r, 255, -1)

    interior_pixels = roi[mask == 255]
    if interior_pixels.size < 4:
        return 0.0

    # ── Adaptive threshold from nearby paper background ───────────────────
    # Sample paper pixels OUTSIDE the full bubble area to get true brightness.
    # 90th-percentile brightness → set threshold at 85% of that, capped at
    # ink_threshold so we never accept ring-border gray as respondent marks.
    outer_mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.circle(outer_mask, (lx, ly), radius + 2, 255, -1)
    outside_pixels = roi[outer_mask == 0]

    if outside_pixels.size >= 10:
        paper_bright = float(np.percentile(outside_pixels, 90))
        # 85% of paper brightness as the ink/paper divide
        adaptive = int(paper_bright * 0.85)
        thresh = min(adaptive, ink_threshold)   # never looser than hard cap
    else:
        thresh = ink_threshold

    # Count interior pixels darker than threshold (= respondent marks)
    ink_pixels   = int(np.sum(interior_pixels < thresh))
    total_pixels = len(interior_pixels)

    return ink_pixels / max(1, total_pixels)


def _bubble_ink_score(scan_img: np.ndarray, field: dict,
                      search_px: int = 65,
                      correction: tuple = None) -> float:
    """
    Full pipeline: locate the bubble then measure interior ink density.
    Returns 0.0–1.0 (higher = more ink = more likely selected).
    """
    cx, cy, radius = _locate_bubble(scan_img, field,
                                    search_px=search_px,
                                    correction=correction)
    return _interior_ink_density(scan_img, cx, cy, radius)


# ─────────────────────────────────────────────────────────────────────────────
# Color-based bubble detection  (for color / orange-circle scans)
# ─────────────────────────────────────────────────────────────────────────────

# HSV range for the orange pre-printed answer circles on the NJ Transit form.
# Tuned empirically from scanned pages (H=5–35, S>60, V>60).
_ORANGE_HSV_LO = np.array([5,  60,  60], dtype=np.uint8)
_ORANGE_HSV_HI = np.array([35, 255, 255], dtype=np.uint8)


def detect_orange_bubbles(color_img: np.ndarray,
                          min_r: int = 5, max_r: int = 20,
                          min_dist: int = 15) -> np.ndarray:
    """
    Find all pre-printed orange answer circles in a color BGR scan page.

    Returns an Nx3 int array of [cx, cy, radius], or empty array if none found.
    Uses Hough on the orange channel only, so text/lines (black/blue) are ignored.
    """
    hsv    = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, _ORANGE_HSV_LO, _ORANGE_HSV_HI)

    circles = cv2.HoughCircles(
        orange,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=30,
        param2=8,          # low threshold — orange mask is clean binary signal
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.int32)
    return np.round(circles[0]).astype(np.int32)


def is_color_scan(color_img: np.ndarray, min_orange_px: int = 500) -> bool:
    """
    Return True if the image contains enough orange pixels to be treated as
    a color scan with pre-printed orange bubbles.
    """
    hsv    = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, _ORANGE_HSV_LO, _ORANGE_HSV_HI)
    return int(orange.sum() // 255) >= min_orange_px


def snap_to_orange_bubble(expected_cx: int, expected_cy: int,
                          orange_circles: np.ndarray,
                          max_snap_px: int = 40) -> tuple:
    """
    Find the orange bubble closest to an expected config position.

    Parameters
    ----------
    expected_cx, expected_cy : config-derived (+ affine-corrected) center
    orange_circles           : Nx3 array [cx, cy, r] from detect_orange_bubbles()
    max_snap_px              : maximum snap distance in pixels (default 40)

    Returns (cx, cy, radius) of the snapped circle, or (expected_cx, expected_cy, 8)
    if no circle is close enough.
    """
    if orange_circles.shape[0] == 0:
        return expected_cx, expected_cy, 8

    dists = np.sqrt((orange_circles[:, 0] - expected_cx) ** 2 +
                    (orange_circles[:, 1] - expected_cy) ** 2)
    best_i = int(np.argmin(dists))
    if dists[best_i] <= max_snap_px:
        c = orange_circles[best_i]
        return int(c[0]), int(c[1]), int(c[2])
    return expected_cx, expected_cy, 8


def _color_ink_density(color_img: np.ndarray,
                       cx: int, cy: int, radius: int) -> float:
    """
    Measure respondent ink density inside an orange pre-printed bubble.

    The strategy differs from the grayscale version:
    - The bubble interior should be WHITE (empty) or DARK (filled with pen/pencil).
    - The printed orange ring is excluded by measuring inside (radius - 2) pixels.
    - We count pixels that are DARK (value < 130) AND NOT orange.
      This ignores scan noise in the orange ink channel while catching
      blue/black pen, pencil, and marker marks reliably.

    Returns 0.0 (empty) – 1.0 (fully filled).
    """
    inner_r = max(3, radius - 2)

    h, w = color_img.shape[:2]
    pad  = radius + 4
    x1 = max(0, cx - pad);  y1 = max(0, cy - pad)
    x2 = min(w, cx + pad);  y2 = min(h, cy + pad)

    roi_bgr = color_img[y1:y2, x1:x2]
    if roi_bgr.size < 12:
        return 0.0

    rh, rw = roi_bgr.shape[:2]
    lx = min(max(0, cx - x1), rw - 1)
    ly = min(max(0, cy - y1), rh - 1)

    # Interior mask
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.circle(mask, (lx, ly), inner_r, 255, -1)

    # Dark-pixel detection: grayscale value < 130 (pen/pencil marks)
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = (roi_gray < 130).astype(np.uint8)

    # Exclude orange pixels from "dark" count (avoids counting the ring edge)
    roi_hsv  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    orange_m = cv2.inRange(roi_hsv, _ORANGE_HSV_LO, _ORANGE_HSV_HI)
    ink_mask = dark_mask & (orange_m == 0)   # dark AND not orange

    interior_ink   = int(np.sum(mask & (ink_mask * 255) > 0))
    interior_total = int(np.sum(mask > 0))

    return interior_ink / max(1, interior_total)


def build_color_grid(color_img: np.ndarray,
                     fields: list,
                     correction=None) -> dict:
    """
    Build a mapping {field_name → (cx, cy, radius)} using the orange circles
    detected in the scan as ground-truth bubble positions.

    For each config field:
    1. Compute expected center from config fractions × scan dimensions.
    2. Apply the grid calibration (2×3 affine matrix OR legacy 4-tuple) to
       account for scan scale, tilt, and offset.
    3. Snap to the nearest detected orange circle within max_snap_px.
    4. If no circle is within range, keep the calibrated expected position.

    The calibration matrix M is computed by calibrate_page_grid() which fits
    a full affine transform (scale + rotation + translation) to the orange
    circle positions detected in THIS scan page — so every page's tilt, zoom,
    and placement offset is independently corrected.

    Returns a dict ready for ink measurement.
    """
    h, w  = color_img.shape[:2]
    bubbles = detect_orange_bubbles(color_img)
    print(f"    [COLOR] {len(bubbles)} orange bubbles detected on page")

    grid = {}
    for f in fields:
        raw_x = (f["x"] + f["w"] / 2) * w
        raw_y = (f["y"] + f["h"] / 2) * h

        if correction is not None:
            if isinstance(correction, np.ndarray) and correction.shape == (2, 3):
                # Full affine matrix from calibrate_page_grid
                ex, ey = affine_apply(correction, raw_x, raw_y)
            else:
                # Legacy 4-tuple (ax, bx, ay, by) from estimate_page_correction
                ax, bx, ay, by_ = correction
                ex = int(ax * raw_x + bx)
                ey = int(ay * raw_y + by_)
        else:
            ex, ey = int(raw_x), int(raw_y)

        ex = max(0, min(w - 1, ex))
        ey = max(0, min(h - 1, ey))
        cx, cy, r = snap_to_orange_bubble(ex, ey, bubbles)
        grid[f["name"]] = (cx, cy, r)

    return grid


def read_checkbox_group_color(color_img: np.ndarray,
                              option_fields: list,
                              correction: tuple = None,
                              max_selections: int = None,
                              dev_margin: float = 0.06) -> dict:
    """
    Color-aware version of read_checkbox_group.

    Uses orange bubble detection to snap to actual printed circle positions,
    then measures dark (non-orange) ink inside each bubble to detect filled marks.
    Identical winner-takes-all / multi-select logic as the grayscale version.
    """
    if not option_fields:
        return {}

    # Build grid: snap each field to nearest orange circle
    grid = build_color_grid(color_img, option_fields, correction=correction)

    # Deduplication: if two fields snapped to the same circle, the farther one
    # falls back to its raw corrected expected position
    h, w = color_img.shape[:2]
    from collections import defaultdict
    circle_users = defaultdict(list)

    def _apply_correction(raw_x, raw_y):
        """Apply correction (matrix or 4-tuple) to raw expected position."""
        if correction is None:
            return int(raw_x), int(raw_y)
        if isinstance(correction, np.ndarray) and correction.shape == (2, 3):
            return affine_apply(correction, raw_x, raw_y)
        ax, bx, ay, by_ = correction
        return int(ax * raw_x + bx), int(ay * raw_y + by_)

    for f in option_fields:
        cx, cy, r = grid[f["name"]]
        raw_x = (f["x"] + f["w"] / 2) * w
        raw_y = (f["y"] + f["h"] / 2) * h
        ex, ey = _apply_correction(raw_x, raw_y)
        dist = float(np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2))
        circle_users[(cx, cy)].append((dist, f["name"], f))

    bubbles = detect_orange_bubbles(color_img)
    for circle_pos, claimants in circle_users.items():
        if len(claimants) <= 1:
            continue
        claimants.sort(key=lambda t: t[0])
        for _dist, fname, f in claimants[1:]:
            raw_x = (f["x"] + f["w"] / 2) * w
            raw_y = (f["y"] + f["h"] / 2) * h
            ex, ey = _apply_correction(raw_x, raw_y)
            # Snap again but excluding the contested circle
            contested = np.array(circle_pos)
            mask_arr  = np.array([
                not (b[0] == contested[0] and b[1] == contested[1])
                for b in bubbles
            ])
            remaining = bubbles[mask_arr] if bubbles.shape[0] > 0 else bubbles
            cx2, cy2, r2 = snap_to_orange_bubble(ex, ey, remaining)
            grid[fname] = (cx2, cy2, r2)

    # Measure ink at each snapped circle
    scores = {f["name"]: _color_ink_density(color_img, *grid[f["name"]])
              for f in option_fields}

    # Artifact cap (same as grayscale version)
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] >= 0.95:
        gap = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        if gap >= 0.35:
            for name in scores:
                if scores[name] >= 0.95:
                    print(f"      [ARTIFACT] {name}: score {scores[name]:.3f} → zeroed")
                    scores[name] = 0.0

    # Winner-takes-all
    if max_selections == 1:
        best_name  = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]
        min_ink    = 0.05    # color approach is cleaner — use slightly higher floor
        prefix     = option_fields[0]["name"].split(": ")[0] if ": " in option_fields[0]["name"] \
                     else option_fields[0]["name"]
        score_str  = "  ".join(f"{f['name'].split(': ')[-1]}={scores[f['name']]:.3f}"
                                for f in option_fields)
        print(f"      [COLOR-SCORES] {prefix} (max_sel=1): winner={best_name.split(': ')[-1]} "
              f"score={best_score:.3f} | {score_str}")
        result = {}
        for name in scores:
            result[name] = "Yes" if (name == best_name and best_score >= min_ink) else "No"
        return result

    # Multi-select adaptive threshold (same logic as grayscale)
    baseline     = min(scores.values())
    max_score    = max(scores.values())
    thr_relative = baseline + dev_margin
    thr_dominant = max_score * 0.55 if max_score > 0.10 else 0.0
    threshold    = max(thr_relative, thr_dominant)

    prefix    = option_fields[0]["name"].split(": ")[0] if ": " in option_fields[0]["name"] \
                else option_fields[0]["name"]
    score_str = "  ".join(f"{f['name'].split(': ')[-1]}={scores[f['name']]:.3f}"
                           for f in option_fields)
    print(f"      [COLOR-SCORES] {prefix}: base={baseline:.3f} thr={threshold:.3f} "
          f"(rel={thr_relative:.3f} dom={thr_dominant:.3f}) | {score_str}")

    return {name: ("Yes" if score >= threshold else "No")
            for name, score in scores.items()}


def read_checkbox_group(scan_img: np.ndarray,
                        option_fields: list,
                        template_img=None,    # kept for API compatibility
                        dev_margin: float = 0.06,
                        correction: tuple = None,
                        max_selections: int = None) -> dict:
    """
    Detect which options in a question group are selected.

    Strategy — relative ink density comparison
    ------------------------------------------
    For every option bubble we measure how much respondent ink is inside
    (excluding the printed ring border).  We then compare RELATIVELY:

        threshold = min_ink_density_in_group + dev_margin

    Options with ink density above the threshold are marked "Yes".
    This self-calibrates: even with scan noise, only the truly marked
    bubble(s) will be significantly darker than their blank siblings.

    Parameters
    ----------
    dev_margin : ink-density margin above the group minimum to count as
                 "selected".  0.06 (6% of interior pixels) works for light
                 pencil ticks.  Raise to 0.10 for cleaner separation.
    """
    if not option_fields:
        return {}

    # ── Step 1: Locate each bubble (Hough + affine correction) ───────────────
    # search_px=25 lets Hough snap to the actual printed circle within 25px of the
    # affine-corrected expected position.  This handles small per-scan registration
    # variations (±20 px) without risking false matches across question rows.
    h, w = scan_img.shape[:2]
    located = {}   # field_name → (cx, cy, radius)
    for f in option_fields:
        located[f["name"]] = _locate_bubble(scan_img, f, search_px=0,
                                            correction=correction)

    # ── Step 2: Deduplication — prevent two fields claiming the same circle ──
    # If two different fields located the exact same pixel center, the one
    # whose corrected expected position is farther away loses the claim and
    # falls back to its own expected center.
    def _corrected_expected(f):
        ex = int((f["x"] + f["w"] / 2) * w)
        ey = int((f["y"] + f["h"] / 2) * h)
        if correction:
            ax, bx, ay, by_ = correction
            ex = max(0, min(w, int(ax * ex + bx)))
            ey = max(0, min(h, int(ay * ey + by_)))
        return ex, ey

    # Build a map from (cx, cy) → [field_names that found that circle]
    from collections import defaultdict
    circle_users = defaultdict(list)   # (cx,cy) → list of (dist, fname, field)
    for f in option_fields:
        cx, cy, r = located[f["name"]]
        ex, ey = _corrected_expected(f)
        dist = float(np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2))
        circle_users[(cx, cy)].append((dist, f["name"], f))

    # For each contested circle, only the closest field keeps it; others fall back
    for circle_pos, claimants in circle_users.items():
        if len(claimants) <= 1:
            continue
        claimants.sort(key=lambda t: t[0])          # sort by distance
        for _dist, fname, f in claimants[1:]:        # losers fall back to expected
            ex_raw = int((f["x"] + f["w"] / 2) * w)
            ey_raw = int((f["y"] + f["h"] / 2) * h)
            er = max(6, int(min(f["w"] * w, f["h"] * h) / 2))
            if correction:
                ax, bx, ay, by_ = correction
                ex_raw = max(er, min(w - er, int(ax * ex_raw + bx)))
                ey_raw = max(er, min(h - er, int(ay * ey_raw + by_)))
            located[fname] = (ex_raw, ey_raw, er)   # fallback to corrected expected

    # ── Step 3: Measure ink at each located circle ───────────────────────────
    scores = {f["name"]: _interior_ink_density(scan_img, *located[f["name"]])
              for f in option_fields}

    # ── Artifact cap: neutralise impossible scores (e.g. solid printed element) ──
    # If any single score is >= 0.95 AND at least 0.35 above every other score,
    # it is almost certainly a printed form element, not a respondent mark.
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] >= 0.95:
        gap = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        if gap >= 0.35:
            for name in scores:
                if scores[name] >= 0.95:
                    print(f"      [ARTIFACT] {name}: score {scores[name]:.3f} → zeroed (form element)")
                    scores[name] = 0.0

    # ── Winner-takes-all for "choose one" question groups ───────────────────────
    # When max_selections=1, only the highest-scoring option is returned as "Yes",
    # provided it clears a minimum ink threshold (avoids selecting blank groups).
    if max_selections == 1:
        best_name = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]
        # With inner_r capped at 9px and fixed threshold=128, a filled bubble
        # scores ~0.05-0.20 (smaller area, stricter threshold than before).
        # Empty bubbles score exactly 0.000, so a low floor like 0.03 reliably
        # separates "answered" from "unanswered" while not requiring high density.
        min_ink_for_selection = 0.03
        prefix = option_fields[0]["name"].split(": ")[0] if ": " in option_fields[0]["name"] \
                 else option_fields[0]["name"]
        score_str = "  ".join(f"{f['name'].split(': ')[-1]}={scores[f['name']]:.3f}"
                               for f in option_fields)
        print(f"      [SCORES] {prefix} (max_sel=1): winner={best_name.split(': ')[-1]} "
              f"score={best_score:.3f} | {score_str}")
        result = {}
        for name in scores:
            result[name] = "Yes" if (name == best_name and best_score >= min_ink_for_selection) else "No"
        return result

    # ── Adaptive threshold: hybrid of relative-minimum and relative-maximum ──
    #
    # thr_relative = baseline + margin
    #   Catches lightly-marked bubbles in otherwise-blank groups
    #   e.g. Q20 No=0.075, Yes=0.007 → threshold=0.067 → only No selected ✓
    #
    # thr_dominant = max_score * 0.55   (only when max > 0.10)
    #   When one bubble clearly dominates, require other options to reach ≥55%
    #   of the winner's ink to also be counted as selected.
    #   e.g. Q27 No=0.398, Hearing=0.129: 0.398*0.55=0.219 > 0.129 → only No ✓
    #
    # We take the HIGHER of the two thresholds (more restrictive).
    # When max ≤ 0.10 (very light marks), thr_dominant falls to 0.0 so the
    # relative threshold governs alone — light pencil ticks are still caught.
    baseline     = min(scores.values())
    max_score    = max(scores.values())
    thr_relative = baseline + dev_margin                 # relative approach
    thr_dominant = max_score * 0.55 if max_score > 0.10 else 0.0
    threshold    = max(thr_relative, thr_dominant)       # more restrictive wins

    prefix = option_fields[0]["name"].split(": ")[0] if ": " in option_fields[0]["name"] \
             else option_fields[0]["name"]
    score_str = "  ".join(f"{f['name'].split(': ')[-1]}={scores[f['name']]:.3f}"
                           for f in option_fields)
    print(f"      [SCORES] {prefix}: base={baseline:.3f} thr={threshold:.3f} "
          f"(rel={thr_relative:.3f} dom={thr_dominant:.3f}) | {score_str}")

    return {name: ("Yes" if score >= threshold else "No")
            for name, score in scores.items()}


def read_standalone_checkbox(scan_img: np.ndarray,
                             field: dict,
                             template_img=None,
                             correction: tuple = None) -> str:
    """
    Detect a single checkbox not part of an option group.
    Uses absolute ink-density threshold (no siblings to compare against).
    """
    score = _bubble_ink_score(scan_img, field, correction=correction)
    # Absolute floor: 5% of interior pixels being dark counts as "Yes"
    result = "Yes" if score >= 0.05 else "No"
    print(f"      [STANDALONE] {field['name']}: ink={score:.3f} → {result}")
    return result


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Serial number detection
# ─────────────────────────────────────────────

def extract_serial_number(pdf_path: str, page_num: int, dpi: int = 200,
                          scan_rotation: int = 90) -> str:
    """
    Read the printed 4-digit serial number from the left strip of a front page.

    The serial is printed vertically on the left edge of the physical form.
    On the PORTRAIT scan (before rotation) it appears at the bottom-left.
    We look at the portrait image (before the 90° CW rotation) because the
    serial is at the left edge of the portrait page.

    Returns the 4-digit serial string (e.g. '0001') or '' if detection fails.
    """
    try:
        import pytesseract  # type: ignore[import-untyped]
    except ImportError:
        return ""

    try:
        # Read WITHOUT rotation so serial strip is on the left of portrait page
        img = pdf_page_to_image(pdf_path, page_num, dpi, scan_rotation=0)
        h, w = img.shape

        # Serial is printed vertically on the left edge, lower half of portrait
        crop = img[int(h * 0.80):int(h * 0.97), 0:int(w * 0.06)]

        # Rotate the strip 90° CW to make digits upright
        rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        upscaled = cv2.resize(rotated, None, fx=4, fy=4,
                              interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(upscaled, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bordered = cv2.copyMakeBorder(thresh, 20, 20, 20, 20,
                                      cv2.BORDER_CONSTANT, value=255)

        raw = pytesseract.image_to_string(
            bordered,
            config="--psm 6 -c tessedit_char_whitelist=0123456789 --oem 3"
        ).strip()

        m = re.search(r'\b\d{4}\b', raw)
        if m:
            return m.group()

        digits = re.sub(r'\D', '', raw)
        if len(digits) == 4:
            return digits

    except Exception as exc:
        print(f"  [WARN] Serial detection failed on page {page_num}: {exc}")

    return ""


# ─────────────────────────────────────────────
# Survey iterator
# ─────────────────────────────────────────────

def iter_surveys(scan_files, pages_per_form: int, multi_mode: bool,
                 dpi: int = 200, scan_rotation: int = 90, skip_pages: int = 0):
    """
    Yields (survey_label, pdf_path, page_offset) tuples.

    skip_pages : number of leading pages to ignore in each PDF before the first
                 survey begins (e.g. 1 to skip a cover/instruction page).
    """
    skip_pages = max(0, int(skip_pages))
    for pdf_path in scan_files:
        stem = pdf_path.stem
        if not multi_mode:
            yield stem, str(pdf_path), skip_pages
        else:
            try:
                total = pdf_page_count(str(pdf_path))
            except Exception as e:
                print(f"  [WARN] Cannot open {pdf_path.name}: {e}")
                continue
            usable    = max(0, total - skip_pages)
            n_surveys = usable // pages_per_form
            remainder = usable % pages_per_form
            if skip_pages:
                print(f"  [SKIP] Ignoring first {skip_pages} page(s) of "
                      f"{pdf_path.name}")
            if remainder:
                print(f"  [WARN] {pdf_path.name} has {usable} usable page(s) "
                      f"(after skipping {skip_pages}) — not a multiple of "
                      f"{pages_per_form}. Last {remainder} page(s) ignored.")
            for i in range(n_surveys):
                page_offset = skip_pages + i * pages_per_form
                serial = extract_serial_number(str(pdf_path), page_offset, dpi,
                                               scan_rotation=scan_rotation)
                if serial:
                    label = serial
                    print(f"  [SERIAL] Survey {i+1}: detected serial '{serial}'")
                else:
                    label = f"{stem}_survey{i+1:03d}"
                    print(f"  [SERIAL] Survey {i+1}: serial not detected — "
                          f"using fallback label '{label}'")
                yield label, str(pdf_path), page_offset


# ─────────────────────────────────────────────
# Process one survey
# ─────────────────────────────────────────────

def process_one_survey(label, pdf_path, page_offset,
                       template_pages, fields, reader, dpi,
                       scan_rotation: int = 90,
                       debug_dir: str = None,
                       group_config: dict = None,
                       blank_cal: dict = None, blank_meta: dict = None,
                       annotate_dir: str = None):
    """
    Returns a dict {field_name: value} for one survey.

    If blank_cal/blank_meta are supplied (built once from a blank template),
    checkbox groups are read with the blank-anchored grid pipeline: every form
    is registered onto the calibrated bubble grid, which handles the variable
    cut/length of mail-back forms far better than fixed config coordinates.

    Detection strategy (auto-selects color or grayscale):
    -------------------------------------------------------
    1. Render each scan page in COLOR and rotate by scan_rotation degrees.
    2. Check for orange pixels → if present, use the COLOR pipeline:
         a. Detect all pre-printed orange circles via Hough on the orange channel.
         b. Build a per-page grid by snapping each config field to the nearest
            orange circle (within max_snap_px), validated by the affine correction.
         c. Measure dark (non-orange) ink inside each snapped bubble.
       If NO orange pixels detected → fall back to the GRAYSCALE pipeline (original).
    3. Within each question group apply winner-takes-all / multi-select logic.
    """
    results = {"Survey": label}

    # ── Blank-anchored path ────────────────────────────────────────────────
    # When a calibrated blank is available, read checkbox groups by registering
    # this form onto the calibrated bubble grid (handles the variable cut).
    if blank_cal is not None and blank_meta is not None:
        import grid_pipeline as _gp
        scan_pages = {}
        for page_idx in sorted(blank_meta.keys()):
            try:
                scan_pages[page_idx] = pdf_page_to_image(
                    pdf_path, page_idx + page_offset, dpi,
                    scan_rotation=scan_rotation, color=True)
            except Exception as e:
                print(f"    [WARN] Could not load page {page_idx}: {e}")
        winners, conf = _gp.read_survey(scan_pages, fields, blank_cal,
                                        blank_meta, group_config or {},
                                        annotate_dir=annotate_dir, label=label)
        review = {}
        for field in fields:
            name = field["name"]
            if field.get("type") != "checkbox":
                results[name] = ""
                continue
            prefix = name.split(": ", 1)[0] if ": " in name else name
            option = name.split(": ", 1)[1] if ": " in name else name
            results[name] = "Yes" if winners.get(prefix) == option else "No"
            review[prefix] = bool(conf.get(prefix, True))
        results["__review__"] = review   # {group: is_confident}
        return results

    # Determine which pages we need
    page_indices = sorted({f.get("page", 0) for f in fields
                           if f.get("type", "text") == "checkbox"})

    scan_pages_color = {}   # page_idx → color BGR image
    scan_pages_gray  = {}   # page_idx → grayscale image
    scan_use_color   = {}   # page_idx → bool
    corrections      = {}   # page_idx → affine correction tuple

    for page_idx in page_indices:
        actual_page = page_idx + page_offset
        try:
            # Always load color for detection; derive grayscale as fallback
            color_img = pdf_page_to_image(pdf_path, actual_page, dpi,
                                          scan_rotation=scan_rotation,
                                          color=True)
            gray_img  = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            scan_pages_color[page_idx] = color_img
            scan_pages_gray[page_idx]  = gray_img

            use_color = is_color_scan(color_img)
            scan_use_color[page_idx] = use_color
            mode = "COLOR (orange circles)" if use_color else "GRAYSCALE (fallback)"
            print(f"    [PAGE {page_idx}] rendered {color_img.shape[1]}×{color_img.shape[0]} px"
                  f" — mode: {mode}")

            page_fields = [f for f in fields
                           if f.get("page", 0) == page_idx
                           and f.get("type") == "checkbox"]

            # Registration correction:
            # Grayscale full-page Hough gives the most accurate global
            # scale+translate fit — ~1100 circles spread across the entire page
            # provide much stronger spatial coverage than the ~90 orange answer
            # bubbles (which are clustered in answer areas).  The 4-param
            # (per-axis scale + translate) correction from grayscale is used for
            # all expected-position computation.
            #
            # For color scans, calibrate_page_grid() is also called to log the
            # tilt angle and scale derived from the pre-printed orange circles.
            # The orange-circle snap in build_color_grid() then locks every answer
            # position to the exact printed bubble center.
            corrections[page_idx] = estimate_page_correction(gray_img, page_fields)
            if use_color:
                calibrate_page_grid(color_img, page_fields, color_img=color_img)

        except Exception as e:
            print(f"    [WARN] Could not load page {page_idx}: {e}")
            scan_pages_color[page_idx] = None
            scan_pages_gray[page_idx]  = None
            scan_use_color[page_idx]   = False
            corrections[page_idx]      = (1.0, 0.0, 1.0, 0.0)

    # Group checkbox fields by question prefix
    from collections import defaultdict
    groups = defaultdict(list)

    for field in fields:
        name  = field["name"]
        ftype = field.get("type", "text")
        if ftype != "checkbox":
            results[name] = ""
            continue
        prefix = name.split(": ", 1)[0] if ": " in name else name
        groups[prefix].append(field)

    # Detect per group
    for prefix, group_fields in groups.items():
        page_idx   = group_fields[0].get("page", 0)
        color_img  = scan_pages_color.get(page_idx)
        gray_img   = scan_pages_gray.get(page_idx)
        use_color  = scan_use_color.get(page_idx, False)
        corr       = corrections.get(page_idx)

        if color_img is None:
            for f in group_fields:
                results[f["name"]] = "ERROR"
            continue

        try:
            gc      = (group_config or {}).get(prefix, {})
            max_sel = gc.get("max_selections", None)

            if use_color:
                # ── Color pipeline ──────────────────────────────────────────
                if len(group_fields) > 1:
                    detections = read_checkbox_group_color(
                        color_img, group_fields,
                        correction=corr,
                        max_selections=max_sel)
                else:
                    f      = group_fields[0]
                    h, w   = color_img.shape[:2]
                    ex     = int((f["x"] + f["w"] / 2) * w)
                    ey     = int((f["y"] + f["h"] / 2) * h)
                    if corr:
                        ax, bx, ay, by_ = corr
                        ex = int(ax * ex + bx); ey = int(ay * ey + by_)
                    bubbles = detect_orange_bubbles(color_img)
                    cx, cy, r = snap_to_orange_bubble(ex, ey, bubbles)
                    score = _color_ink_density(color_img, cx, cy, r)
                    val   = "Yes" if score >= 0.05 else "No"
                    print(f"      [COLOR-STANDALONE] {f['name']}: ink={score:.3f} → {val}")
                    detections = {f["name"]: val}
            else:
                # ── Grayscale pipeline (fallback for B&W scans) ─────────────
                if len(group_fields) > 1:
                    detections = read_checkbox_group(gray_img, group_fields,
                                                     correction=corr,
                                                     max_selections=max_sel)
                else:
                    f   = group_fields[0]
                    val = read_standalone_checkbox(gray_img, f, correction=corr)
                    detections = {f["name"]: val}

            for name, val in detections.items():
                results[name] = val

        except Exception as e:
            for f in group_fields:
                print(f"    ✗ {f['name']}: ERROR — {e}")
                results[f["name"]] = "ERROR"

    # Optional debug images
    if debug_dir:
        for page_idx in page_indices:
            gray_img = scan_pages_gray.get(page_idx)
            if gray_img is None:
                continue
            corr = corrections.get(page_idx)
            out_path = os.path.join(debug_dir, f"{label}_page{page_idx}.png")
            save_debug_page(gray_img, fields, corr, out_path, label, page_idx)

    return results


# ─────────────────────────────────────────────
# Debug visualisation
# ─────────────────────────────────────────────

def save_debug_page(scan_img: np.ndarray, fields: list, correction: tuple,
                    output_path: str, survey_label: str, page_idx: int):
    """
    Save a colour-annotated version of one scan page showing:
      • Green circle  = detected bubble position (after Hough / affine)
      • Red dot       = expected position from config (before correction)
      • Cyan text     = field name + ink density

    Useful for diagnosing why certain bubbles are not found correctly.
    """
    # Convert grayscale → colour
    vis = cv2.cvtColor(scan_img, cv2.COLOR_GRAY2BGR)
    h, w = scan_img.shape[:2]

    cb_fields = [f for f in fields if f.get("type") == "checkbox"
                 and f.get("page", 0) == page_idx]

    for f in cb_fields:
        # Raw expected center
        ex_raw = int((f["x"] + f["w"] / 2) * w)
        ey_raw = int((f["y"] + f["h"] / 2) * h)

        # Located center
        cx, cy, r = _locate_bubble(scan_img, f, correction=correction)
        density    = _interior_ink_density(scan_img, cx, cy, r)

        # Draw
        cv2.circle(vis, (ex_raw, ey_raw), 4, (0, 0, 255), -1)          # red dot = expected
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)                    # green circle = located
        label = f["name"].split(": ")[-1][:12]
        cv2.putText(vis, f"{label} {density:.2f}", (cx - 20, cy - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"    [DEBUG] Saved: {output_path}")


# ─────────────────────────────────────────────
# Collapse checkbox groups → answer strings
# ─────────────────────────────────────────────

def collapse_checkbox_groups(fields: list, results_list: list):
    """
    Transforms raw per-field Yes/No results into human-readable answer strings.

    "Q3 - Came From: Home" / ": Work" / ": School"
    → display row "Q3 - Came From", value "Home" (or "Home, Work" if both ticked)

    Returns display_rows (ordered list of row metadata), display_results (list of dicts).
    """
    seen        = {}
    display_rows = []

    for field in fields:
        name  = field["name"]
        ftype = field.get("type", "text")

        if ftype == "checkbox" and ": " in name:
            prefix, option = name.split(": ", 1)
            if prefix not in seen:
                seen[prefix] = len(display_rows)
                display_rows.append({
                    "name":     prefix,
                    "row_type": "option_group",
                    "sources":  [(name, option)],
                })
            else:
                display_rows[seen[prefix]]["sources"].append((name, option))
        else:
            seen[name] = len(display_rows)
            display_rows.append({
                "name":     name,
                "row_type": ftype,
                "sources":  [(name, None)],
            })

    display_results = []
    for result in results_list:
        dr = {"Survey": result["Survey"]}
        for row in display_rows:
            if row["row_type"] == "option_group":
                checked = [opt for orig, opt in row["sources"]
                           if result.get(orig) == "Yes"]
                dr[row["name"]] = ", ".join(checked)
            else:
                orig = row["sources"][0][0]
                dr[row["name"]] = result.get(orig, "")
        display_results.append(dr)

    return display_rows, display_results


# ─────────────────────────────────────────────
# Excel output — transposed layout
# ─────────────────────────────────────────────

def save_transposed_excel(all_results: list, fields: list, output_path: str):
    """
    Write results as a transposed table:
      Row 1   : header  ("Question" | Survey1 | Survey2 | …)
      Rows 2+ : one row per question / field group
      Columns B+ : one column per survey
    """
    display_rows, display_results = collapse_checkbox_groups(fields, all_results)

    # Show only answerable bubble questions — one row per question. The many
    # handwritten text fields (route number, addresses, "specify" boxes) are not
    # read by this pipeline and would otherwise appear as empty clutter rows.
    keep_types = {"option_group", "checkbox"}
    display_rows = [r for r in display_rows if r["row_type"] in keep_types]

    wb = Workbook()
    ws = wb.active
    ws.title = "Survey Results"

    thin      = Side(style="thin", color="CCCCCC")
    border    = Border(left=thin, right=thin, top=thin, bottom=thin)

    hdr_font  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    hdr_fill  = PatternFill("solid", fgColor="1F3864")
    hdr_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    q_font    = Font(name="Arial", size=10)
    val_font  = Font(name="Arial", size=10)
    val_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    q_align   = Alignment(vertical="center", wrap_text=True)

    yes_fill  = PatternFill("solid", fgColor="C6EFCE")
    no_fill   = PatternFill("solid", fgColor="FFCDD2")
    yes_font  = Font(name="Arial", size=10, color="276221")
    no_font   = Font(name="Arial", size=10, color="9C0006")

    sel_fill  = PatternFill("solid", fgColor="E2EFDA")
    sel_font  = Font(name="Arial", size=10, color="375623")

    alt_fill  = PatternFill("solid", fgColor="EEF2FF")
    plain_fill = PatternFill("solid", fgColor="FFFFFF")

    survey_labels = [r["Survey"] for r in display_results]

    # confidence lookup: (survey_index, group_name) -> is_confident
    review_lookup = {}
    for idx, res in enumerate(all_results):
        for grp, isconf in (res.get("__review__") or {}).items():
            review_lookup[(idx, grp)] = isconf
    review_fill = PatternFill("solid", fgColor="FFE08A")   # amber = review
    review_font = Font(name="Arial", size=10, color="7A4F01")
    needs_review = []   # (survey, question, answer)

    def hdr_cell(r, c, val):
        cell = ws.cell(r, c, val)
        cell.font, cell.fill, cell.alignment, cell.border = (
            hdr_font, hdr_fill, hdr_align, border)

    hdr_cell(1, 1, "Question")
    for col_idx, label in enumerate(survey_labels, start=2):
        hdr_cell(1, col_idx, label)

    for row_idx, row_meta in enumerate(display_rows, start=2):
        fname    = row_meta["name"]
        row_type = row_meta["row_type"]
        fill_bg  = alt_fill if row_idx % 2 == 0 else plain_fill

        qc = ws.cell(row_idx, 1, fname)
        qc.font, qc.fill, qc.alignment, qc.border = q_font, fill_bg, q_align, border

        for col_idx, survey_data in enumerate(display_results, start=2):
            val     = survey_data.get(fname, "")
            display = "" if val is None else str(val)
            c       = ws.cell(row_idx, col_idx, display)
            c.font, c.alignment, c.border = val_font, val_align, border

            if row_type == "option_group":
                if display:
                    c.fill, c.font = sel_fill, sel_font
                else:
                    c.fill = fill_bg
            elif row_type == "checkbox":
                if display == "Yes":
                    c.fill, c.font = yes_fill, yes_font
                elif display == "No":
                    c.fill, c.font = no_fill, no_font
                else:
                    c.fill = fill_bg
            else:
                c.fill = fill_bg

            # Flag low-confidence answers (amber) so the user reviews only these.
            if review_lookup.get((col_idx - 2, fname)) is False:
                c.fill = review_fill
                c.font = review_font
                needs_review.append(
                    (survey_data.get("Survey", ""), fname, display or "(blank)"))

    ws.column_dimensions["A"].width = 44
    for col_idx in range(2, len(survey_labels) + 2):
        ws.column_dimensions[get_column_letter(col_idx)].width = 30
    ws.row_dimensions[1].height = 40
    for r in range(2, len(display_rows) + 2):
        ws.row_dimensions[r].height = 20

    ws.freeze_panes = "B2"

    # ── "Needs Review" sheet — SAME grid layout as the results sheet, but only
    #    the low-confidence cells are filled in (confident cells left blank) so
    #    you can scan the same table and check only what's flagged. ──
    if needs_review:
        rv = wb.create_sheet("Needs Review")
        rv.cell(1, 1, "Question").font = hdr_font
        rv.cell(1, 1).fill = hdr_fill; rv.cell(1, 1).alignment = hdr_align
        rv.cell(1, 1).border = border
        for c_idx, label in enumerate(survey_labels, start=2):
            cell = rv.cell(1, c_idx, label)
            cell.font, cell.fill, cell.alignment, cell.border = (
                hdr_font, hdr_fill, hdr_align, border)

        for row_idx, row_meta in enumerate(display_rows, start=2):
            fname = row_meta["name"]
            fill_bg = alt_fill if row_idx % 2 == 0 else plain_fill
            qc = rv.cell(row_idx, 1, fname)
            qc.font, qc.fill, qc.alignment, qc.border = (
                q_font, fill_bg, q_align, border)
            for col_idx, survey_data in enumerate(display_results, start=2):
                c = rv.cell(row_idx, col_idx, "")
                c.alignment, c.border = val_align, border
                if review_lookup.get((col_idx - 2, fname)) is False:
                    val = survey_data.get(fname, "")
                    c.value = "" if val in (None, "") else str(val)
                    if c.value == "":
                        c.value = "(blank)"
                    c.fill, c.font = review_fill, review_font
                else:
                    c.fill = fill_bg     # confident -> leave empty

        rv.column_dimensions["A"].width = 44
        for col_idx in range(2, len(survey_labels) + 2):
            rv.column_dimensions[get_column_letter(col_idx)].width = 30
        rv.row_dimensions[1].height = 40
        rv.freeze_panes = "B2"
        total = len(display_rows) * len(survey_labels)
        print(f"[INFO] {len(needs_review)} of {total} answers flagged for review "
              f"({100*len(needs_review)/max(1,total):.0f}%) — see 'Needs Review' tab")

    wb.save(output_path)


# ─────────────────────────────────────────────
# Main processing pipeline
# ─────────────────────────────────────────────

def process_forms(template_pdf, scans_folder: str, config: dict,
                  output_xlsx: str = "survey_results.xlsx", dpi: int = 300,
                  scan_rotation: int = 90, debug_dir: str = None,
                  skip_pages: int = 0, blank_pdf: str = None,
                  annotate_dir: str = None,
                  precomputed_cal=None):
    """
    Main pipeline.

    template_pdf   : optional blank form PDF.  Used ONLY to set the target
                     image dimensions for consistent coordinate mapping.
                     Detection is purely ink-based, not template-comparison.
    scan_rotation  : clockwise degrees to rotate scanned pages before
                     processing.  Default 90 corrects NJ Transit scans
                     that were placed 90° CCW in the scanner.
    """
    fields         = config["fields"]
    pages_per_form = config.get("pages_per_form", 1)
    multi_mode     = config.get("multi_survey_pdf", False)
    group_config   = config.get("groups", {})

    print(f"[INFO] Mode          : {'multi-survey PDF' if multi_mode else 'one PDF per survey'}")
    print(f"[INFO] Pages/survey  : {pages_per_form}")
    print(f"[INFO] DPI           : {dpi}")
    print(f"[INFO] Scan rotation : {scan_rotation}° CW")
    print(f"[INFO] Template      : {template_pdf or '(none)'}")

    template_pages = []
    if template_pdf and os.path.isfile(template_pdf):
        template_pages = [pdf_page_to_image(template_pdf, p, dpi, scan_rotation=0)
                          for p in range(pages_per_form)]
        for i, tp in enumerate(template_pages):
            print(f"[INFO] Template page {i}: {tp.shape[1]}×{tp.shape[0]} px")
    else:
        print("[INFO] No template — using native scan dimensions")

    reader = TextReader()   # lazy-loaded only if text fields exist

    # Calibrate the bubble grid once from the blank template (if provided).
    blank_cal = blank_meta = None
    if precomputed_cal is not None:
        blank_cal, blank_meta = precomputed_cal      # reuse across many files
    elif blank_pdf:
        if not os.path.isfile(blank_pdf):
            print(f"[ERROR] Blank template not found: {blank_pdf}")
            sys.exit(1)
        import grid_pipeline as _gp
        print(f"[INFO] Blank template: {blank_pdf} — calibrating bubble grid…")
        # Auto-load hand-verified overrides for groups the auto-calibration gets
        # wrong (looked for next to the config / in the working directory).
        ov_path = os.path.join(os.path.dirname(os.path.abspath(blank_pdf)),
                               "calibration_overrides.json")
        if not os.path.isfile(ov_path):
            ov_path = "calibration_overrides.json"
        overrides = _gp.load_overrides(ov_path)
        if overrides:
            print(f"[INFO] Loaded {len(overrides)} calibration overrides "
                  f"from {ov_path}")
        blank_cal, blank_meta = _gp.calibrate_blank(blank_pdf, fields,
                                                    scan_rotation=scan_rotation,
                                                    overrides=overrides)
        n_cal = sum(1 for v in blank_cal.values() if v)
        print(f"[INFO] Calibrated {n_cal} bubble positions from blank")

    scan_path = Path(scans_folder)
    if scan_path.is_file() and scan_path.suffix.lower() == ".pdf":
        scan_files = [scan_path]                      # single PDF
    else:
        scan_files = sorted(scan_path.glob("*.pdf"))  # folder of PDFs
    if not scan_files:
        print(f"[ERROR] No PDF files found in '{scans_folder}'")
        sys.exit(1)

    all_results = []
    survey_iter = list(iter_surveys(scan_files, pages_per_form, multi_mode,
                                    dpi, scan_rotation, skip_pages=skip_pages))
    print(f"[INFO] Found {len(survey_iter)} survey(s) to process\n")

    for idx, (label, pdf_path, page_offset) in enumerate(survey_iter, start=1):
        print(f"[{idx}/{len(survey_iter)}] Survey: {label}  "
              f"(file: {Path(pdf_path).name}, page offset: {page_offset})")
        result = process_one_survey(label, pdf_path, page_offset,
                                    template_pages, fields, reader, dpi,
                                    scan_rotation=scan_rotation,
                                    debug_dir=debug_dir,
                                    group_config=group_config,
                                    blank_cal=blank_cal, blank_meta=blank_meta,
                                    annotate_dir=annotate_dir)
        all_results.append(result)

    display_rows, _ = collapse_checkbox_groups(fields, all_results)
    save_transposed_excel(all_results, fields, output_xlsx)
    print(f"\n[DONE] {len(all_results)} survey(s) → {output_xlsx}")
    print(f"       Layout: {len(display_rows)} question rows × {len(all_results)} surveys")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract answers from scanned survey PDFs → transposed Excel"
    )
    parser.add_argument("--template", default=None,
                        help="(Optional) Blank form PDF — used only for target image "
                             "dimensions, not for detection comparison")
    parser.add_argument("--scans",    required=True,
                        help="Folder containing scanned survey PDF(s)")
    parser.add_argument("--config",   default="form_config.json",
                        help="Field config JSON (default: form_config.json)")
    parser.add_argument("--output",   default="survey_results.xlsx",
                        help="Output Excel file (default: survey_results.xlsx)")
    parser.add_argument("--dpi",      type=int, default=300,
                        help="Rendering DPI (default: 300)")
    parser.add_argument("--scan-rotation", type=int, default=90,
                        dest="scan_rotation",
                        help="Degrees to rotate scan pages clockwise before "
                             "processing (default: 90 — NJ Transit scans are "
                             "placed 90° CCW in the scanner). Use 0 if scans "
                             "are already in the correct landscape orientation.")
    parser.add_argument("--debug-dir", default=None, dest="debug_dir",
                        help="(Optional) Folder to save annotated page images "
                             "showing detected circle positions and ink densities. "
                             "Useful for diagnosing detection errors. "
                             "E.g.: --debug-dir debug_images/")
    parser.add_argument("--skip-pages", type=int, default=0, dest="skip_pages",
                        help="Number of leading pages to ignore in each PDF "
                             "before the first survey (e.g. 1 to skip a cover/"
                             "instruction page). Default: 0.")
    parser.add_argument("--blank", default=None, dest="blank",
                        help="(Recommended) Blank scanned form PDF in the SAME "
                             "format as the filled scans. Calibrates the bubble "
                             "grid once and registers each form onto it — far "
                             "more accurate than fixed coordinates for forms cut "
                             "to slightly different lengths.")
    parser.add_argument("--annotate", default=None, dest="annotate",
                        help="(Optional) Folder to save one overlay image per "
                             "survey page showing which circle the code read for "
                             "each option (green=selected, red=considered). Use "
                             "it to eyeball alignment. Requires --blank.")
    args = parser.parse_args()

    if args.template and not os.path.isfile(args.template):
        print(f"[ERROR] Template not found: {args.template}")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)
    if not (os.path.isdir(args.scans) or
            (os.path.isfile(args.scans) and args.scans.lower().endswith(".pdf"))):
        print(f"[ERROR] Scans path not found (folder or .pdf): {args.scans}")
        sys.exit(1)

    config = load_config(args.config)
    process_forms(args.template, args.scans, config, args.output,
                  args.dpi, scan_rotation=args.scan_rotation,
                  debug_dir=args.debug_dir, skip_pages=args.skip_pages,
                  blank_pdf=args.blank, annotate_dir=args.annotate)


if __name__ == "__main__":
    main()
