"""Improved bubble localization + fill scoring for orange-circle survey forms.

Core idea: config fractional coordinates are only approximate. The pre-printed
orange bubbles are detected reliably, so each question group is registered
LOCALLY (ICP affine) to the detected bubbles, recovering exact centers even
when the global config is structurally off. A blank-gate prevents forcing an
answer onto unanswered questions.
"""
import cv2, numpy as np
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Wide orange range: mail-back scans render the printed ring faint/desaturated,
# so a tight range misses ~half the bubbles. We favour recall here (a few false
# circles are harmless because options are snapped one-to-one onto the grid).
_ORANGE_LO = np.array([3, 40, 40],  dtype=np.uint8)
_ORANGE_HI = np.array([40, 255, 255], dtype=np.uint8)


def detect_orange_bubbles(color_img, min_r=6, max_r=24, min_dist=14, param2=8):
    """Detect pre-printed orange answer circles. Returns Nx3 [cx,cy,r].

    Tuned for recall on faint scans; de-duplicates overlapping detections and
    keeps the larger radius when two collide.
    """
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, _ORANGE_LO, _ORANGE_HI)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    circles = cv2.HoughCircles(orange, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=min_dist, param1=40, param2=param2,
                               minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return np.empty((0, 3), float)
    # sort by radius desc so de-dup keeps the more confident larger circle
    c = sorted(circles[0], key=lambda t: -t[2])
    keep = []
    for x, y, r in c:
        if all((x-kx)**2+(y-ky)**2 > (min_dist*0.8)**2 for kx, ky, _ in keep):
            keep.append((x, y, r))
    return np.array(keep, float)


def _assign(P, B):
    """Nearest assignment of points P to points B (Hungarian if available)."""
    D = np.sqrt(((P[:, None, :] - B[None, :, :2])**2).sum(2))
    if _HAS_SCIPY and len(B) >= len(P):
        ri, ci = linear_sum_assignment(D)
        return ci, D[ri, ci]
    ci = D.argmin(1)
    return ci, D[np.arange(len(P)), ci]


def _fit_affine(P, Q):
    """Least-squares affine A (2x3) mapping P->Q. Regularized to stay near rigid."""
    n = len(P)
    X = np.hstack([P, np.ones((n, 1))])
    A, *_ = np.linalg.lstsq(X, Q, rcond=None)
    return A.T  # 2x3


def _onehot_cost(Pt, B):
    """Mean distance under an optimal one-to-one assignment (Hungarian).

    Using a one-to-one cost (rather than nearest-neighbour) is essential on a
    regular bubble grid: nearest-neighbour cost is minimised equally well by a
    whole-row shift that collapses several options onto the same bubbles, which
    produces off-by-one row errors.  One-to-one assignment penalises that.
    """
    D = np.sqrt(((Pt[:, None, :] - B[None, :, :2])**2).sum(2))
    if _HAS_SCIPY and len(B) >= len(Pt):
        ri, ci = linear_sum_assignment(D)
        return D[ri, ci].mean()
    return D.min(1).mean()


def localize_group(color_img, option_fields, all_bubbles, precorrect=None,
                   region_margin=90, snap_tol=16, default_r=13, search_rng=20):
    """
    Return {field_name: (cx, cy, r, is_real)} locking each option onto its
    detected orange bubble.  is_real=False means no bubble was found near the
    option (predicted position used, or the group has no printed bubbles).

    Two-stage registration:
      1. Apply the page-level correction `precorrect` (ax,bx,ay,by from
         estimate_page_correction) so option centres land within ~half a
         bubble-spacing of the truth.
      2. Refine with a SMALL local translation search (+/- search_rng px) using
         a one-to-one (Hungarian) cost.  The small range cannot jump a whole
         row, and the one-to-one cost cannot collapse options onto a shifted
         grid -- together these eliminate the off-by-one errors that plagued the
         old nearest-snap approach.
    Each option is finally locked to a *detected* bubble centre, so residual
    scale/tilt is absorbed by the snap.
    """
    H, W = color_img.shape[:2]
    P = np.array([[(f["x"]+f["w"]/2)*W, (f["y"]+f["h"]/2)*H]
                  for f in option_fields], float)
    names = [f["name"] for f in option_fields]

    if precorrect is not None:
        ax, bx, ay, by = precorrect
        P = np.column_stack([ax*P[:, 0]+bx, ay*P[:, 1]+by])

    x0, y0 = P.min(0)-region_margin
    x1, y1 = P.max(0)+region_margin
    m = ((all_bubbles[:, 0] > x0) & (all_bubbles[:, 0] < x1) &
         (all_bubbles[:, 1] > y0) & (all_bubbles[:, 1] < y1))
    B = all_bubbles[m]

    if len(B) == 0:
        return {nm: (int(p[0]), int(p[1]), default_r, False)
                for nm, p in zip(names, P)}

    # Small local translation search with one-to-one cost.
    best = None
    for dx in range(-search_rng, search_rng+1, 2):
        for dy in range(-search_rng, search_rng+1, 2):
            t = np.array([dx, dy], float)
            cost = _onehot_cost(P + t, B)
            if best is None or cost < best[0]:
                best = (cost, t)
    Pt = P + best[1]

    ci, d = _assign(Pt, B)
    med_r = float(np.median(B[:, 2]))
    out = {}
    for i, nm in enumerate(names):
        px = int(min(W-1, max(0, Pt[i, 0])))
        py = int(min(H-1, max(0, Pt[i, 1])))
        if d[i] <= snap_tol:
            b = B[int(ci[i])]
            out[nm] = (int(b[0]), int(b[1]), int(round(b[2])), True)
        else:
            out[nm] = (px, py, int(round(med_r)), False)
    return out


def fill_score_diff(diff_img, cx, cy, r, thr=35):
    """Respondent-ink fraction from a template-subtracted difference image.

    diff_img = clip(blank_warped_gray - scan_gray, 0, 255): how much DARKER the
    scan is than the aligned blank at each pixel — i.e. pure respondent ink, with
    the printed ring and text already cancelled out. Because the background is
    removed, even a faint tick stands out and blank bubbles read ~0. Needs an
    accurate registration (ORB homography) for the subtraction to cancel cleanly.
    """
    outer = max(4, int(r) + 3)
    h, w = diff_img.shape[:2]
    x1 = max(0, cx-outer); y1 = max(0, cy-outer)
    x2 = min(w, cx+outer); y2 = min(h, cy+outer)
    roi = diff_img[y1:y2, x1:x2]
    if roi.size < 10:
        return 0.0
    rh, rw = roi.shape[:2]
    lx = min(max(0, cx-x1), rw-1); ly = min(max(0, cy-y1), rh-1)
    mask = np.zeros((rh, rw), np.uint8)
    cv2.circle(mask, (lx, ly), outer, 255, -1)
    ink = ((roi > thr) & (mask > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    inked = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] >= 3)
    total = int((mask > 0).sum())
    return inked/max(1, total)


def fill_score(color_img, cx, cy, r):
    """
    Respondent-ink fraction in/around a bubble. Returns 0..1.

    Real respondents mark bubbles in many ways: solid fills, diagonal pencil
    slashes, and — very commonly — small CHECKMARKS/ticks that cross the bubble
    EDGE rather than filling it. The old version only looked at the inner disc
    (radius-2) and only counted dark pixels, so an edge tick or a light/blue-pen
    tick read ~0 and was missed. This version:
      * measures out to radius+3 so edge-crossing ticks are captured,
      * excludes the printed orange ring (HSV) from numerator AND denominator,
      * counts a pixel as ink if it is DARK *or* COLORED (saturated, non-orange)
        — the colored test catches blue/coloured pen that is light in grayscale,
      * drops isolated speckle, keeping connected strokes.
    Empty bubbles still read ~0 (interior is clean white, ring excluded).
    """
    outer = max(4, int(r) + 3)            # reach past the ring to catch edge ticks
    pad = outer + 4
    h, w = color_img.shape[:2]
    x1 = max(0, cx-pad); y1 = max(0, cy-pad)
    x2 = min(w, cx+pad); y2 = min(h, cy+pad)
    roi = color_img[y1:y2, x1:x2]
    if roi.size < 12:
        return 0.0
    rh, rw = roi.shape[:2]
    lx = min(max(0, cx-x1), rw-1); ly = min(max(0, cy-y1), rh-1)

    disc = np.zeros((rh, rw), np.uint8)
    cv2.circle(disc, (lx, ly), outer, 255, -1)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, _ORANGE_LO, _ORANGE_HI)
    sat = hsv[:, :, 1]

    # measurement region = disc minus the printed orange ring
    region = (disc == 255) & (orange == 0)
    if region.sum() < 4:
        return 0.0
    paper = np.percentile(gray[region], 80)
    thr = min(paper * 0.82, 165)          # dark ink (incl. faint pencil)
    ink = ((gray < thr) | (sat > 60)) & region       # dark OR coloured
    ink = ink.astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    inked = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] >= 3)
    total = int(region.sum())
    return inked/max(1, total)


def read_group(color_img, option_fields, all_bubbles, max_selections=1,
               precorrect=None,
               abs_min=0.10, rel_factor=1.8, margin=0.05, multi_frac=0.55):
    """
    Decide selected option(s). Returns (results_dict, debug_dict).

    Hybrid blank-gate for single-select: the winning bubble is accepted only if
    its ink score clears a low absolute floor (abs_min) AND stands out from the
    group, either by exceeding `rel_factor` x the group median or by beating the
    runner-up by `margin`. This catches faint pencil slashes (low absolute, but
    clearly above blank siblings) while rejecting unanswered questions, where
    all bubbles read near zero. Bubble-less groups (e.g. handwritten AM/PM)
    return all "No". dbg["confident"] flags answers worth trusting vs reviewing.
    """
    grid = localize_group(color_img, option_fields, all_bubbles,
                          precorrect=precorrect)
    names = [f["name"] for f in option_fields]
    real = {nm: grid[nm][3] for nm in names}
    scores = {nm: (fill_score(color_img, *grid[nm][:3]) if real[nm] else 0.0)
              for nm in names}
    dbg = {"scores": scores, "grid": grid, "confident": True}

    if all(not real[nm] for nm in names):
        dbg["confident"] = False
        return {nm: "No" for nm in names}, dbg

    ordered = sorted(scores.values(), reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    median = float(np.median(list(scores.values())))

    if max_selections == 1:
        best = max(scores, key=scores.get)
        stands_out = (top >= rel_factor * median if median > 0.03
                      else True) or (top - second >= margin)
        ok = (top >= abs_min) and stands_out
        # low-confidence: a real but weak/ambiguous winner worth human review
        dbg["confident"] = ok and (top >= 0.25 or (top - second) >= 0.12)
        return ({nm: ("Yes" if (nm == best and ok) else "No")
                 for nm in names}, dbg)

    # multi-select: clear both an absolute floor and a fraction of the top
    thr = max(abs_min, top * multi_frac)
    return ({nm: ("Yes" if scores[nm] >= thr else "No")
             for nm in names}, dbg)
