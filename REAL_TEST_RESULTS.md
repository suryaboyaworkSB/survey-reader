# Real Test Results — 8 surveys, scored against your answer key

## UPDATE 3 — confidence flagging (review workflow)

Tried to push accuracy higher via better ink measurement; it didn't work.
Correct and incorrect picks have nearly identical ink scores (median 0.26 vs
0.28) because the errors are mostly registration shifts (a dark mark on the
WRONG bubble), not weak marks. Centre-weighting and darkness-weighting both
*lowered* accuracy. So 50.6% is the ceiling for the automatic read on these
faint-pencil scans.

Instead, each answer now carries a confidence flag. The output gains a
**"Needs Review"** tab and amber-highlighted cells for everything uncertain.

Measured against ground truth:
- **12 answers are flagged confident → 75% correct** (only clearly-dark marks
  with a wide margin qualify).
- **96% of all errors fall in the "needs review" set** — so reviewing the
  flagged cells catches almost every mistake.
- It flags ~90% of bubble answers, which is the honest reality of faint-pencil
  data: most of it needs a human glance. Reviewing the flagged cells and
  trusting the rest gets you to ~98% effective accuracy.

Practical takeaway: the single biggest real-world lever is **pen instead of
pencil** at collection time — bold marks score 0.5–0.7 and separate cleanly.

## UPDATE 2 — locked calibration with verified overrides → 50.6%

Diagnosed all 87 errors: 53 were "mislocated" (wrong bubble position), 18
"wrong winner" (right area, ink bled to a neighbour), 12 calibration gaps.
The mislocations clustered in a few complex-layout groups that fail on every
form — i.e. one-time calibration errors.

Added a `calibration_overrides.json` mechanism: hand-verified bubble positions
for the groups the auto-calibration gets wrong (so far Q20 and the 2-column
Q24 race question). These are loaded automatically by `survey_ocr.py --blank`.

| Stage | On answered (151) |
|---|---|
| Original code | 20.5% |
| Cut blank + region filter + partial calibration | 43.7% |
| + per-group affine refinement | 45.0% |
| **+ verified overrides (Q20, Q24)** | **48.3%** (50.6% overall) |

### Remaining errors, now split by type
- **Q15** (15-option dense column): a subtle one-row calibration ambiguity —
  the toughest layout; needs careful per-row override.
- **Ink / wrong-winner** (Q8, Q16, ~18 errors): the bubble is located correctly
  but the respondent's mark bleeds into a neighbour, or a faint mark loses. This
  is a *fill-measurement* problem, not calibration — a separate lever.
- **Q2 (AM/PM)**: handwritten, not a bubble question.

### How to add more overrides
Edit `calibration_overrides.json` — `{"Group: Option": [x_fraction, y_fraction]}`
in blank-page coordinates. Values there win over the automatic calibration.

---

## UPDATE — with the cut blank template

After you supplied a blank scanned in the same cut/format as the mail-backs
(`BRNB422005F2960_004268.pdf`), accuracy on answered questions climbed:

| Stage | Overall (160) | On answered (151) |
|---|---|---|
| Original `survey_ocr.py` (your run) | 20.6% | 20.5% |
| Rebuilt + landscape blank | ~38% | ~37% |
| Rebuilt + **cut blank** + region filter + partial calibration | **46.2%** | **43.7%** |

What moved the needle, in order of impact:
1. **Cut blank in matching geometry** — calibrate option→bubble once, then
   register onto every form (global affine absorbs each form's different length).
2. **Survey-region filter** — the back page has a "WIN a pass" promo block with
   ~20 of its own orange bubbles; excluding everything outside the question
   area fixed page-2 questions (+~6 points).
3. **Partial calibration** — assign the bubbles we find even when a faint scan
   drops one, instead of discarding the whole group (recovered Q8, page-2 groups).
4. Slash-tolerant ink metric + blank-gate (decision thresholds already optimal —
   a sweep confirmed current values are best).

Per-form, accuracy tracks registration quality: the cleanest-cut form (survey 6,
3.7px registration) reads ~60%; the most distorted (survey 7, 14px) ~22%.

## Still open (the path to usable accuracy)
- **Dense/duplicated groups** Q15 (15 options), Q24, Q8 still mis-calibrate by a
  row and need a one-time verified fix.
- **Per-form registration** on the most-distorted cuts (a piecewise/local model
  would tighten the 12–17px forms toward the 3px ones).
- Q2 ("AM/PM") is handwritten, not a bubble question — needs separate handling.

Files: `grid_pipeline.py` (calibration + reader), `grid_detect.py`,
`form_config_calibrated.json` (126 locked bubble coordinates).

---

# (Original results, for reference)

Tested on `Mailback 5_12.pdf` (8 surveys, both sides), scored against the
ground-truth answer key you filled in (`NJT_AnswerKey_TEMPLATE.xlsx`):
20 single-select questions × 8 surveys = 160 cells (151 answered, 9 blank).

## Headline numbers

| Pipeline | Overall (160) | On answered (151) |
|---|---|---|
| **Current `survey_ocr.py` (baseline)** | **15.6%** | **15.9%** |
| Rebuilt detector, blank-anchored + tuned | ~38% | ~37% |

The baseline barely works on real data — it is essentially guessing. The
rebuild more than doubles it, but is not yet production-ready (see ceiling
below).

## What the test proved

1. **The problem is localization, not ink thresholds.** The baseline measures
   ink in the wrong place — on neighbouring bubbles or heading text — so the
   answer is wrong before any threshold is applied.

2. **Detection was incomplete on these scans.** The mail-back scans render the
   orange ring faint, so the old detector found only ~60% of the bubbles
   (e.g. 6 of 12 on Q5). Widening the colour range fixed this (now ~11–12/12).

3. **The blank template can be registered onto each scan with ~3px accuracy.**
   This is the key architectural win: detect the bubble grid on the clean blank
   once, then fit a global affine from the blank's bubbles to each scan's
   bubbles (median residual 3.3px). Geometry/aspect differences between the
   blank and the scans are absorbed by that fit. So we can calibrate
   *option → bubble* a single time and map it onto all 8 surveys.

## The remaining ceiling (why it's ~38%, not ~90%)

Two coupled bottlenecks, both one-time/fixable:

1. **Calibration of dense question groups.** The config's coordinates are
   inaccurate by a *median 33px* — about 3/4 of the gap between bubble rows —
   and the error is non-linear (no global transform removes it). For widely
   spaced groups (Q3, Q10, Q11, Q12 …) the auto-calibration locks on perfectly
   (sub-pixel). For tightly packed or duplicated groups (Q5 vs the identical Q8
   list, Q15, Q17, Q18, the small page-2 groups) it grabs a neighbouring row —
   an off-by-one that then poisons every survey.

2. **Page-2 detection + faint pencil.** Page 2 detects ~32 of ~35 bubbles, and
   a few surveys use faint marks that sit near the decision threshold.

## What unlocks high accuracy

The decisive missing piece is a **correct one-time calibration** of all 20
question groups on the blank. That is cleanest with:

1. **A blank PDF in the same scan geometry as the mail-back forms** (you offered
   to provide one). On a clean blank with complete bubbles, the calibration can
   be locked and then *visually verified/corrected* once — after which the 3px
   global registration carries it to every survey.
2. A short verification pass on the ~12 dense groups (one-time, on the clean
   image — no per-survey work).
3. Minor per-scan tuning already in place (snap tolerance, blank-gate).

With an exact calibration, the well-aligned groups already demonstrate the
ceiling is high; the architecture is sound. The blocker is purely the one-time
coordinate map.

## Files

- `grid_detect.py` — bubble detection, localization, slash-tolerant ink metric,
  blank-gate (prototype).
- `grid_pipeline.py` — blank-anchored calibration + global-registration reader.
- `NJT_AnswerKey_TEMPLATE.xlsx` — your filled ground-truth key.
- Baseline run: `baseline.xlsx` (scored 15.6%).
- Original `survey_ocr.py` is unchanged. Scratch `_*.py` / `sweep*.py` scripts
  can be deleted.
