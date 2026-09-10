# Bubble / Checkbox Accuracy — Findings & Fix Plan

Tested the current pipeline on all 4 surveys in `scans/your_scan.pdf` (300 DPI,
color path) and traced every wrong answer to its cause. Summary: **the problem
is bubble *localization*, not ink thresholds.** The detector frequently measures
ink at the wrong place (a neighbouring bubble or heading text), and winner‑takes‑all
then commits to that noise.

## Evidence

**Localization is off.** On page 0, of 102 option bubbles vs 108 reliably detected
orange circles:

- only **8/102** options land within 12 px of a real bubble,
- only **33/102** within one bubble‑radius (22 px),
- **42/102** are more than 40 px from *any* bubble, so the detector gives up and
  measures ink at the config's expected spot — which sits on **heading text or
  whitespace**.

You can see it in the overlays: green "measured here" circles land on the bold
question numbers "10. / 11. / 13." instead of the orange bubbles. That is why
"Home" wins Q10 on *all four* surveys at ~0.4 — it is reading the dark "10."
heading, not a filled bubble.

**Downstream effect.** Across all 80 single‑select answers: 47 have a top score
below 0.25 (weak/blank) and 39 are near‑ties (gap < 0.06). Winner‑takes‑all with
a 0.05 floor turns these into confident‑looking but wrong answers. Confirmed by
eye that e.g. survey 0007 Q16 and Q18 are **blank**, yet the old pipeline reports
"Company Business" and "Non‑Binary/Gender Fluid."

**Verified mis‑reads the new approach fixes.** Survey 2 Q5 is clearly "Walked Only"
and Q16 is clearly "Work" — the old pipeline reported "Another Bus" and
"Company Business" respectively. The rebuilt localization reads both correctly.

## Root cause (the decisive test)

The config's per‑field coordinates are simply inaccurate. Measured against the
**blank template** `NJTransitSurvey2026_BLANK.pdf` (no scan distortion at all),
config → nearest‑bubble distance is a **median 33 px**, with only 21/102 within
12 px. A page‑level affine fit cannot remove it (residual stays ~24 px), so the
error is **non‑linear / per‑field**, not a global scale/offset. The form was
laid out by estimating coordinates (as the README describes), and those estimates
are ~0.75 of a row‑spacing off — close enough to the ~43 px inter‑row spacing
that the detector aliases onto the wrong row.

Two secondary issues compound it:

1. **Forced winner.** Single‑select groups always emit a "Yes", even when the
   question is blank (min‑ink floor 0.05 is below typical blank‑bubble noise).
2. **Mark style.** Surveys 2–3 use bold pen fills (easy, score 0.6–0.7);
   surveys 1 & 4 use *faint pencil slashes* through the bubble (score ~0.15,
   sometimes ambiguous even to a human). Erosion‑based ink metrics erase the
   thin slash entirely.

## What I changed / prototyped (`grid_detect.py`)

A self‑contained module demonstrating the fix direction:

- **Per‑group local registration** — anchors each question's option set to the
  detected orange bubbles, cutting residual from ~37 px to ~2–10 px on most
  groups. Locks every answer onto a *detected* bubble centre rather than a
  config guess.
- **Slash‑tolerant ink metric** — no erosion; thresholds relative to local paper
  brightness and drops speckle via connected components, so faint diagonal
  pencil marks survive while empty bubbles read ~0.
- **Hybrid blank‑gate** — accepts a winner only if it clears a low absolute floor
  *and* stands out from its siblings; bubble‑less groups (e.g. the handwritten
  "AM/PM" in Q2, which is **not** a bubble question) and unanswered questions
  return blank. Flags low‑confidence answers for review.

This corrects the documented bold‑mark mis‑reads. It does **not** fully solve the
faint‑pencil and grid‑aliasing cases, because those are limited by the inaccurate
config coordinates (a 7‑bubble column can't disambiguate which row is marked when
the anchor is 33 px off).

## Recommended fix (the real solution)

**Re‑author `form_config.json` coordinates from the blank template.** A one‑time
calibration that (a) detects every orange bubble on the blank form and (b) assigns
each to its config option by structure, then writes exact fractional coordinates.
With accurate coordinates (a few px, not 33 px), the localization above becomes
robust and the row‑aliasing disappears. This needs one supporting improvement:
**more complete bubble detection** — the current Hough pass misses whole regions
on the template (Q14, Q16 found 0 bubbles; page 1 found 32 of ~35).

Then, in priority order:

1. Build the template calibration tool → accurate config.
2. Ship the `grid_detect.py` localization + ink metric + blank‑gate.
3. Surface a per‑answer confidence flag; route low‑confidence (faint/ambiguous)
   answers to a quick human review pass instead of guessing.
4. Practical: pen marks read far more reliably than faint pencil — worth noting
   for any future data collection.

## Notes

- The original `survey_ocr.py` is unchanged. `grid_detect.py` is a prototype.
- The built‑in `--debug-dir` overlay draws the *grayscale* path, not the color
  path that actually runs — it is misleading for diagnosis; worth fixing.
- Scratch analysis scripts (`_*.py`) were left in the folder; they can be deleted.
