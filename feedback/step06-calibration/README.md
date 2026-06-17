# Step 6 calibration review — human sign-off

This folder holds the **durable, committed evidence** for Step 6's review gate
(tier-slot calibration, issue #39). It mirrors `feedback/step05-smoothing/`.

## What's here

- `decision.json` — the overall **accept / reject / iterate** sign-off on the
  chosen calibration config, with the fitted temperatures and the held-out
  evidence (NLL / ECE / distinctiveness) embedded. This is the gate evidence.

Unlike Step 5 (per-candidate bleed ratings, because embedding bleed is not
measurable), calibration's objective is **quantitative** — held-out likelihood
and reliability. So the review is a single sign-off on the diversity-vs-
distinctiveness trade-off, not per-row judgments.

## How a review round works

1. Sweep calibration granularities (analysis-only, regenerable):
   `uv run subtitle-gen run-calibration-autoresearcher`
2. Build the chosen calibrated artifact + replayable metadata:
   `uv run subtitle-gen build-tier-slot-calibration --granularity per_tier`
3. Read `tier_slot_calibration_report.md`, then record the sign-off:
   `uv run subtitle-gen ingest-calibration-decision --submission <payload.json>`

Nothing in this loop changes the served `tier_slot_filler_distribution_v1.csv`.
