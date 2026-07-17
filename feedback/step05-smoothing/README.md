# Step 5 smoothing review — human feedback

This folder holds the **durable, committed evidence** for Step 5's human-review
gate (semantic smoothing, issue #38). It exists because human judgments are *not*
regenerable, unlike the ablation/metrics/feed under `generated-artifacts/`.

## What's here

- `decision.json` — the overall **accept / reject / iterate** decision plus an
  embedded summary of the per-candidate ratings. This is the gate evidence.

Per-candidate ratings themselves live in the `smoothing_ratings` table of the
working DB (`data/db/subtitles.db`), mirroring the existing `human_ratings`
pattern. The `run_id` in `decision.json` ties the decision to the exact candidate
feed it judged.

## How a review round works

1. Build the candidate feed (analysis-only, regenerable):
   `uv run subtitle-gen build-smoothing-review-feed --variant <name>`
2. Open the **Step 5 smoothing review** canvas and rate the suggestions; saving
   writes ratings to the DB and `decision.json` here (via
   `subtitle-gen ingest-smoothing-ratings`).
3. Feed the verdicts back into the AutoResearcher (they become its objective):
   `uv run subtitle-gen run-semantic-smoothing-autoresearcher`

Nothing in this loop changes the served distribution.
