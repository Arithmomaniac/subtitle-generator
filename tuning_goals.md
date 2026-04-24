# Tuning Goals

Human-readable objectives for the autoresearch tuning loop.
The tuning agent reads this file each iteration to guide parameter proposals.

## Quality Goals

### Pop Tone
- Should sound like **airport bookstore bestsellers**: "Race, Power, and the Pursuit of Happiness"
- Fillers should be recognizable, culturally familiar words (CNN, Wall Street, Jesus, Nixon)
- Avoid academic jargon in pop mode — no "Helmontian Chymistry" or "Pragmatic Constructivism"

### Mainstream Tone
- Should sound like **indie bookstore nonfiction**: "Politics, Business, and the Meaning of Community"
- Mix of accessible and slightly elevated vocabulary
- Not as populist as pop, not as obscure as niche

### Niche Tone
- Should sound like **university press titles**: "Vasubandhu, Samizdat, and the Reinvention of Transcendence"
- Uncommon, specialist, or academic fillers are expected and desirable
- Should still be coherent — word salad is never acceptable

## Tone Separation Goals

- **Pop and niche must produce clearly different output.** A human should be able to tell
  which tone was used just by reading 5 subtitles from each.
- Measured by distributional overlap of filler scores (blended log10(1+freq) and popularity_score
  per `pop_tone_blend`).
- Target: tone_separation ≥ 0.5 (at least 50% non-overlapping distributions).

## Popularity Scoring

The system uses empirical popularity scoring from multiple data sources blended
with corpus frequency. Each source is percentile-normalized via `percentile(log1p(raw))`
then combined as a weighted average over available demand sources, with OL edition
count as a confidence-weighted prior (not a peer source). OL-only works are capped
at 0.5 composite. NYT bestseller appearances get a binary boost (0.8 floor).

**Highest-impact discovery:** `pop_classification_blend` (0.9) separates tier
classification from tone sampling. Before this, classification and tone bias
used the same blend — changing `pop_tone_blend` above 0.5 hurt tone quality
but classification needed a higher blend to use actual popularity data.

### How it works

- `pop_classification_blend` controls tier classification score: 0=log10(1+freq), 1=popularity_score.
  Also used for Gaussian bias alignment in tone targeting during generation.
- `pop_tone_blend` controls base weight blending for tone bias: 0=log10(1+freq), 1=popularity_score.
  Now only used for base weights, NOT for tone bias alignment (which uses `pop_classification_blend`).
- `pop_base_weight_blend` controls the sampling base weight: 0=sqrt(freq), 1=sqrt(pop)
- `pop_weight_spl` / `pop_weight_ol` / `pop_weight_gr` / `pop_weight_nyt` / `pop_weight_library`
  control the composite formula (takes effect when `populate-popularity` is re-run)
- `pop_exponent` applies power-law scaling to raw signals before combining
- Filler L1 scores use top-3 mean across source works (not MAX) to prevent single-outlier inflation

### Scale and thresholds (auto-calibrated)

Thresholds are auto-calibrated from percentile analysis of the blended score
distribution (using `pop_classification_blend=0.9`). Current values:
- `accessibility_threshold_pop`: 0.602 — p92, 1,091 fillers in pop tier
- `accessibility_threshold_mainstream`: 0.480 — p64, 4,570 fillers in mainstream tier
- Niche: 7,199 fillers
- `tier_center_pop`: 0.699, `tier_center_mainstream`: 0.480, `tier_center_niche`: 0.301
- `tone_target_*` values match tier centers

### Current accuracy and next steps

- **Pop tier accuracy ~50%** (9/18 in the most recent spot-check batch).
  Mainstream 67%, niche 67%. Overall 60%. The earlier "~10% pop accuracy"
  claim in this doc was an artifact of a broken `_histogram_overlap` metric
  (fixed: now auto-ranges bins instead of hardcoding [0, 3]).
- Tone separation under the current scoring scale tops out around 0.6–0.7
  (Cohen's d ≈ 1.35, so distributions are genuinely well-separated).
- `pop_weight_gr`, `pop_weight_nyt`, `pop_weight_library` are set but unexplored —
  changing them requires re-running `subtitle-gen populate-popularity`.

### Metric calibration history (April 2026)

The `_histogram_overlap` function used by `measure_tone_separation` originally
hardcoded its bin range to [0.0, 3.0]. That fit the original
`log10(1+freq)` score scale. After `pop_classification_blend` was raised
toward 1.0, blended scores moved into [0, 1] and 7 of 10 bins became
unused — inflating measured overlap and depressing reported separation by
~0.13 absolute. Fixed by auto-ranging the histogram from the union of
both samples. Many "discard" decisions during the post-regime tuning
phase were driven by this artifact.

### Exploration history

Initial tuning round (April 2026) explored all `pop_*` params. Key findings:
- `pop_tone_blend=0.5` beat 1.0 — blending freq+pop works better than pure pop
- `pop_base_weight_blend`: both 0.25 and 0.75 hurt from 0.5 — 0.5 is the sweet spot
- Per-slot multipliers: reducing bias on secondary slots helped (list_item=0.8, action_noun=0.9),
  but changing of_object in either direction (0.9 or 1.2) hurt — leave at 1.0
- `pop_exponent=1.2` slightly helped — more contrast in popularity scores
- `pop_missing_default`: lowering to 0.05 hurt — 0.1 is fine

Second round (April 2026) added `pop_classification_blend` and recalibrated:
- `pop_classification_blend=0.9` — high value drives tiers by actual popularity
- Thresholds auto-calibrated from percentiles of blended score distribution
- Tone targets aligned to calibrated tier centers
- Pop accuracy still low — `weighted_sample_spread` needs narrowing for new scale

Third round (April 2026) — multi-source scoring redesign:
- Changed composite from additive sum to weighted average over available demand sources
- Normalized each source via percentile(log1p(x)) for cross-source comparability
- OL treated as confidence-weighted prior, not peer source; OL-only capped at 0.5
- Changed filler L1 aggregation from MAX to top-3 mean
- Added NYT as binary boost (0.8 floor + weeks increment)
- Sources now: SPL (81k works), Goodreads (149k works), Ottawa (31k works), NYT (114 works), OL (5.96M works as prior)
- SPL and Goodreads are uncorrelated (r=0.046) — they measure different things
- Pop fillers with demand backing rose from 63 to 302; OL-only in pop dropped 876 to 254

Fourth round (April 2026) — diagnosed phantom regression:
- After regime changes pushing `pop_classification_blend` to 0.9 then 1.0,
  composite collapsed from ~0.74 to ~0.55 and the autotune loop spent ~16
  iterations in a near-random walk of `discard` decisions.
- Root cause: `_histogram_overlap` had a hardcoded [0, 3] bin range (calibrated
  for the old freq-based score). Under the new percentile-blended scores in [0, 1],
  7 of 10 bins were empty. **Fixed: auto-range the histogram.**
- Sweeps after the fix (3–5 seeds, n=30 per tone): `weighted_sample_spread` is
  insensitive in the 0.05–0.15 range (all give separation ≈ 0.57–0.60).
  `pop_classification_blend=0.5` gives separation ≈ 0.68; blend=0.7 gives ≈ 0.63;
  blend=1.0 (current) gives ≈ 0.59. **Lower blend is better.**
- The agent's recent move to push `pop_classification_blend` from 0.9 → 1.0 was
  wrong; it should be reverted toward 0.5–0.7. **Recalibration of thresholds +
  tier centers is now automatic** — `pop_classification_blend` and
  `pop_missing_default` trigger `calibrate_thresholds` inside the tune loop,
  and the popularity-weight params do too on both fast and full repopulate
  paths. (Prior to this fix the thresholds were stale relative to whatever
  blend the agent had set, which is another reason historical separation
  numbers from rounds 2–3 are not directly comparable.)

## Coherence Constraints

- Every subtitle must be grammatically plausible as a real book subtitle.
- The of-object ("the Z of W") must make semantic sense — "the Pursuit of Happiness" yes,
  "the Pursuit of Refrigerator" no.
- Articles (the/a/an) before of-objects should match corpus usage and sound natural.
- Remixed of-objects may be whimsical but should still parse as English.

## Parameter Bounds

Reasonable ranges for each tunable parameter. The autoresearch loop should not
propose values outside these bounds.

| Parameter | Min | Max | Current | Notes |
|---|---|---|---|---|
| `weighted_sample_spread` | 0.05 | 0.5 | 0.10 | Gaussian width. Insensitive in [0.05, 0.15] under current scoring scale. |
| `weighted_sample_bias_floor` | 0.01 | 0.30 | 0.01 | Minimum weight; pinned at lower bound. |
| `tone_target_pop_*` | 0.4 | 1.0 | 0.699 | Aligned to tier_center_pop (auto-calibrated). Higher = more common words only. |
| `tone_target_mainstream_*` | 0.2 | 0.6 | 0.480 | Aligned to tier_center_mainstream (auto-calibrated). Between pop and niche. |
| `tone_target_niche_*` | 0.1 | 0.5 | 0.301 | Aligned to tier_center_niche (auto-calibrated). Lower = rarer words. |
| `sample_tone_spread` | 0.2 | 1.5 | 0.6 | Tier sampling Gaussian width |
| `tier_center_pop` | 0.4 | 1.0 | 0.699 | Center score for pop tier (auto-calibrated from percentile distribution) |
| `tier_center_mainstream` | 0.2 | 0.6 | 0.480 | Center score for mainstream tier (auto-calibrated) |
| `tier_center_niche` | 0.1 | 0.5 | 0.301 | Center score for niche tier (auto-calibrated) |
| `accessibility_threshold_pop` | 0.3 | 1.0 | 0.602 | Score above which subtitle is classified as pop (auto-calibrated, p92) |
| `accessibility_threshold_mainstream` | 0.2 | 0.6 | 0.480 | Score above which subtitle is classified as mainstream (auto-calibrated, p64) |
| `article_of_min_freq` | 1 | 10 | 1 | Min corpus occurrences before trusting of-object article |
| `article_action_min_freq` | 1 | 10 | 1 | Min corpus occurrences before trusting action article |
| `article_remix_heuristic_threshold` | 0.5 | 1.0 | 0.6 | Min majority fraction for remix head-noun article backoff |
| `remix_reject_double_of` | 0 | 1 | 1 | Reject type-2 remixes where inner prep is "of" (avoids double-of) |
| `pop_weight_spl` | 0.0 | 1.0 | 0.7 | Weight of SPL checkout signal in popularity composite |
| `pop_weight_ol` | 0.0 | 1.0 | 0.3 | Weight of OL edition count signal in popularity composite |
| `pop_weight_freq` | 0.0 | 1.0 | 0.0 | Weight of corpus freq fallback in popularity composite |
| `pop_exponent` | 0.5 | 2.0 | 1.2 | Power-law exponent applied to raw scores before combining |
| `pop_base_weight_blend` | 0.0 | 1.0 | 0.5 | Blend: 0=sqrt(freq) for base weight, 1=sqrt(popularity). Sweet spot at 0.5. |
| `pop_tone_blend` | 0.0 | 1.0 | 0.5 | Blend for base weights only: 0=log10(1+freq), 1=popularity_score. Tone bias now uses pop_classification_blend. |
| `pop_classification_blend` | 0.0 | 1.0 | 1.0 | Blend for tier classification: 0=log10(1+freq), 1=popularity_score. **Currently at 1.0; sweeps suggest 0.5–0.7 gives better separation. Thresholds + tier centers are now auto-recalibrated on every change.** |
| `pop_weight_gr` | 0.0 | 1.0 | 0.2 | Weight of Goodreads ratings signal in popularity composite |
| `pop_weight_nyt` | 0.0 | 1.0 | 0.1 | Weight of NYT bestseller signal in popularity composite |
| `pop_weight_library` | 0.0 | 1.0 | 0.05 | Weight of other library lists signal in popularity composite |
| `pop_missing_default` | 0.01 | 0.5 | 0.1 | Default popularity_score for fillers with no empirical data |
| `pop_slot_mult_list_item` | 0.5 | 2.0 | 0.8 | Multiplier on tone_target for list items. Lower helped (less pop bias on items). |
| `pop_slot_mult_action_noun` | 0.5 | 2.0 | 0.9 | Multiplier on tone_target for action nouns. Lower helped. |
| `pop_slot_mult_of_object` | 0.5 | 2.0 | 1.0 | Multiplier on tone_target for of-objects. Both 0.9 and 1.2 hurt — leave at 1.0. |

## Priority Order

From tuning history — parameters ranked by impact and exploration status:

1. **`pop_classification_blend`** — **HIGHEST PRIORITY.** Currently at 1.0 (pushed
   up from 0.9 in last regime). Sweeps after the histogram-metric fix show
   blend=0.5 gives the best separation (~0.68 vs ~0.59 at 1.0). Recommend
   reverting to 0.5–0.7. Threshold/tier-center recalibration is now automatic
   on every change, so no manual `populate-popularity` step is needed.
2. **`weighted_sample_spread`** — Currently 0.10. Insensitive in [0.05, 0.15];
   all values give separation ~0.57–0.60. Probably close to optimal once
   blend is fixed. Stop tuning unless quality is regressing.
3. **`pop_slot_mult_action_noun`** — historic +0.050 win, reduced 1.0→0.9
4. **`pop_tone_blend`** — +0.022, reduced 1.0→0.5. Used for base weights only.
5. **`pop_slot_mult_list_item`** — +0.012, reduced 1.0→0.8.
6. **`pop_exponent`** — +0.003, raised 1.0→1.2.
7. `sample_tone_spread` — never tuned, may interact with popularity differently
8. `pop_weight_gr` / `pop_weight_nyt` / `pop_weight_library` — **unexplored**, data source
   weights. Changing requires `subtitle-gen populate-popularity` re-run.
9. `pop_weight_spl` / `pop_weight_ol` — not yet explored (requires populate-popularity re-run)
10. `weighted_sample_bias_floor` — historically impactful, pinned at lower bound (0.01)
11. `accessibility_threshold_*` — auto-calibrated on every popularity-related change; do not tune manually
12. `tone_target_*` — aligned to tier centers. Coordinate with `tier_center_*` if adjusting.
13. `pop_base_weight_blend` — explored both directions from 0.5, both hurt. Stable at 0.5.
14. `pop_slot_mult_of_object` — explored both directions from 1.0, both hurt. Stable at 1.0.
15. `pop_missing_default` — lowering hurt. Stable at 0.1.

## Multi-Source Popularity

### Current source coverage

| Source | Raw entries | Matched to works | Notes |
|---|---|---|---|
| SPL (Seattle Public Library) | 994k ISBNs | 81k works | Primary checkout-based signal |
| Goodreads (UCSD Book Graph) | 2.94M ISBNs | 149k works | Global reader engagement (ratings_count) |
| Ottawa Public Library | 125k ISBNs | 31k works | Canadian library holds data |
| NYT bestsellers (partial) | 1.8k ISBNs | 114 works | 2008–2013 nonfiction lists; needs API key to continue |
| Open Library | 5.98M ISBNs | 5.96M works | Edition count prior (not a demand signal); OL-only capped at 0.5 |

### Composite formula

`demand_score = sum(w_i * percentile_i) / sum(w_i)` over observed demand sources only.
`composite = confidence * demand_score + (1-confidence) * ol_percentile`
where confidence scales with number of demand sources.
OL-only works capped at 0.5. NYT appearance floors at 0.8.

### Key observations

- **SPL and Goodreads are uncorrelated** (r=0.046) — they measure different things
  (local library checkouts vs global reader engagement). Both are valuable.
- **Goodreads is fiction-dominated** — most popular fiction ISBNs don't appear in our
  academic-heavy subtitle corpus. The 149k that match are high-signal.
- **Ottawa** provides excellent Canadian library coverage with ISBNs directly from CSV.
- **NYT** is partial (2008–2013, nonfiction only). Full historical pull needs ~3 more days
  of API polling. Run `subtitle-gen download-popularity --sources nyt --nyt-api-key KEY`.
- Weight tuning for `pop_weight_gr`, `pop_weight_nyt`, `pop_weight_library` requires
  re-running `subtitle-gen populate-popularity` after each change.

## Simplicity Criterion

Per the autoresearch pattern: prefer simpler parameter values when quality is equal.
If a round number (0.5, 1.0) scores within 2% of a non-round number (0.47, 1.03),
keep the round number.
