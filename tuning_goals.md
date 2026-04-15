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
- Measured by distributional overlap of filler log10(1+freq) scores.
- Target: tone_separation ≥ 0.5 (at least 50% non-overlapping distributions).

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
| `weighted_sample_spread` | 0.1 | 1.0 | 0.4 | Gaussian width; too low = only exact-match fillers, too high = no tone effect |
| `weighted_sample_bias_floor` | 0.01 | 0.30 | 0.05 | Minimum weight; too low = complete suppression, too high = no suppression |
| `tone_target_pop_*` | 0.5 | 2.5 | 1.0–1.5 | Higher = more common words only |
| `tone_target_mainstream_*` | 0.3 | 2.0 | 0.8–1.0 | Should be between pop and niche |
| `tone_target_niche_*` | 0.0 | 1.0 | 0.25 | Lower = rarer words |
| `sample_tone_spread` | 0.2 | 1.5 | 0.6 | Tier sampling Gaussian width |
| `tier_center_pop` | 1.0 | 2.5 | 1.5 | Center score for pop tier |
| `tier_center_mainstream` | 0.3 | 1.2 | 0.75 | Center score for mainstream tier |
| `tier_center_niche` | 0.0 | 0.5 | 0.25 | Center score for niche tier |
| `accessibility_threshold_pop` | 0.7 | 1.5 | 1.0 | Score above which subtitle is classified as pop |
| `accessibility_threshold_mainstream` | 0.2 | 0.8 | 0.5 | Score above which subtitle is classified as mainstream |
| `article_of_min_freq` | 1 | 10 | 1 | Min corpus occurrences before trusting of-object article |
| `article_action_min_freq` | 1 | 10 | 1 | Min corpus occurrences before trusting action article |
| `article_remix_heuristic_threshold` | 0.5 | 1.0 | 0.6 | Min majority fraction for remix head-noun article backoff |
| `remix_reject_double_of` | 0 | 1 | 1 | Reject type-2 remixes where inner prep is "of" (avoids double-of) |

## Priority Order

From git history analysis — parameters that had the biggest impact when tuned:

1. `weighted_sample_bias_floor` — 6× change (0.3→0.05), completely changed suppression behavior
2. `weighted_sample_spread` — 2× change (0.8→0.4), halved the Gaussian width
3. `tone_target_mainstream_*` — +33% shift (0.75→1.0), moved a whole tier
4. `tone_target_pop_of_object` — -33% shift (1.5→1.0), slot-specific data-driven adjustment
5. `sample_tone_spread` — never changed from initial 0.6, may be suboptimal
6. `tier_center_*`, `accessibility_threshold_*` — never changed, lowest priority

## Simplicity Criterion

Per the autoresearch pattern: prefer simpler parameter values when quality is equal.
If a round number (0.5, 1.0) scores within 2% of a non-round number (0.47, 1.03),
keep the round number.
