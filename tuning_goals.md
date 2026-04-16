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

## Popularity Scoring — NEW, MUST EXPLORE

The system recently added empirical popularity scoring (SPL library checkouts +
Open Library edition counts) as an alternative to the old corpus-frequency-only
scoring. **The `pop_*` parameters are new and have NEVER been tuned.** They are
the highest-priority exploration targets because:

1. Corpus frequency correlates weakly (r=0.29–0.50) with actual book popularity
2. Academic words like "gender" (freq=36) are massively overweighted by freq
3. Genuinely popular topics like "Home" (freq=2, pop=1.56) are underweighted

### How it works

- `pop_tone_blend` controls the tone bias score: 0=old log10(1+freq), 1=popularity_score
- `pop_base_weight_blend` controls the sampling base weight: 0=sqrt(freq), 1=sqrt(pop)
- `pop_weight_spl` / `pop_weight_ol` / `pop_weight_freq` control the composite formula
  (only takes effect when `populate_popularity.py` is re-run)
- `pop_exponent` applies power-law scaling to raw signals before combining

### Scale difference (CRITICAL)

The popularity_score scale (0.09–2.32, median ~0.15) is DIFFERENT from the old
log10(1+freq) scale (0–2.5, median ~0.5). When `pop_tone_blend` is high:
- `tone_target_pop_*` values of 1.5 are likely TOO HIGH — most fillers score below 0.5
- Suggested starting points: pop=0.5, mainstream=0.2, niche=0.1
- The `tier_center_*` and `accessibility_threshold_*` params also need recalibration

### Exploration strategy

Initial tuning round (April 2026) explored all `pop_*` params. Key findings:
- `pop_tone_blend=0.5` beat 1.0 — blending freq+pop works better than pure pop
- `pop_base_weight_blend`: both 0.25 and 0.75 hurt from 0.5 — 0.5 is the sweet spot
- Per-slot multipliers: reducing bias on secondary slots helped (list_item=0.8, action_noun=0.9),
  but changing of_object in either direction (0.9 or 1.2) hurt — leave at 1.0
- `pop_exponent=1.2` slightly helped — more contrast in popularity scores
- `pop_missing_default`: lowering to 0.05 hurt — 0.1 is fine
- Tone targets were NOT successfully recalibrated — attempts to move them hurt.
  This may need a coordinated multi-param adjustment rather than single-param hill climbing.

Future tuning should focus on:
1. Coordinated tone target recalibration (move all `tone_target_*` together for the blended scale)
2. `pop_weight_spl` / `pop_weight_ol` exploration (requires `populate_popularity.py` re-run)
3. `sample_tone_spread` — still never changed from 0.6, may interact differently with popularity

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
| `weighted_sample_spread` | 0.1 | 1.0 | 0.35 | Gaussian width; too low = only exact-match fillers, too high = no tone effect |
| `weighted_sample_bias_floor` | 0.01 | 0.30 | 0.05 | Minimum weight; too low = complete suppression, too high = no suppression |
| `tone_target_pop_*` | 0.5 | 2.5 | 1.0–1.5 | Higher = more common words only. May need recalibration for blended scale. |
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
| `pop_weight_spl` | 0.0 | 1.0 | 0.7 | Weight of SPL checkout signal in popularity composite |
| `pop_weight_ol` | 0.0 | 1.0 | 0.3 | Weight of OL edition count signal in popularity composite |
| `pop_weight_freq` | 0.0 | 1.0 | 0.0 | Weight of corpus freq fallback in popularity composite |
| `pop_exponent` | 0.5 | 2.0 | 1.2 | Power-law exponent applied to raw scores before combining |
| `pop_base_weight_blend` | 0.0 | 1.0 | 0.5 | Blend: 0=sqrt(freq) for base weight, 1=sqrt(popularity). Sweet spot at 0.5. |
| `pop_tone_blend` | 0.0 | 1.0 | 0.5 | Blend: 0=log10(1+freq) for tone bias, 1=popularity_score. 0.5 beat 1.0. |
| `pop_missing_default` | 0.01 | 0.5 | 0.1 | Default popularity_score for fillers with no empirical data |
| `pop_slot_mult_list_item` | 0.5 | 2.0 | 0.8 | Multiplier on tone_target for list items. Lower helped (less pop bias on items). |
| `pop_slot_mult_action_noun` | 0.5 | 2.0 | 0.9 | Multiplier on tone_target for action nouns. Lower helped. |
| `pop_slot_mult_of_object` | 0.5 | 2.0 | 1.0 | Multiplier on tone_target for of-objects. Both 0.9 and 1.2 hurt — leave at 1.0. |

## Priority Order

From tuning history — parameters ranked by impact and exploration status:

1. **`pop_slot_mult_action_noun`** — biggest single gain (+0.050), reduced from 1.0→0.9
2. **`pop_tone_blend`** — +0.022, reduced from 1.0→0.5. Blending freq+pop beat pure pop.
3. **`pop_slot_mult_list_item`** — +0.012, reduced from 1.0→0.8. Less pop bias on items helped.
4. **`pop_exponent`** — +0.003, raised from 1.0→1.2. More score contrast helped slightly.
5. `weighted_sample_spread` — historically impactful, currently at 0.35
6. `weighted_sample_bias_floor` — historically impactful, pinned at lower bound (0.05)
7. `tone_target_*` — need coordinated recalibration for blended scale (single-param changes hurt)
8. `sample_tone_spread` — never tuned, may interact with popularity differently
9. `pop_base_weight_blend` — explored both directions from 0.5, both hurt. Stable at 0.5.
10. `pop_slot_mult_of_object` — explored both directions from 1.0, both hurt. Stable at 1.0.
11. `pop_missing_default` — lowering hurt. Stable at 0.1.
12. `pop_weight_spl` / `pop_weight_ol` — not yet explored (requires DB rebuild)

## Simplicity Criterion

Per the autoresearch pattern: prefer simpler parameter values when quality is equal.
If a round number (0.5, 1.0) scores within 2% of a non-round number (0.47, 1.03),
keep the round number.
