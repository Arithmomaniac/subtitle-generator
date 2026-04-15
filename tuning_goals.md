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

1. FIRST: Try adjusting `pop_tone_blend` (currently 1.0) and `pop_base_weight_blend`
   (currently 0.5) — these have the most direct impact on output character
2. THEN: Recalibrate `tone_target_*` to match the popularity_score scale
3. THEN: Fine-tune `pop_exponent` to control score distribution shape
4. LAST: Adjust `pop_weight_spl` / `pop_weight_ol` (requires DB rebuild)

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
| `pop_weight_spl` | 0.0 | 1.0 | 0.7 | Weight of SPL checkout signal in popularity composite |
| `pop_weight_ol` | 0.0 | 1.0 | 0.3 | Weight of OL edition count signal in popularity composite |
| `pop_weight_freq` | 0.0 | 1.0 | 0.0 | Weight of corpus freq fallback in popularity composite |
| `pop_exponent` | 0.5 | 2.0 | 1.0 | Power-law exponent applied to raw scores before combining |
| `pop_base_weight_blend` | 0.0 | 1.0 | 0.0 | Blend: 0=sqrt(freq) for base weight, 1=sqrt(popularity) |
| `pop_tone_blend` | 0.0 | 1.0 | 0.0 | Blend: 0=log10(1+freq) for tone bias, 1=popularity_score |
| `pop_missing_default` | 0.01 | 0.5 | 0.1 | Default popularity_score for fillers with no empirical data |
| `pop_slot_mult_list_item` | 0.5 | 2.0 | 1.0 | Multiplier on tone_target for list items (lower = less pop bias on items) |
| `pop_slot_mult_action_noun` | 0.5 | 2.0 | 1.0 | Multiplier on tone_target for action nouns |
| `pop_slot_mult_of_object` | 0.5 | 2.0 | 1.0 | Multiplier on tone_target for of-objects (higher = stronger pop bias on of-obj) |

## Priority Order

**NEW popularity params are the highest priority** — they have never been tuned and
the system was specifically built to enable their exploration.

1. **`pop_tone_blend`** — NEW, never tuned. Controls whether tone uses freq or popularity. Start at 1.0.
2. **`pop_base_weight_blend`** — NEW, never tuned. Controls base sampling weight source.
3. **`pop_slot_mult_of_object`** — NEW. Human feedback shows of-objects dominate tier perception.
   Academic of-objects ("Carceral Expansion", "Dispute Resolution") make pop feel niche.
   Try values > 1.0 to strengthen pop bias specifically on of-objects.
4. **`pop_slot_mult_list_item`** — NEW. Human feedback shows list items are less tier-discriminating
   ("Spies", "Religion" appear familiar across tiers). Try values < 1.0 to reduce pop bias on items.
5. **`tone_target_*`** — Need recalibration for popularity scale. See Popularity Scoring section.
6. **`pop_exponent`** — NEW, controls score distribution shape.
7. `weighted_sample_bias_floor` — 6× change (0.3→0.05), previously highest impact
8. `weighted_sample_spread` — 2× change (0.8→0.4), previously second-highest
9. `tier_center_*`, `accessibility_threshold_*` — may need recalibration for popularity scale

## Simplicity Criterion

Per the autoresearch pattern: prefer simpler parameter values when quality is equal.
If a round number (0.5, 1.0) scores within 2% of a non-round number (0.47, 1.03),
keep the round number.
