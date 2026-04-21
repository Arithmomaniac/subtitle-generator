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

The system uses empirical popularity scoring from multiple data sources (SPL,
Canadian libraries, Goodreads, Open Library) blended with corpus frequency.
The `pop_*` parameters have been partially tuned across two rounds.

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
  control the composite formula (takes effect when `populate_popularity.py` is re-run)
- `pop_exponent` applies power-law scaling to raw signals before combining

### Scale and thresholds (recalibrated)

Thresholds were recalibrated from percentile analysis of the blended score
distribution (using `pop_classification_blend=0.9`):
- `accessibility_threshold_pop`: 0.6021 (was 1.0) — 60th percentile of blended scores
- `accessibility_threshold_mainstream`: 0.301 (was 0.5) — 30th percentile
- `tier_center_pop`: 0.78, `tier_center_mainstream`: 0.3, `tier_center_niche`: 0.16
- `tone_target_*` values now match tier centers (pop=0.78, mainstream=0.3, niche=0.16)

### Current accuracy and next steps

- **Pop tier generation accuracy is still low (~10%).** The `weighted_sample_spread` at 0.35
  is too wide for the compressed blended scale — fillers that should be pop-only leak into
  mainstream and vice versa. Narrowing spread is the **highest priority** next tuning target.
- Niche and mainstream accuracy are reasonable.
- `pop_weight_gr`, `pop_weight_nyt`, `pop_weight_library` are new and unexplored — they
  control how Goodreads, NYT, and other library signals blend into the composite score.

### Exploration history

Initial tuning round (April 2026) explored all `pop_*` params. Key findings:
- `pop_tone_blend=0.5` beat 1.0 — blending freq+pop works better than pure pop
- `pop_base_weight_blend`: both 0.25 and 0.75 hurt from 0.5 — 0.5 is the sweet spot
- Per-slot multipliers: reducing bias on secondary slots helped (list_item=0.8, action_noun=0.9),
  but changing of_object in either direction (0.9 or 1.2) hurt — leave at 1.0
- `pop_exponent=1.2` slightly helped — more contrast in popularity scores
- `pop_missing_default`: lowering to 0.05 hurt — 0.1 is fine

Second round (May 2026) added `pop_classification_blend` and recalibrated:
- `pop_classification_blend=0.9` — high value drives tiers by actual popularity
- Thresholds auto-calibrated from percentiles of blended score distribution
- Tone targets aligned to calibrated tier centers
- Pop accuracy still low — `weighted_sample_spread` needs narrowing for new scale

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
| `tone_target_pop_*` | 0.3 | 1.5 | 0.78 | Aligned to tier_center_pop. Higher = more common words only. |
| `tone_target_mainstream_*` | 0.1 | 0.8 | 0.3 | Aligned to tier_center_mainstream. Between pop and niche. |
| `tone_target_niche_*` | 0.0 | 0.5 | 0.16 | Aligned to tier_center_niche. Lower = rarer words. |
| `sample_tone_spread` | 0.2 | 1.5 | 0.6 | Tier sampling Gaussian width |
| `tier_center_pop` | 0.4 | 1.5 | 0.78 | Center score for pop tier (calibrated from blended distribution) |
| `tier_center_mainstream` | 0.1 | 0.6 | 0.3 | Center score for mainstream tier (calibrated from blended distribution) |
| `tier_center_niche` | 0.0 | 0.3 | 0.16 | Center score for niche tier (calibrated from blended distribution) |
| `accessibility_threshold_pop` | 0.3 | 1.0 | 0.6 | Score above which subtitle is classified as pop (auto-calibrated from 60th percentile) |
| `accessibility_threshold_mainstream` | 0.1 | 0.6 | 0.3 | Score above which subtitle is classified as mainstream (auto-calibrated from 30th percentile) |
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
| `pop_classification_blend` | 0.0 | 1.0 | 0.9 | Blend for tier classification: 0=log10(1+freq), 1=popularity_score. Also used for tone bias in generation. |
| `pop_weight_gr` | 0.0 | 1.0 | 0.2 | Weight of Goodreads ratings signal in popularity composite |
| `pop_weight_nyt` | 0.0 | 1.0 | 0.1 | Weight of NYT bestseller signal in popularity composite |
| `pop_weight_library` | 0.0 | 1.0 | 0.05 | Weight of other library lists signal in popularity composite |
| `pop_missing_default` | 0.01 | 0.5 | 0.1 | Default popularity_score for fillers with no empirical data |
| `pop_slot_mult_list_item` | 0.5 | 2.0 | 0.8 | Multiplier on tone_target for list items. Lower helped (less pop bias on items). |
| `pop_slot_mult_action_noun` | 0.5 | 2.0 | 0.9 | Multiplier on tone_target for action nouns. Lower helped. |
| `pop_slot_mult_of_object` | 0.5 | 2.0 | 1.0 | Multiplier on tone_target for of-objects. Both 0.9 and 1.2 hurt — leave at 1.0. |

## Priority Order

From tuning history — parameters ranked by impact and exploration status:

1. **`weighted_sample_spread`** — **HIGHEST PRIORITY.** At 0.35, too wide for the compressed
   blended scale (scores range ~0.09–2.32, most fillers below 0.5). Pop generation accuracy
   is only ~10% because pop and mainstream fillers aren't separated enough. Try 0.15–0.25.
2. **`pop_classification_blend`** — set to 0.9. Separates tier classification from tone
   sampling. Biggest structural improvement — classification now tracks actual book popularity.
3. **`pop_slot_mult_action_noun`** — biggest single gain (+0.050), reduced from 1.0→0.9
4. **`pop_tone_blend`** — +0.022, reduced from 1.0→0.5. Now only used for base weights,
   not tone bias (which uses `pop_classification_blend`).
5. **`pop_slot_mult_list_item`** — +0.012, reduced from 1.0→0.8. Less pop bias on items helped.
6. **`pop_exponent`** — +0.003, raised from 1.0→1.2. More score contrast helped slightly.
7. `sample_tone_spread` — never tuned, may interact with popularity differently
8. `pop_weight_gr` / `pop_weight_nyt` / `pop_weight_library` — **unexplored**, new data source
   weights. Changing requires `populate_popularity.py` re-run with new data.
9. `pop_weight_spl` / `pop_weight_ol` — not yet explored (requires DB rebuild)
10. `weighted_sample_bias_floor` — historically impactful, pinned at lower bound (0.05)
11. `accessibility_threshold_*` — now auto-calibrated from percentiles, unlikely to need manual tuning
12. `tone_target_*` — aligned to tier centers. Coordinate with `tier_center_*` if adjusting.
13. `pop_base_weight_blend` — explored both directions from 0.5, both hurt. Stable at 0.5.
14. `pop_slot_mult_of_object` — explored both directions from 1.0, both hurt. Stable at 1.0.
15. `pop_missing_default` — lowering hurt. Stable at 0.1.

## Multi-Source Popularity — NEW

### Current source coverage

| Source | Raw entries | Matched to works | Notes |
|---|---|---|---|
| SPL (Seattle Public Library) | ~81k works | 81k | Primary checkout-based signal |
| Canadian libraries | 243k entries (102k ISBNs) | 12.8k works | Excellent ISBN→work matching |
| Goodreads | 18k entries | 118 works | Low overlap expected — popular fiction ISBNs aren't in academic-heavy subtitle corpus |
| Open Library | ~5.9M editions | ~5.9M | Edition count proxy for popularity |
| NYT bestsellers | — | — | API harness built, needs API key + ~7 days background polling |
| Wikipedia bestsellers | — | — | Scraper script ready, needs full run |
| NYPL/VPL/PLR | 120 entries | — | Small but high-quality curated lists |

### Key observations

- **Canadian library data** provided the best new coverage: 102k ISBNs → 12.8k matched works
  with high confidence. The ISBN-to-work pipeline resolved editions effectively.
- **Goodreads low overlap (118/18k)** is expected and not a data quality issue — most popular
  fiction ISBNs (thrillers, romance) simply don't appear in our academic-heavy subtitle corpus.
  The 118 that do match are high-signal.
- **NYT API** harness is built (`data/nyt_bestsellers/`) but requires an API key and ~7 days
  of rate-limited background polling to collect the full historical bestseller list.
- **Wikipedia bestsellers** scraper is ready (`data/wikipedia_bestsellers/`) but needs a full
  run to extract and match ISBNs.
- New composite weights (`pop_weight_gr=0.2`, `pop_weight_nyt=0.1`, `pop_weight_library=0.05`)
  are set but unexplored — actual impact depends on running `populate_popularity.py` with all
  sources integrated.

## Simplicity Criterion

Per the autoresearch pattern: prefer simpler parameter values when quality is equal.
If a round number (0.5, 1.0) scores within 2% of a non-round number (0.47, 1.03),
keep the round number.
