# Tuning Goals

Human-readable objectives for the autoresearch tuning loop. The tuning agent
reads this file each iteration to guide generated-output parameter proposals.

This loop is **not** responsible for fitting popularity evidence. Source
popularity weights, classification blends, tier gates, and tier centers should
come from source-title label calibration/regression. Autoresearch tunes the
generation-side behavior that remains after the popularity evidence function is
fixed.

## Quality Goals

### Pop Tone

- Should sound like airport bookstore bestsellers: "Race, Power, and the
  Pursuit of Happiness".
- Fillers should be recognizable, culturally familiar words.
- Avoid academic jargon in pop mode; no "Helmontian Chymistry" style fillers.

### Mainstream Tone

- Should sound like indie bookstore nonfiction: "Politics, Business, and the
  Meaning of Community".
- Mix accessible and slightly elevated vocabulary.
- Not as populist as pop, not as obscure as niche.

### Niche Tone

- Should sound like university press titles: "Vasubandhu, Samizdat, and the
  Reinvention of Transcendence".
- Uncommon, specialist, or academic fillers are expected and desirable.
- Should still be coherent; word salad is never acceptable.

## Tone Separation Goals

- Pop and niche must produce clearly different output. A human should be able
  to tell which tone was used from a small sample.
- Mainstream should sit between pop and niche, not collapse into either side.
- Measured by distributional overlap of generated filler scores and by LLM
  review of generated output.
- Target: tone separation >= 0.5 while maintaining acceptable quality.

## Division of Responsibility

### Supervised source-title calibration owns popularity evidence

These values should be fitted or derived from source-title labels, not tuned by
autoresearch from generated-output ratings:

- `pop_weight_spl`
- `pop_weight_ol`
- `pop_weight_gr`
- `pop_weight_nyt`
- `pop_weight_library`
- `pop_exponent`
- `pop_base_weight_blend`
- `pop_classification_blend`
- `pop_missing_default`
- `pop_slot_mult_list_item`
- `pop_slot_mult_action_noun`
- `pop_slot_mult_of_object`
- `generation_tier_ratio_pop`
- `generation_tier_ratio_mainstream`
- `generation_tier_ratio_niche`
- `accessibility_threshold_pop`
- `accessibility_threshold_mainstream`
- `tier_pop_min_demand_confidence`
- `tier_pop_min_lower_tail`
- `tier_center_pop`
- `tier_center_mainstream`
- `tier_center_niche`

Rationale: these parameters define or gate the evidence function used to decide
whether a real title is pop, mainstream, or niche. Now that real source-title
labels exist, this evidence function should be calibrated directly against
those labels instead of inferred indirectly from generated subtitle ratings.

### Autoresearch owns only article fallback behavior

The autoresearch loop may tune exactly these article fallback parameters:

- `article_of_min_freq`
- `article_action_min_freq`
- `article_remix_heuristic_threshold`

The loop should optimize only whether article choices in generated subtitles
sound natural. It should not tune popularity, tier ratios, sampling shape,
default generation targets, or remix policy.

## Current Source-Label Calibration State

- Source-title label infrastructure is available via `classify-source-tiers`.
- A seeded random pilot produced 546 labels total: 472 niche, 64 mainstream, 10 pop.
- The pilot is provisional, but useful enough to set an initial supervised
  calibration baseline that can be regenerated as the label set grows.
- `calibrate-tier-gates` fits the current threshold gates against those DB
  labels and derives default generation ratios from the same label
  distribution: pop 0.0183, mainstream 0.1172, niche 0.8645.
- `tier_center_*` values are also offline calibration outputs. They are
  generation-control targets, so they should be recalibrated by a deterministic
  generation-side controllability search rather than computed at application
  launch or raw-medianed from source-label accessibility scores.
- The current committed calibration values are provisional outputs of that
  supervised source-label loop, not autoresearch discoveries.
- The broader calibration loop should own all `pop_*` parameters because they
  feed the popularity evidence and require deterministic table refits against
  actual source data.
- Sampling-shape and default-target behavior is also out of autoresearch for
  now. If it needs work later, use a focused deterministic or human-review
  workflow rather than mixing it into popularity calibration.

## Coherence Constraints

- Every subtitle must be grammatically plausible as a real book subtitle.
- The of-object ("the Z of W") must make semantic sense:
  "the Pursuit of Happiness" yes, "the Pursuit of Refrigerator" no.
- Articles before of-objects should match corpus usage and sound natural.
- Remixed of-objects may be whimsical but should still parse as English.
  Dedicated remix calibration is deferred to a separate workflow.

## Parameter Bounds

Reasonable ranges for each autoresearch-managed parameter. The autoresearch
loop should not propose values outside these bounds.

| Parameter | Min | Max | Current | Notes |
|---|---:|---:|---:|---|
| `article_of_min_freq` | 1 | 10 | 1 | Min corpus occurrences before trusting of-object article. |
| `article_action_min_freq` | 1 | 10 | 1 | Min corpus occurrences before trusting action article. |
| `article_remix_heuristic_threshold` | 0.5 | 1.0 | 0.6 | Min majority fraction for remix head-noun article backoff. |

## Priority Order

1. `article_of_min_freq`: tune only if generated of-object articles sound
   unreliable under sparse corpus evidence.
2. `article_action_min_freq`: tune only if generated action articles sound
   unreliable under sparse corpus evidence.
3. `article_remix_heuristic_threshold`: tune only if remix article backoff
   chooses unnatural articles.

## Simplicity Criterion

Prefer simpler parameter values when quality is equal. If a round number
(`0.5`, `1.0`) scores within 2% of a non-round number (`0.47`, `1.03`), keep
the round number.
