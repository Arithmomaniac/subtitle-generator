# Tuning Goals

Human-readable objectives for the narrow autoresearch tuning loop. The tuning
agent reads this file each iteration to guide generated-output article fallback
proposals.

This loop is **not** responsible for fitting learned tier categorization or
generation policy. Offline book-model workflows own pop/mainstream/niche filler
probabilities. Deterministic calibration owns default tier ratios and deployment
artifacts. Autoresearch tunes only the small article fallback surface that
remains after those evidence functions are fixed.

## Quality Goals

### Pop Tone

- Should sound like airport bookstore bestsellers: "Race, Power, and the
  Pursuit of Happiness".
- Fillers should usually be recognizable, culturally familiar words.
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

### Offline categorization owns tier evidence

These surfaces should be fitted or derived by offline source-book/model
workflows, not tuned by autoresearch from generated-output ratings:

- `slot_filler_model_scores.score_pop`
- `slot_filler_model_scores.score_mainstream`
- `slot_filler_model_scores.score_niche`
- `book_model_tier`
- exported runtime student/rollup selection, such as `export-slot`
- any future source popularity features that feed the offline model

Rationale: these values define the evidence function used to decide whether a
real slot filler belongs in pop, mainstream, or niche generation. They should be
validated against source-book labels, shadow rollups, deployment gates, and human
spot-checks rather than inferred indirectly from ratings of generated subtitles.

### Deterministic generation calibration owns tier policy

These values control generation behavior and should not be tuned by the
autoresearch loop:

- `generation_tier_ratio_pop`
- `generation_tier_ratio_mainstream`
- `generation_tier_ratio_niche`

Rationale: tier ratios define how often default/no-tone generation should target
each market tier. Explicit tier requests use learned model probabilities, not
Gaussian target centers. Ratio changes should be made through a focused
controllability/spot-check workflow rather than inferred from LLM ratings of
individual generated subtitles.

### Autoresearch owns only article fallback behavior

The autoresearch loop may tune exactly these article fallback parameters:

- `article_of_min_freq`
- `article_action_min_freq`
- `article_remix_heuristic_threshold`

The loop should optimize only whether article choices in generated subtitles
sound natural. It should not tune learned tier probabilities, popularity
features, tier ratios, sampling shape, default generation targets, literal-bad
guardrails, or remix policy.

## Current categorization state

- Source-title label infrastructure is available via `classify-source-tiers`.
- Offline book-model artifacts train richer source-book classifiers and distill
  exportable runtime signals.
- Runtime deployment uses `slot_filler_model_scores`, currently installed from
  the selected `export-slot` filler rollup.
- Explicit tier generation uses the requested tier's learned score directly:
  `sqrt(freq) * score_for_requested_tier`.
- Runtime classification uses averaged learned per-slot probabilities when the
  model-score table is present, with deterministic tie-breaking.
- Default/no-tone generation still uses configured tier ratios to select target
  tiers. Those ratios are deployment policy, not autoresearch outputs.
- The literal-bad guardrail is intentionally narrow. It blocks broken artifacts
  and incompatible fragments, but it must not block funny conceptual absurdity
  such as `Emissions Trading`.

## Coherence Constraints

- Every subtitle must be grammatically plausible as a real book subtitle.
- The of-object ("the Z of W") must be syntactically usable. Conceptual stretch
  is allowed and often desirable; literal incompatibility or artifact text is not.
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
