> Created/edited by GitHub Copilot with human review/feedback by Avi Levin.

# End-to-end pipeline walkthrough

This document explains the pipeline in the order you would run it from a clean
local database to a deployed generator. It is not a history of implementation
changes. Each stage names the command, the state it consumes, the state it
writes, and why that state matters at runtime.

The short version:

```text
raw catalog records
  -> parsed source candidates
  -> strict slot fillers
  -> popularity and remix enrichment
  -> runtime config calibration
  -> labeled source-book training rows
  -> rich Torch teacher
  -> exportable Torch student
  -> filler-level pop/mainstream/niche scores
  -> tier-conditioned filler distributions
  -> tracked CSVs
  -> mini DB
  -> CLI, local web, Azure Functions
```

Runtime generation never invents new filler text. All ML work changes how the
generator weights existing strict fillers for a requested tier.

## Vocabulary

| Term | Meaning |
|---|---|
| Tier | One of `pop`, `mainstream`, or `niche`. A requested tier changes which existing fillers are weighted higher; it is not a separate generator. |
| Source book | A real Library of Congress or Open Library record used as evidence. Source books can provide labels, popularity signals, and slot fillers. |
| Source candidate | A source title/subtitle string that might match the `X, Y, and the Z of W` pattern. |
| Slot | One position in the generated pattern: `list_item`, `action_noun`, or `of_object`. |
| Slot filler | A deduplicated phrase that can fill one slot, such as a list item or final of-object. |
| Strict filler universe | The set of fillers that passed extraction and validation gates. Runtime generation samples only from this universe. |
| Remix | Recombining parts of a multi-word `of_object` to make a new final object while staying close to real source phrasing. |
| Runtime config | Rows in the DB `config` table that tune generation behavior and are exported to `api\data\config.csv`. |
| Teacher model | The rich offline Torch model trained with the broadest feature set. |
| Student model | The exportable Torch model trained to imitate the teacher using durable/export-safe features. Its rollup is evidence for the tier-slot distribution and remains available to the legacy rollback path. |
| Rollup | Aggregation from book-level predictions back onto filler-level scores. |

## 0. Runtime contract

The deployed generator has four responsibilities:

| Responsibility | Runtime state | Effect |
|---|---|---|
| Filler universe | `slot_fillers` | Defines which source-derived fillers are allowed at all. |
| Tier-conditioned generation | `tier_slot_filler_distribution_v1` | Gives normalized `P(filler | tier, slot_type)` over the runtime-eligible strict universe. |
| Legacy rollback | `slot_filler_model_scores` | Retains the old `score_pop`, `score_mainstream`, and `score_niche` path. |
| Generation | `generate.py` | Chooses a tier, then samples each slot directly from its normalized distribution. |
| Literal guardrail | `generate.py` | Blocks broken artifacts without blocking funny absurdity. |

The configured default is `generation_runtime_mode=artifact`. For requested tier
`T` and slot type `S`, generation draws from:

```text
P(filler | T, S)
```

The stored anchored probabilities already include evidence/prior shaping. Runtime
does not multiply by frequency or reapply calibration. An explicit sampling
temperature is the only optional policy transform, and its default is `1.0`.

The old score path remains available with `--runtime legacy`, request body
`{"runtime_mode":"legacy"}`, environment variable
`SUBTITLE_GEN_RUNTIME_MODE=legacy`, or:

```powershell
uv run subtitle-gen set-runtime-default --mode legacy
```

Older databases without `generation_runtime_mode` stay on legacy. A database
configured for artifact mode but missing the artifact temporarily falls back to
legacy; an installed but invalid artifact fails validation rather than silently
changing probability mass.

The guardrail is intentionally narrow. It rejects literal artifacts such as bare
adjective-shaped final objects (`Christian`), typo/acronym artifacts (`Imf`),
run-together initials (`H.G.W.ells`), suffix artifacts (`Con Men, Jr`), and
known standalone artifacts (`Xcalibur`). It does not reject grammatical
nonfiction absurdity such as `Emissions Trading` or `Second Indochina War`.

## 1. Download and extract source records

Run this when rebuilding the corpus from raw third-party data:

```powershell
uv run subtitle-gen download --parts all
uv run subtitle-gen extract
uv run subtitle-gen download-ol
uv run subtitle-gen extract-ol
```

These commands create normalized source rows in the full local SQLite DB.

| Table/field | Meaning |
|---|---|
| `subtitles.title` | Source title display text. |
| `subtitles.subtitle` | Source subtitle display text, when present. |
| `subtitles.candidate_text` | Text that slot extraction will parse. |
| `subtitles.candidate_source` | Whether the candidate came from title or subtitle text. |
| `isbn`, `lccn`, `source_file`, `lang` | Identifier and provenance metadata used later by enrichment and modeling. |

Extraction repairs obvious title/subtitle duplication when it can recover a real
split. Rows that cannot be repaired remain out of the runtime filler path.

## 2. Build the strict filler universe

```powershell
uv run subtitle-gen build-slots
```

`build-slots` searches for source candidates shaped like:

```text
X, Y, and [the/a/an] Z of [the/a/an] W
```

The regex match is only the first pass. `slots.py` then applies stricter gates:

| Gate | Behavior |
|---|---|
| List shape | Accept exactly two or three source list clauses; generated output uses two. |
| List item validation | Reject the whole candidate if any original list item is malformed. |
| Action noun validation | Reject weak, truncated, or implausible action nouns. |
| Of-object validation | Reject malformed final objects and bad starts such as `using`, `with`, or `for`. |
| Source repair | Preserve title-only source display for title-derived candidates. |

Accepted candidates create the runtime universe:

| Output | Meaning |
|---|---|
| `pattern_matches` | Parsed source rows with list items, action noun, of-object, and observed articles. |
| `slot_fillers` | Deduplicated strict fillers by slot type. |
| `slot_filler_sources` | Links from fillers back to source books. |

Everything later, including ML, works inside this strict universe. The book model
does not add new fillers.

## 3. Add popularity evidence

```powershell
uv run subtitle-gen download-popularity
uv run subtitle-gen populate-popularity
```

Popularity is source evidence. It is not the final deployed tier classifier.

| Source | Signal |
|---|---|
| Seattle Public Library | Checkout demand. |
| Goodreads / UCSD Book Graph | Rating count and engagement. |
| NYT | Bestseller appearances. |
| Ottawa / library data | Library demand and holdings. |
| Trove Australia | Library breadth via `holdingsCount`. |
| Open Library edition count | Edition-breadth signal and a learnable source in the constrained popularity block. |

The key filler fields are:

| Column | Meaning |
|---|---|
| `popularity_score` | Source popularity score used as a model/calibration feature. |
| `popularity_level` | Coarse availability of source popularity evidence. |
| `popularity_confidence` | Confidence in the popularity score. |

The source-specific values are first normalized onto comparable percentile-like
scales. For most sources, the normalized value is the percentile rank of
`log10(1 + raw_count)` among works with that source. NYT uses a high-intent
bestseller signal instead:

```text
nyt_score = min(1.0, 0.8 + 0.2 * log10(1 + weeks_on_list) / 2.0)
```

The deployed popularity scalar is a weighted average over available source
signals divided by the total configured source weight. Missing source signals
contribute zero by being absent from the numerator. Open Library is one of the
source signals; it is not a separate runtime backoff term. The source weights
shown as `alpha_i` below are runtime config values: a first clean run uses
defaults, and the normal tuned values are assigned later by
`calibrate-runtime-tier-model`.

$$
\begin{aligned}
\mathrm{pop}(w) &=
  \frac{\sum_{i \in P(w)} \alpha_i x_i(w)}
       {\sum_{j \in A} \alpha_j} \\
\mathrm{pop}(f) &=
  \mathrm{mean}\left(
    \mathrm{top3}\{\mathrm{pop}(w): w \rightarrow f\}
  \right)
\end{aligned}
$$

Here `w` is a source work, `f` is a filler, `P(w)` is the set of available
source signals for that work, `A` is the full configured source set, `x_i` is
the normalized source signal, and `alpha_i` is the configured source weight. The
implementation-shaped version is:

```text
work_popularity_score =
  sum(source_weight_i * normalized_source_i for present sources)
  / sum(all configured source weights)

filler_popularity_score =
  mean(top 3 source-book work_popularity_score values linked to the filler)
```

Fillers without Level 1 source/work popularity data keep popularity missing
instead of using corpus frequency as fake popularity:

```text
filler_popularity_score = NULL
popularity_level = 0
popularity_confidence = 0.0
```

Corpus phrase frequency is still a separate signal. Runtime tiering uses
`frequency_score = log10(1 + freq)`, and offline book features carry max/average
filler frequency-score aggregates separately from max/average filler popularity.

The `pop_weight_*` rows are source weights for this scalar, not tier weights.
They answer "how much should this data source count when computing source/filler
popularity?" Their learned values are listed in the runtime tier-model
calibration section, where they are assigned.

Popularity matters because it gives the models demand evidence, but it is too
blunt to distinguish all pop/mainstream/niche behavior by itself. A popular
source can still provide a niche-shaped filler, and a niche source can still
provide a broadly accessible phrase.

## 4. Precompute runtime-safe remix features

```powershell
uv run subtitle-gen precompute-vectors
```

Runtime remixing cannot load build-time NLP dependencies. This step stores scalar
fields that let serving evaluate remix coherence cheaply.

Remix is only for multi-word `of_object` fillers. The precompute step classifies
which final objects can be decomposed safely:

| Field | Meaning |
|---|---|
| `remix_type` | `type1` for a two- or three-word compound noun phrase; `type2` for a noun phrase containing a preposition. |
| `remix_prep` | The preposition used by a Type 2 remix, such as `in`, `of`, or `for`. |
| `remix_word_count` | Original phrase length. |
| `centroid_dot`, `norm_sq`, `token_count` | Scalar vector approximation fields. |
| `centroid_norm`, `avg_cross_sim_t1`, `avg_cross_sim_t2` | Config constants for runtime similarity approximation. |

Examples:

| Remix class | Original shape | How it recombines |
|---|---|---|
| Type 1 compound | `American Public Identity` | Splits into modifier/head-like parts and recombines with compatible compound parts. |
| Type 2 prepositional | `Knowledge in Late Antiquity` | Preserves the preposition frame and recombines topic/complement-like parts. |
| Not remixable | one-word objects or unsafe compounds | Stays atomic and can still be sampled as a normal `of_object`. |

Generation can then compose candidate parts and reject low-similarity remixes
without spaCy or full vector state.

## 5. Runtime config handoff points

Runtime constants live in the full DB `config` table, then get exported to
`api\data\config.csv` and packaged into the mini DB. In a normal rerun, these
rows may already exist; you do not rerun every tuner just because you are
retraining the book model.

The easiest way to reason about config is to ask two questions: **what reads
this row?** and **who writes it, at what point in the pipeline?**

| Config family | What reads it | Who writes it, and when | If it changes, rerun/export |
|---|---|---|---|
| `article_stats_action_noun`, `article_stats_of_object` | Article selection while generating action nouns and final objects. | `build-slots`, when it derives article counts from observed source titles. | Rebuild slots, then export/build DB. |
| `centroid_norm`, `avg_cross_sim_t1`, `avg_cross_sim_t2` | Runtime remix similarity approximation. | `precompute-vectors`, after slot fillers exist and before runtime remix checks. | Rerun remix precompute, then export/build DB. |
| `remix_calibrated_min_sim`, `remix_calibrated_remix_prob` | CLI/API remix defaults; absent rows fall back to `0.1` min similarity and `0.8` remix probability. | `calibrate-remix`, or the remix phase of `tune`, after generation can produce realistic samples. | Final sample checks, then export/build DB. |
| `article_of_min_freq`, `article_action_min_freq`, `article_remix_heuristic_threshold` | Article backoff/majority heuristics during generation. | Autoresearch tone loop, `tune --phase tone`, after generation is working and sample quality can be rated. | Final sample checks, then export/build DB. |
| `generation_tier_ratio_pop/mainstream/niche` | Default tier selection when no explicit tone is requested. | Defaults from `config.py`, or an explicit config edit before final runtime validation. | Export/build DB after review. |
| `pop_weight_*` | Popularity recomputation for source/filler popularity scores. | Runtime tier-model calibration in section 12, after teacher predictions and selected student rollups exist. | Apply calibration, recompute popularity, rebuild book features, then rerun dependent model steps if needed. |
| `tier_classifier_*` | `compute_tier_evidence()` when classifying assembled subtitles at runtime. | Runtime tier-model calibration in section 12, after the selected rollup exists; otherwise defaults from `config.py`. | Export/build DB after classifier calibration. |
| `pop_missing_default` | Legacy compatibility key; missing popularity is now represented by derived observed/missingness features instead of substituting this value. | Historical config rows or defaults from `config.py`; current runtime tier calibration does not write it. | No direct rerun; leave in place for older DB compatibility. |

Defaults are defined in `config.py`; DB rows override them. Because `export-data`
only exports rows that exist in the DB, default-only values may not appear in the
checked-in `api\data\config.csv`.

### What autoresearch owns

`subtitle-gen tune` is the autoresearch-inspired loop. It does not train the book
model and it does not own popularity or tier-classifier weights. It runs
generation experiments, rates outputs, and edits only generation heuristics:

| Phase | Config it can change | How it works |
|---|---|---|
| Remix | `remix_calibrated_min_sim`, `remix_calibrated_remix_prob` | Grid-search generated samples, rate them, and store the best remix defaults. |
| Tone/article | `article_of_min_freq`, `article_action_min_freq`, `article_remix_heuristic_threshold` | LLM proposes one bounded parameter move, generated samples are scored, and the move is kept only if it improves the score. |

Human feedback enters between runs through `tuning_goals.md`, spot checks, and
reviewed ratings. Popularity and `tier_classifier_*` coefficients stay out of
this loop because they are fitted from source-title labels, teacher predictions,
and rollup tables rather than from generated-output ratings.

Useful commands:

```powershell
# Remix defaults: grid-search min_sim and remix probability on generated samples.
uv sync --extra tune
uv run subtitle-gen calibrate-remix --samples 50

# Runtime tier model: train popularity ratios and final classifier coefficients
# after Distill and Shadow have produced the selected rollup.
uv sync --extra ml
uv run subtitle-gen calibrate-runtime-tier-model `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --rollup generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv

# Apply learned runtime config only after reviewing the report.
uv run subtitle-gen calibrate-runtime-tier-model `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --rollup generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv `
  --apply

# Current autoresearch loop is intentionally narrow: article/remix heuristics.
uv run subtitle-gen tune --phase all --samples 50 --iterations 30

```

The feedback loop is:

```text
current DB state + config
  -> generate or score fixed sample sets
  -> inspect/score outputs
  -> write updated config rows
  -> rerun the affected downstream boundary
  -> export config.csv if the change should deploy
```

For a book-model refresh, use the existing config rows unless you are explicitly
changing a runtime knob. The refresh consumes those rows through generation,
feature extraction, validation, and export; it does not imply retuning them.

## 6. Prepare source-book labels

Source-book labels are the supervision signal for the book model. They are
separate from generated-subtitle feedback.

```powershell
uv sync --extra tune
uv run subtitle-gen classify-source-tiers --dry-run --limit 20
uv run subtitle-gen classify-source-tiers --limit 200 --batch-size 10
uv run subtitle-gen source-tier-distribution
```

The label workflow writes `pattern_matches.llm_market_tier*` in the full local DB
and exports `api\data\source_tier_labels.csv`. Those labels feed offline
training and evaluation; they are not read by the deployed mini DB.

The current labeled set is imbalanced:

| Label | Count | Consequence |
|---|---:|---|
| pop | 48 | Very sparse; pop validation metrics are directional. |
| mainstream | 220 | Enough to learn broad trade/nonfiction texture, but still limited. |
| niche | 1,032 | Dominates the label set and pulls naive models toward niche. |

The Torch trainer uses class and sample weighting so niche does not completely
dominate the loss, but this label distribution still shapes every model.

## 7. Build book-model features

```powershell
uv sync --extra ml
uv run subtitle-gen build-book-features
```

For the richest run, include offline metadata first:

```powershell
uv run subtitle-gen build-book-metadata
uv run subtitle-gen build-book-features `
  --metadata-csv generated-artifacts\book-model\book_metadata.csv
```

`build-book-features` joins source rows, tier labels, filler links, popularity,
optional metadata, and slot-derived text into book-level training artifacts:

| Artifact | Meaning |
|---|---|
| `generated-artifacts\book-model\book_features.csv` | One row per labeled/source candidate with numeric and text features. |
| `generated-artifacts\book-model\book_labels.csv` | Training labels aligned by `pattern_match_id`. |
| `generated-artifacts\book-model\book_feature_label_report.md` | Coverage report for labels and feature availability. |

These artifacts are ignored local outputs. They are meant to be regenerated, not
committed.

## 8. Run the baseline model

```powershell
uv run subtitle-gen train-book-model
```

The baseline is an interpretable calibration model. It uses text, centroid,
provenance, and scalar features to check whether the labels are learnable before
moving to the two-stage Torch path.

Latest rerun:

| Model | Pop | Mainstream | Niche | Role |
|---|---:|---:|---:|---|
| Baseline predictions | 955 | 2,398 | 3,340 | Calibration artifact, not deployed. |

This broad distribution is useful as a sanity check, but the baseline is not the
source of runtime weights.

## 9. Train the rich Torch teacher

```powershell
uv run subtitle-gen train-book-model-torch `
  --output-dir generated-artifacts\book-model\torch-all-spacy `
  --feature-set all `
  --semantic-vectors spacy
```

This is the first gradient-descent model in the deployed path. It is allowed to
use the broadest offline feature set because it is a research/training artifact,
not something production loads.

| Feature family | Examples | Why it matters |
|---|---|---|
| Persisted/source features | title/subtitle text, candidate source, language, ISBN/LCCN/work-key flags | The source book's own wording is the strongest direct tier signal. |
| Popularity features | checkouts, ratings, edition count, bestseller/list/library signals | Useful, but not sufficient: popularity alone cannot separate mainstream from niche academic texture. |
| Slot interaction text | list-item pairs, action/object pairs, slot-frame text | Captures what kind of subtitle grammar produced the filler. |
| Offline metadata | publisher, format, subjects, call numbers, page counts | Helps separate academic, trade, religious, and reference-book neighborhoods. |
| spaCy semantic vector | 300-dimensional document vector over source text | Adds broad semantic similarity that sparse hashed tokens cannot infer. |

Feature-family ablations showed why the teacher combines signals:

| Teacher feature set | Validation exact/macro | Takeaway |
|---|---:|---|
| Persisted/source features | 0.782 / 0.480 | Source shape helps, but misses much of the tier boundary. |
| Popularity features | 0.789 / 0.501 | Popularity helps slightly more than source shape, but is too blunt alone. |
| Slot interactions | 0.805 / 0.469 | Slot grammar is useful for examples, but weak standalone. |
| Metadata | 0.808 / 0.483 | Metadata adds signal, but not enough alone. |
| All rich hashed features | 0.782 / 0.529 | Combining families matters more than any single family. |
| All rich + spaCy vectors | 0.812 / 0.566 | Semantic vectors gave the best macro result and became the teacher candidate. |

Latest validated rerun:

| Model | Pop | Mainstream | Niche | Validation |
|---|---:|---:|---:|---|
| Rich Torch teacher predictions | 264 | 1,867 | 4,562 | 0.782 exact / 0.565 macro |

The teacher is intentionally offline-rich. It is not directly exportable.

### Runtime calibration dependency

Runtime tier-model calibration is described after rollup because the normal
deployed path consumes both the rich teacher predictions and the selected
`export-slot` filler rollup. If you only want a popularity-scalar report, the
command can run earlier without `--rollup`; the final classifier calibration
waits until after distillation and shadow rollup.

## 10. Distill exportable Torch students

```powershell
uv run subtitle-gen distill-book-model `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --output-dir generated-artifacts\book-model\distill-export-current `
  --feature-set export-current

uv run subtitle-gen distill-book-model `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --output-dir generated-artifacts\book-model\distill-export-slot `
  --feature-set export-slot
```

Distillation is the second gradient-descent stage. The student learns to imitate
the rich teacher using only durable/export-safe features.

| Student | Export-safe signal | Pop | Mainstream | Niche | Teacher agreement |
|---|---|---:|---:|---:|---:|
| `export-current` | title/subtitle text plus basic source shape | 261 | 1,895 | 4,537 | 95.8% |
| `export-slot` | `export-current` plus slot aggregate scalars such as source-link count, distinct strict filler count, max filler popularity, and average filler popularity | 392 | 2,071 | 4,230 | 87.4% |

`export-slot` is the selected runtime source. It agrees less tightly with the
teacher than `export-current`, but its slot aggregate features move the export
toward a broader pop/mainstream pool. That is desirable for generation because
the deployed scores are sampling weights over strict fillers, not final claims
about a book's market category.

The Torch file itself is not loaded by the deployed CLI/API. Its output
probabilities are rolled onto fillers and installed as
`slot_filler_model_scores`, then combined with labeled-source confidence and
priors to build the anchored runtime distribution:

```text
student book probabilities
  -> average probabilities over source books linked to each strict filler
  -> slot_filler_model_scores.score_pop/mainstream/niche
  -> empirical-Bayes tier-slot evidence
  -> tier_slot_filler_distribution_v1
```

The score table remains the legacy rollback input. A student refresh affects the
default runtime only after rebuilding and reinstalling the tier-slot artifact.

## 11. Roll book predictions up to fillers

```powershell
uv run subtitle-gen shadow-book-model
```

`shadow-book-model` joins book-level predictions back through
`slot_filler_sources` and averages prediction probabilities for each strict
filler.

| Rollup column | Meaning |
|---|---|
| `avg_score_pop` | Average predicted pop probability across source books that contributed this filler. |
| `avg_score_mainstream` | Average predicted mainstream probability. |
| `avg_score_niche` | Average predicted niche probability. |
| `book_model_tier` | Highest average score after rollup. |
| `source_prediction_count` | Number of source-book predictions behind the filler score. |

The selected `export-slot` rollup now produces this deployed distribution:

| Rollup | Pop | Mainstream | Niche |
|---|---:|---:|---:|
| Prior checked-in export | 463 | 2,867 | 10,870 |
| Latest validated export | 770 | 3,763 | 9,667 |

The filler universe is unchanged; only the learned probabilities changed. This
is why a refreshed `slot_filler_model_scores.csv` can broaden pop/mainstream
generation without changing which text is allowed.

For a filler `F` and tier `T`, the rollup is:

```text
score_T(F) =
  average(student_score_T(book) for each source book that contributed F)
```

Those `score_T(F)` values are the deployed per-filler model weights.

## 12. Calibrate the runtime tier model

After the rich teacher, exportable students, and selected shadow rollup exist,
run the single runtime tier-model calibration step:

```powershell
uv run subtitle-gen calibrate-runtime-tier-model `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --rollup generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv
```

The step fits the collapsed popularity scalar:

$$
\mathrm{popularity\_scalar}(x) =
  \sum_i
    \frac{\max(0, \mathrm{pop\_weight}_i)}
         {\sum_j \max(0, \mathrm{pop\_weight}_j)}
    x_i
$$

```text
popularity_scalar =
  share_spl*SPL
+ share_ol*OpenLibrary
+ share_goodreads*Goodreads
+ share_library*Library
+ share_nyt*NYT
+ share_trove*Trove

share_i = max(0, pop_weight_i) / sum(max(0, pop_weight_*))
```

This constrained popularity calibration trains against the default accessibility
target, `teacher_pop + 0.5 * teacher_mainstream`, unless run with a different
target mode. It produces source-share weights that can be collapsed back into
runtime `pop_weight_*` values.

It then fits final assembled-subtitle classifier coefficients over the chosen
slot model probabilities, slot popularity, popularity interactions, and
frequency evidence. The command writes a report by default. With `--apply`, it
updates DB config, recomputes runtime popularity, and rebuilds book features in
the same pipeline step:

```powershell
uv run subtitle-gen calibrate-runtime-tier-model `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --rollup generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv `
  --apply
```

If `--apply` changes popularity-derived feature inputs and you want those
refreshed inputs reflected in the exportable student, rerun `Distill` and
`Shadow` before installing scores and exporting.

The per-tier popularity weights in the DB are part of this final assembled-subtitle
classifier, not the source-popularity scalar above. They are learned by minimizing
MSE against the rich teacher's `score_pop`, `score_mainstream`, and `score_niche`
for generated book-feature examples that have complete slot evidence. For each
generated subtitle:

$$
\begin{aligned}
z_T &= b_T +
  \frac{\sum_{k \in S}\lambda_k\left(
    \beta_m s_T(k) +
    \beta_{p,T}\mathrm{pop}(k) +
    \beta_{q,T}\mathrm{pop}(k)s_T(k) +
    \beta_{o,T}\mathrm{popObserved}(k) +
    \beta_{v,T}\mathrm{popObserved}(k)s_T(k) +
    \beta_{r,T}\mathrm{freqScore}(k) +
    \beta_{u,T}\mathrm{freqScore}(k)s_T(k)
  \right)}
  {\sum_{k \in S}\lambda_k} \\
\mathrm{runtimeScore}_T &=
  \frac{\exp(z_T / \tau)}
       {\sum_U \exp(z_U / \tau)}
\end{aligned}
$$

Here `S` is the generated subtitle's scored slots, `lambda_k` is the configured
slot weight for the slot type, `s_T(k)` is the selected student rollup score for
tier `T`, `pop(k)` is `0` when source/work popularity is missing, and
`popObserved(k)` is `1` only when the slot has real source/work popularity.
The `beta`/`b`/`tau` terms are the `tier_classifier_*` config values. Runtime
applies the slot weight to the whole per-slot contribution, then divides by the
total slot weight. Calibration builds equivalent unweighted example features
because the current exported slot weights are all `1`.

The implementation-shaped runtime version is:

```text
slot_contribution_T =
  tier_classifier_model_score_weight * slot_score_T
  + tier_classifier_popularity_weight_T * slot_observed_popularity
  + tier_classifier_popularity_interaction_T * slot_observed_popularity * slot_score_T
  + tier_classifier_popularity_observed_weight_T * slot_popularity_observed
  + tier_classifier_popularity_observed_interaction_T * slot_popularity_observed * slot_score_T
  + tier_classifier_frequency_weight_T * slot_frequency_score
  + tier_classifier_frequency_interaction_T * slot_frequency_score * slot_score_T

logit_T =
  tier_classifier_intercept_T
  + weighted_average(slot_contribution_T over generated slots)

runtime_score_T = softmax(logit_T / tier_classifier_temperature)
```

That means:

| Config family | What it means |
|---|---|
| `tier_classifier_popularity_weight_T` | Observed popularity's direct push toward tier `T`, regardless of the student's tier score. Missing popularity contributes `0` here. |
| `tier_classifier_popularity_interaction_T` | Observed popularity's amplification or dampening of the student's evidence for tier `T`. Missing popularity contributes `0` here. |
| `tier_classifier_popularity_observed_weight_T` | Whether the presence of real popularity evidence itself pushes the assembled subtitle toward tier `T`. |
| `tier_classifier_popularity_observed_interaction_T` | Whether the presence of real popularity evidence amplifies or dampens the student's evidence for tier `T`. |
| `tier_classifier_frequency_weight_T` | Whether common source fillers push the assembled subtitle toward tier `T`. |
| `tier_classifier_frequency_interaction_T` | Frequency's amplification or dampening of the student's evidence for tier `T`. |

The current exported tier popularity weights are small compared with the model
score multiplier, so they adjust the selected student rollup rather than replace
it. For example, `tier_classifier_popularity_weight_pop` is negative while
`tier_classifier_popularity_interaction_pop` is positive: raw popularity alone
does not automatically make a subtitle pop, but popularity can strengthen pop
evidence when the student already sees pop-shaped slots.

## 13. Optional sample review gates

The review gates are optional sanity-check tooling. They do not grade real
books, train models, compute weights, or block export by themselves.

```powershell
uv run subtitle-gen categorization-gate --dry-run
```

| Command | Purpose | Status |
|---|---|---|
| `categorization-gate` | Generate fixed pure-categorization samples for human/ad hoc LLM review. | Preferred review check for learned-tier runtime behavior. |

`categorization-gate` samples from the same strict, literal-filtered filler
universe as runtime. Dry-run writes sample/report artifacts without LLM review.
Use `-ReviewGates` in the runner only when you actually want LLM judging.

## 14. Install selected scores into the full DB

```powershell
uv run subtitle-gen install-book-model-scores `
  --input generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv
```

`install-book-model-scores` replaces the local `slot_filler_model_scores` table
atomically.

| Installed column | Source rollup column |
|---|---|
| `slot_filler_id` | `slot_filler_id` |
| `score_pop` | `avg_score_pop` |
| `score_mainstream` | `avg_score_mainstream` |
| `score_niche` | `avg_score_niche` |
| `model_tier` | `book_model_tier` |
| `source_prediction_count` | `source_prediction_count` |

If the CSV is missing required columns or contains invalid rows, the old score
table is left intact.

## 15. Export tracked deployment CSVs and build the mini DB

```powershell
uv run subtitle-gen build-tier-slot-distribution
uv run subtitle-gen install-tier-slot-runtime `
  --artifact generated-artifacts\tier-slot-distribution\tier_slot_filler_distribution_v1.csv
uv run subtitle-gen export-data -o api\data
uv run subtitle-gen build-db -d api\data -o api\data\subtitles.mini.db
```

Tracked deployment inputs:

| File | Purpose |
|---|---|
| `api\data\slot_fillers.csv` | Strict filler inventory and runtime scalar state. |
| `api\data\tier_slot_filler_distribution_v1.csv` | Default anchored `P(filler | tier, slot_type)` runtime artifact. |
| `api\data\slot_filler_model_scores.csv` | Learned tier probabilities retained as build evidence and rollback state. |
| `api\data\sources.csv` | Source-book attribution. |
| `api\data\config.csv` | Runtime configuration. |
| `api\data\source_tier_labels.csv` | Offline source-label evidence for training/evaluation continuity. |

Ignored local/CI-built artifacts:

| Path | Purpose |
|---|---|
| `api\data\subtitles.mini.db` | Built SQLite artifact for local serving and Azure Functions; CI rebuilds it from tracked CSVs. |
| `generated-artifacts\` | Local reports, features, predictions, rollups, and gate outputs. |

If `slot_filler_model_scores.csv` exists, `build-db` requires one score row for
every exported slot filler. Partial coverage is rejected so deployment cannot
mix learned-tier rollback sampling with unscored filler choices. When
`generation_runtime_mode=artifact`, `build-db` also requires and validates the
tier-slot distribution CSV.

## 16. Repeat the book-model path with the runner

For repeatable operation, prefer the checked-in runner:

```powershell
pwsh -File scripts\run-book-model-pipeline.ps1 -Steps Inventory

pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps Features,Baseline,Torch,Distill,Shadow,CalibrateRuntimeTierModel,CategorizationGate

# Apply using the existing selected rollup after reviewing the calibration report.
pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps CalibrateRuntimeTierModel `
  -ApplyPopularityCalibration

# If applied popularity should feed export-safe features, rerun dependent outputs.
pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps Distill,Shadow,CategorizationGate

pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps InstallScores,TierSlotDistribution,InstallTierSlotRuntime,ExportData,BuildDb,Validate
```

The runner defaults to the safe inventory step and executes requested steps in
its built-in dependency order, so `CalibrateRuntimeTierModel` runs after
`Distill` and `Shadow` when they are selected together. `CalibrateRuntimeTierModel`
is report-only unless `-ApplyPopularityCalibration` is set; when applied, that
single step recomputes `popularity_score`, rebuilds book features, and writes
the learned runtime classifier coefficients. Expensive training, installation,
export, mini-DB build, validation, and optional review sampling are explicit
steps. It fails fast when a native command fails.

## 17. Runtime generation and classification

Generation loads strict candidates for:

| Slot | Source |
|---|---|
| `list_item` | Two sampled list fillers. |
| `action_noun` | One sampled action noun. |
| `of_object` | One sampled final object, optionally remixed. |

The default runtime chooses the requested/default tier and samples directly from
the installed normalized distribution for each slot:

```text
P(filler | requested_tier, slot_type)
```

This direct path intentionally avoids the legacy classifier-retry loop that
caused catastrophic pop/mainstream repetition. `slot_filler_model_scores` and
the `sqrt(freq) * score` formula are used only by explicit legacy rollback.

`compute_tier_evidence()` classifies generated subtitles:

| DB state | Classifier behavior |
|---|---|
| Complete `slot_filler_model_scores` plus calibrated `tier_classifier_*` config | Compute calibrated per-tier logits from student rollup scores, slot popularity, popularity interactions, and frequency, then softmax and choose the highest tier with deterministic tie-breaking. |
| Complete `slot_filler_model_scores` with default classifier config | Average per-slot learned tier probabilities and choose the highest tier with deterministic tie-breaking. This is the fallback because defaults set model weight to `1.0` and all intercept/popularity/frequency coefficients to `0.0`. |
| Remixed `of_object` | Classify from the structured remix components (`of_modifier` + `of_head`, or `of_topic` + `of_complement`) rather than looking for the newly composed object as a separate filler. |
| Missing generated-slot evidence | Return neutral mainstream evidence for arbitrary/user-entered text; generated subtitles should carry scored slots or remix parts. |

The classifier also returns compatibility evidence such as per-slot details,
accessibility score, lower-tail score, and demand confidence for UI/debugging.

## 18. Serving surfaces

```text
CLI -> generate.py / jacket.py
local web -> serve.py -> handlers.py -> shared runtime modules
Azure Functions -> api\function_app.py -> handlers.py -> shared runtime modules
```

`handlers.py` is the shared API boundary:

| Handler | Purpose |
|---|---|
| `handle_generate` | Generate a subtitle and return slot/source/remix metadata. |
| `handle_jacket` | Build a jacket prompt; local mode can also stream LLM output. |
| `handle_rate` | Store human feedback. |
| `handle_health` | Health and mode response. |

The frontend is intentionally thin. It renders API results, shows sources and
remix parts, captures feedback, and builds jacket prompts. Runtime decisions stay
server-side.

## 19. Validate locally

Run these before deployment or merging changes:

```powershell
uv run subtitle-gen validate-pipeline
uv run ruff check
uv run ty check
uv run pytest -q
```

For web/API or browser-facing changes, run the local e2e gate:

```powershell
uv sync --extra deploy --extra tune --extra e2e
uv run playwright install --with-deps chromium
pwsh -File scripts\run-local-e2e.ps1
```

`scripts\run-local-e2e.ps1` starts `subtitle-gen serve --no-open` on
`http://127.0.0.1:8742`, waits for readiness, captures before/after
screenshots, runs `tests\test_e2e.py`, runs `tests\test_e2e_spot_check.py`, and
writes artifacts to `test-results\local-e2e`.

## 20. Deploy

Deployment is handled by GitHub Actions:

| Workflow | Trigger | Responsibilities |
|---|---|---|
| `.github\workflows\deploy-infra.yml` | Push to `infra/**` or manual dispatch | Deploy Azure resources, static website setup, monitoring, alerts, and optional RBAC role assignments. |
| `.github\workflows\deploy.yml` | Push to `master` touching `web/**`, `api/**`, `src/**`, or manual dispatch | Build the mini DB from CSVs, deploy Azure Functions, upload the static website, smoke-test the API, and run deployed e2e. |

Required GitHub configuration:

| Name | Type |
|---|---|
| `AZURE_CLIENT_ID` | Secret |
| `AZURE_TENANT_ID` | Secret |
| `AZURE_SUBSCRIPTION_ID` | Secret |
| `AZURE_FUNCTIONAPP_NAME` | Variable |
| `ALERT_EMAIL` | Optional secret or workflow input |

The production site currently targets:

```text
https://subtitlegenst.z13.web.core.windows.net
```
