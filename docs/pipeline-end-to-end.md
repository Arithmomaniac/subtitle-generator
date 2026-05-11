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
| Student model | The exportable Torch model trained to imitate the teacher using durable/export-safe features. |
| Rollup | Aggregation from book-level predictions back onto filler-level scores. |

## 0. Runtime contract

The deployed generator has four responsibilities:

| Responsibility | Runtime state | Effect |
|---|---|---|
| Filler universe | `slot_fillers` | Defines which source-derived fillers are allowed at all. |
| Tier categorization | `slot_filler_model_scores` | Gives each filler `score_pop`, `score_mainstream`, and `score_niche`. |
| Generation | `generate.py` | Samples strict fillers using frequency and the requested tier score. |
| Literal guardrail | `generate.py` | Blocks broken artifacts without blocking funny absurdity. |

When model scores are present, requested tier `T` uses:

```text
weight = sqrt(freq) * score_T
```

If model scores are absent, generation falls back to legacy
frequency/popularity/tone-target weighting. Deployment is expected to include a
complete `slot_filler_model_scores.csv`, so the fallback is mainly for old local
DBs and debugging.

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

Popularity is source evidence and fallback signal. It is not the final deployed
tier classifier when model scores are present.

| Source | Signal |
|---|---|
| Seattle Public Library | Checkout demand. |
| Goodreads / UCSD Book Graph | Rating count and engagement. |
| NYT | Bestseller appearances. |
| Ottawa / library data | Library demand and holdings. |
| Trove Australia | Library breadth via `holdingsCount`. |
| Open Library edition count | Edition-breadth signal. It can act like a fallback prior when other demand evidence is sparse, but it is also a learnable source in the constrained popularity block. |

The key filler fields are:

| Column | Meaning |
|---|---|
| `popularity_score` | Historical/fallback source popularity score. |
| `popularity_level` | Coarse availability of source popularity evidence. |
| `popularity_confidence` | Confidence in the popularity score. |

Popularity matters because it gives the models and fallback generator demand
evidence, but it is too blunt to distinguish all pop/mainstream/niche behavior.

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

## 5. Check runtime config at the boundary it affects

Runtime constants live in the full DB `config` table, then get exported to
`api\data\config.csv` and packaged into the mini DB. In a normal rerun, these
rows may already exist; you do not rerun every tuner just because you are
retraining the book model.

The timing depends on what the constant feeds:

| If you change... | Run or rerun here | Then rerun... | Why |
|---|---|---|---|
| `pop_weight_*`, `pop_exponent` | Before or during `populate-popularity`; source ratios can be learned after the rich teacher with `calibrate-popularity-weights` | `populate-popularity`, `build-book-features`, student distillation, export | These change `slot_fillers.popularity_score`, which is both runtime fallback state and student-model input. |
| `remix_calibrated_remix_prob`, `remix_calibrated_min_sim` | After slots, popularity, and remix precompute exist | final sample checks and `export-data` | CLI/API defaults and sample review gates consume these values directly. |
| `article_of_min_freq`, `article_action_min_freq`, `article_remix_heuristic_threshold` | After generation can produce realistic samples | final sample checks and `export-data` | These are generation heuristics; changing them does not retrain weights unless you use generated samples as review evidence. |
| `generation_tier_ratio_pop/mainstream/niche` | Before final runtime validation/export | `export-data`, `build-db` | These control the default tier mix when no explicit tone is requested. |
| `tier_center_*`, `accessibility_threshold_*`, `weighted_sample_*`, `pop_base_weight_blend`, `pop_classification_blend`, `pop_missing_default` | Before fallback diagnostics or legacy sampling checks | diagnostics, sample checks, export if changed | These mostly affect fallback tiering/legacy sampling when model scores are missing, plus diagnostic evidence. |

Useful commands:

```powershell
# Remix defaults: grid-search min_sim and remix probability on generated samples.
uv sync --extra tune
uv run subtitle-gen calibrate-remix --samples 50

# Popularity source ratios: train a constrained student, write a report only.
uv sync --extra ml
uv run subtitle-gen calibrate-popularity-weights `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv

# Apply learned pop_weight_* rows only after reviewing the report.
uv run subtitle-gen calibrate-popularity-weights `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --apply

# Current autoresearch loop is intentionally narrow: article/remix heuristics.
uv run subtitle-gen tune --phase all --samples 50 --iterations 30

# Fallback tier thresholds from source-title labels; not the primary classifier
# when slot_filler_model_scores is complete.
uv run subtitle-gen calibrate-tier-gates --apply
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

### Learn popularity source ratios before distillation

If you want the flat runtime `popularity_score` to use learned source ratios,
calibrate those ratios after the rich teacher exists and before distilling
exportable students:

```powershell
uv run subtitle-gen calibrate-popularity-weights `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv
```

The calibration is a constrained student. It computes:

```text
popularity_scalar =
  w_spl*SPL
+ w_ol*OpenLibrary
+ w_goodreads*Goodreads
+ w_library*Library
+ w_nyt*NYT
+ w_trove*Trove
```

That scalar can then interact with text, slot, and metadata features inside the
student, but individual source signals cannot get separate text/slot/metadata
interaction weights. This keeps the learned source coefficients readable and
collapsible into `pop_weight_*`.

The command writes a report by default. To make the learned ratios feed the
actual runtime score, rerun with `--apply`, then recompute popularity and
features before distillation:

```powershell
uv run subtitle-gen calibrate-popularity-weights `
  --features generated-artifacts\book-model\book_features.csv `
  --teacher-predictions generated-artifacts\book-model\torch-all-spacy\book_torch_predictions.csv `
  --apply

uv run subtitle-gen populate-popularity --skip-data-model
uv run subtitle-gen build-book-features
```

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

## 12. Optional sample review gates

The review gates are optional sanity-check tooling. They do not grade real
books, train models, compute weights, or block export by themselves.

```powershell
uv run subtitle-gen categorization-gate --dry-run
```

| Command | Purpose | Status |
|---|---|---|
| `categorization-gate` | Generate fixed pure-categorization samples for human/ad hoc LLM review. | Preferred review check for learned-tier runtime behavior. |
| `deployment-gate` | Legacy scalar/blend strategy comparison from the rejected intermediate design. | Kept for historical comparison and regression checks. |

`categorization-gate` samples from the same strict, literal-filtered filler
universe as runtime. Dry-run writes sample/report artifacts without LLM review.
Use `-ReviewGates` in the runner only when you actually want LLM judging.

## 13. Install selected scores into the full DB

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

## 14. Export tracked deployment CSVs and build the mini DB

```powershell
uv run subtitle-gen export-data -o api\data
uv run subtitle-gen build-db -d api\data -o api\data\subtitles.mini.db
```

Tracked deployment inputs:

| File | Purpose |
|---|---|
| `api\data\slot_fillers.csv` | Strict filler inventory and runtime scalar state. |
| `api\data\slot_filler_model_scores.csv` | Learned runtime tier probabilities. |
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
silently mix learned-tier sampling with legacy popularity sampling.

## 15. Repeat the book-model path with the runner

For repeatable operation, prefer the checked-in runner:

```powershell
pwsh -File scripts\run-book-model-pipeline.ps1 -Steps Inventory

pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps Features,Baseline,Torch,CalibratePopularity,Distill,Shadow,CategorizationGate

pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps CalibratePopularity,PopulatePopularity,Distill,Shadow,CategorizationGate `
  -ApplyPopularityCalibration

pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps InstallScores,ExportData,BuildDb,Validate
```

The runner defaults to the safe inventory step. `CalibratePopularity` is
report-only unless `-ApplyPopularityCalibration` is set. When applying learned
source ratios, include `PopulatePopularity` so the full DB recomputes
`popularity_score` and rebuilds book features before distillation. Expensive
training, installation, export, mini-DB build, validation, and optional review
sampling are explicit steps. It fails fast when a native command fails.

## 16. Runtime generation and classification

Generation loads strict candidates for:

| Slot | Source |
|---|---|
| `list_item` | Two sampled list fillers. |
| `action_noun` | One sampled action noun. |
| `of_object` | One sampled final object, optionally remixed. |

When complete model scores are present, generation uses the requested tier score
as the sampling signal:

```text
weight = sqrt(freq) * score_for_requested_tier
```

`compute_tier_evidence()` classifies generated subtitles:

| DB state | Classifier behavior |
|---|---|
| Complete `slot_filler_model_scores` | Average per-slot learned tier probabilities and choose the highest tier with deterministic tie-breaking. |
| No model scores | Fall back to the legacy blended frequency/popularity score and calibrated thresholds. |

The classifier also returns compatibility evidence such as per-slot details,
accessibility score, lower-tail score, and demand confidence for UI/debugging.

## 17. Serving surfaces

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

## 18. Validate locally

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

## 19. Deploy

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
