# Step 8 diversity-first reevaluation

This is a separate policy decision over the frozen Step 8 samples, ratings,
and metrics. It does not modify `feedback/step08-validation/decision.json`.

**Decision:** `promote`

**Recommended variant:** `anchored_base`

Catastrophic subtitle repetition is an automatic failure. Distributional
tone separation, quality, tail retention, and a minimum compatible tier
signal remain required. Retry-era per-subtitle continuity is diagnostic.

Reproduce with:

```powershell
uv run subtitle-gen reevaluate-diversity-first
```
