# Step 8 artifact-runtime validation

This folder contains the tracked promote/iterate/defer decision for issue #41.
Regenerable samples, ratings, artifacts, and the full report remain under
`test-results/step08-validation/`.

## Replay

```powershell
uv run subtitle-gen validate-artifact-runtime `
  --db C:\_SRC\subtitle-generator\data\db\subtitles.db
```

**Decision:** `promote`
**Decision digest:** `669c34635203cf7407fb1e81da16d87ffab7bf68c3a606b017a7dda6294cab80`

Step 8 recommends promote. The bounded winner is anchored_base.

Code binding: `3781f1976cb623c5bc558b5c5ea0e7ec819b4c04` / `59aafe997fa5710d4eaf5f628522bb2f72a5702bc27a8e72c49d23dddc81435b`

The command does not change runtime defaults.
