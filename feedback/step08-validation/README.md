# Step 8 artifact-runtime validation

This folder contains the tracked promote/iterate/defer decision for issue #41.
Regenerable samples, ratings, artifacts, and the full report remain under
`test-results/step08-validation/`.

## Replay

```powershell
uv run subtitle-gen validate-artifact-runtime `
  --db C:\_SRC\subtitle-generator\data\db\subtitles.db
```

**Decision:** `defer`

Step 8 recommends defer. No bounded variant cleared every frozen gate.

The command does not change runtime defaults.
