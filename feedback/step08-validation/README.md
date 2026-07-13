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
**Decision digest:** `a168f33f758cab041d7b46b224c0f2031be5ed4ea9bfc4419051e90bbc3da0ff`

Step 8 recommends defer. No direct-draw variant cleared the complete rollout gate; anchored_base is the best shadow candidate.

Evaluation source digest: `2315c8cd5439c778ca6ad922d416603795c47b51d407a250accd16a142e9d583`

The command does not change runtime defaults.
