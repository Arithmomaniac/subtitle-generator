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
**Decision digest:** `aaa5abf6e2de890779ddc079dc82f08212eeec60c2634b471957112592e60635`

Step 8 recommends defer. No direct-draw variant cleared the complete rollout gate; anchored_base is the best shadow candidate.

Evaluation source digest: `282e1fc9aacb8d7ccf34f324b1a5bac59805d67cdd5010584ce95eacc842c822`
(base revision provenance: `3d9846c5c5fa6a1a6185f88ffe15928827ec1c7b`)

The command does not change runtime defaults.
