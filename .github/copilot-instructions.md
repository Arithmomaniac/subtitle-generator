# Copilot instructions

This project uses `uv` for Python dependency management. Run Python commands through `uv run`.

For changes touching the web UI, browser-facing API behavior, `src/subtitle_generator/serve.py`, or Playwright tests, run the local browser verification script:

```bash
pwsh -File scripts/run-local-e2e.ps1
```

The script starts `subtitle-gen serve --no-open` on `http://127.0.0.1:8742`, waits for readiness, captures before/after screenshots, runs the home and spot-check Playwright flows, and stops the server. Screenshots and server logs are written to `test-results/local-e2e/`.

If Playwright is not already installed in the environment, install the e2e dependencies first:

```bash
uv sync --extra deploy --extra tune --extra e2e
uv run playwright install --with-deps chromium
```
