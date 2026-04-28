"""Pytest collection settings for unit/characterization tests."""

# Browser e2e scripts expect a running local server and are invoked via
# scripts/run-local-e2e.ps1, not by the unit-test pytest run.
collect_ignore = ["test_e2e.py", "test_e2e_spot_check.py"]
