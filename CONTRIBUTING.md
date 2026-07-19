# Contributing to open_dvm

Thanks for your interest in contributing! This document covers how to set up a development environment, run the test suite, and submit changes.

## Development Setup

```bash
git clone https://github.com/dvanmoorselaar/open_dvm.git
cd open_dvm
pip install -e ".[dev]"
```

This installs `open_dvm` in editable mode along with test/lint tooling (`pytest`, `pytest-cov`, `pytest-xdist`, `black`, `isort`, `flake8`, `mypy`).

## Running Tests

```bash
pytest tests/
```

Tests are organized under `tests/test_analysis/`, `tests/test_stats/`, `tests/test_support/`, and `tests/test_visualization/`, mirroring the `open_dvm/` package layout. Three markers are available (defined in `pyproject.toml`):

- `unit` -- fast, isolated tests
- `integration` -- slower tests exercising multiple components
- `slow` -- long-running tests; skip with `pytest -m "not slow"` (this is what CI runs)

Coverage is configured via `[tool.coverage.*]` in `pyproject.toml`:

```bash
coverage run --source=open_dvm -m pytest tests/ -q
coverage report --show-missing
```

## Code Style

`black`, `isort`, `flake8`, and `mypy` configs are already present in `pyproject.toml` for anyone who wants to use them locally. **They are not currently enforced in CI** -- the existing codebase has substantial pre-existing formatting differences from `black`'s defaults (notably tab-based indentation throughout), so a strict style gate isn't in place yet. When touching a file, prefer matching its existing surrounding style over introducing a mix of tabs and spaces or unrelated reformatting in the same diff.

For file and folder naming conventions (raw/processed data layout, subject/session naming, etc.), see [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md).

## Submitting Changes

1. Branch from `master`.
2. Make your changes, adding or updating tests for any new behavior.
3. Make sure `pytest tests/` passes locally.
4. Open a pull request against `master`. CI (GitHub Actions) runs the test suite across Python 3.9-3.11 on every PR.
5. Keep PRs focused -- a bug fix or feature addition doesn't need to also reformat unrelated code.

## Questions?

Open a [GitHub issue](https://github.com/dvanmoorselaar/open_dvm/issues) or see the [README](README.md) for contact details.
