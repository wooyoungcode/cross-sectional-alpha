# Contributing

## Development Setup

```bash
git clone https://github.com/wooyoungcode/alpha-research.git
cd alpha_research
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

Use `configs/synthetic.toml` for all tests — it runs offline with no market data download.

## Project Layout

```
src/alpha_research/   Package source
tests/                pytest test suite
configs/              TOML research configs
docs/                 Generated research report and templates
outputs/              Generated plots and CSVs (git-ignored)
data/cache/           Pickle cache for downloaded panels (git-ignored)
```

## Making Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes and add tests for any new behaviour.
3. Run `pytest` and confirm all tests pass.
4. Open a pull request with a clear description of what changed and why.

## Code Style

- Python 3.11+ with full type hints on all public functions.
- Use `from __future__ import annotations` at the top of every module.
- All cross-sectional operations must be per-date to avoid look-ahead bias.
- Do not add runtime dependencies beyond what is in `pyproject.toml`.

## Reporting Issues

Open an issue with a minimal reproducible example. Include your Python version and the config file used.
