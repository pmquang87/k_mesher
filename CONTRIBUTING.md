# Contributing to k-mesher

Thanks for your interest in improving k-mesher! This guide covers local
development setup, testing, linting, and the release/version-bump procedure.

## Development setup

k-mesher is a `k_mesher/` package with a setuptools build backend. Install it
in editable mode together with the development tools:

```bash
pip install -e ".[all]"
pip install pytest pytest-cov ruff
```

`pip install -e .` (or `pip install -r requirements.txt`, which does the same
thing) pulls in the runtime dependencies (`gmsh`, `numpy`) and exposes the
console scripts `k-mesher`, `k-mesher-gui`, `k-mesher-doe`, `k-mesher-convert`
and `k-mesher-post`; the `[all]` extra adds the optional bridge dependencies
(meshio, lasso-python, scipy, matplotlib) whose tests are otherwise skipped.

## Running tests

```bash
pytest tests/ -v
```

To collect coverage the way CI does:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=xml
```

Some tests exercise the Tkinter GUI. They need `tkinter` available and a
display. On a headless Linux machine, wrap the run with Xvfb:

```bash
xvfb-run -a pytest tests/ -v
```

## Linting

```bash
ruff check .
```

The active ruleset in `ruff.toml` is intentionally kept to the ruff defaults
(`E4`/`E7`/`E9`/`F`) so CI stays green.

### Recommended broader ruleset

Over time we want to adopt a broader, still-safe ruleset. Once the codebase has
been cleaned up to pass it, enable the following `select` in `ruff.toml`:

```toml
[lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
```

- `E`   — pycodestyle errors
- `F`   — pyflakes
- `I`   — isort (import sorting)
- `UP`  — pyupgrade (modern syntax)
- `B`   — flake8-bugbear (likely bugs)
- `SIM` — flake8-simplify
- `RUF` — ruff-specific rules

Start with the low-risk `I` and `UP`, fix what they flag, then layer in `B`,
`SIM`, and `RUF`.

## Release procedure

The version lives in a single place: `k_mesher/_version.py`.

```python
__version__ = "0.6.0"
```

`pyproject.toml` reads it dynamically:

```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = { attr = "k_mesher._version.__version__" }
```

To cut a release:

1. Edit `__version__` in `k_mesher/_version.py`.
2. Move the `## [Unreleased]` items into a new dated section in `CHANGELOG.md`
   following the [Keep a Changelog](https://keepachangelog.com/) format, and add
   the corresponding compare/release links at the bottom.
3. Because the version is dynamic, you do **not** need to edit `pyproject.toml`
   — it always follows `_version.py`.
4. Merge those changes to `main`, then either tag and push the tag:

   ```bash
   git tag v0.6.0
   git push origin v0.6.0
   ```

   or dispatch the `release` workflow on `main` ("Run workflow" in the
   Actions tab) — the dispatch path derives the tag from `_version.py` and
   creates it for you (it refuses to run if the tag already exists).

   Either way the `release` workflow (`.github/workflows/release.yml`) builds
   the sdist + wheel, `twine check`s them, verifies the tag matches
   `_version.py`, smoke-tests the console entry point, and creates a GitHub
   Release with the artifacts attached.

### PyPI publishing (Trusted Publishing)

The workflow's `pypi` job publishes via [PyPI Trusted
Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no API token
stored in the repo). It requires one-time setup on pypi.org: add a trusted
publisher for the `k-mesher` project with repository `pmquang87/k_mesher`,
workflow `release.yml`, and environment `pypi`. Until that is configured the
`pypi` job fails while the GitHub Release still ships.

Verify the resolved version after a bump:

```bash
python -m build          # produces dist/k_mesher-<version>-*.whl
```
