# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: entry point for running the full workflow locally.
- `today_eat_what/`: package code; `workflow.py` holds the LangGraph pipeline, `models.py` and `config.py` define data/contracts, `clients.py` manages external model calls, `utils.py` provides helpers, `__init__.py` exposes package metadata.
- `README.md`: quickstart and environment notes; keep it aligned with code updates.
- `pyproject.toml` and `uv.lock`: dependency sources for `uv`; update via `uv add` or `uv lock` when changing requirements.

## Build, Test, and Development Commands
- `uv sync`: install/update all dependencies for Python 3.12+.
- `uv run python -m today_eat_what` (or `uv run python main.py`): execute the end-to-end Xiaohongshu publishing workflow with current config/mock fallbacks.
- `uv add <package>`: add runtime dependencies and regenerate `uv.lock`.
- When adding dev tooling (e.g., pytest/ruff), add with `--dev` to keep runtime small.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; keep modules and functions small and focused on one responsibility.
- Prefer explicit type hints and dataclasses/Pydantic models for request/response shapes.
- Naming: modules and functions use `snake_case`; classes use `CamelCase`; constants use `UPPER_SNAKE_CASE`.
- Keep network/model client code centralized in `clients.py`; share config defaults in `config.py` to avoid drift.

## Testing Guidelines
- No formal test suite yet; add `tests/` with `test_*.py` alongside fixtures as features grow.
- Recommend `pytest` (added as a dev dependency) and short, isolated tests for each workflow step; mock remote APIs to keep runs offline.
- Aim for coverage of branch logic in `workflow.py` (timeouts, fallbacks, retries) and validation layers in `models.py`.
- Run tests with `uv run pytest` once tooling is added.

## Commit & Pull Request Guidelines
- Commits: concise present-tense subjects (e.g., `Add meal-time classifier fallback`); group related changes and avoid mixing refactors with feature work when possible.
- Pull requests: include what changed, why, and how to verify (commands/output); link related issues/tasks and note config additions (new env vars, flags).
- Add screenshots or sample outputs for workflow steps that affect generated content; list any new dependencies or external services touched.

## Configuration & Security Notes
- Secrets live in `.env`; never commit keys. Provide placeholder values in examples and document required env vars in `README.md`.
- Keep mock fallbacks intact for local/offline runs; ensure new client code preserves timeouts and retries.
