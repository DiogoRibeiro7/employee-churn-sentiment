# Contributing

Thanks for your interest in improving **employee-churn-sentiment**. This guide
covers the workflow and quality bar for changes.

## Development setup

```bash
poetry install
poetry run pre-commit install   # optional: run formatting on commit
```

The package is importable without installation when running from the repo root
(tests and the example notebook bootstrap `sys.path`), but `poetry install` is
recommended for a clean environment.

## Workflow

1. Create a feature branch: `git checkout -b feature/<short-name>`.
2. Make focused changes with tests alongside them.
3. Run the quality gate locally (see below).
4. Open a pull request describing the change and its rationale.

## Quality gate

All of the following must pass before a change is merged — CI enforces them on
Python 3.10–3.12:

```bash
poetry run black --check .     # formatting
poetry run pytest              # full test suite
```

Guidelines:

- **Style**: Black formatting; Google-style docstrings on public functions;
  type hints throughout.
- **Tests**: add unit tests for new behavior. Prefer the synthetic data
  generator (`employee_churn.data.make_synthetic_employee_data`) for fixtures so
  tests stay reproducible and need no real data.
- **Config over constants**: read tunables from `employee_churn.config` /
  `configs/*.yaml` rather than hardcoding paths or magic numbers.
- **No PII**: never commit real employee data. The anonymization helpers are
  baseline only.

## Commit messages

Write imperative, present-tense summaries (e.g. "Add fairness summary"). Keep the
subject under ~72 characters and explain the "why" in the body when it is not
obvious.

## Releasing

1. Move entries from `[Unreleased]` to a new version section in
   [CHANGELOG.md](CHANGELOG.md).
2. Bump the version in `pyproject.toml`.
3. Tag the release: `git tag vX.Y.Z`.
4. Update `ROADMAP.md` with completed milestones.
