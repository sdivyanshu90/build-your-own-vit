# Contributing

Thanks for your interest in improving **Build Your Own ViT**! This guide gets you
productive quickly.

## Development setup

```bash
git clone https://github.com/your-org/build-your-own-vit
cd build-your-own-vit
python -m pip install -e ".[all]"   # dev + serve + export extras
pre-commit install                  # install git hooks (or: make hooks)
```

## Workflow

1. Create a branch: `git checkout -b feat/short-description`.
2. Make focused changes with tests and docs.
3. Run the full gate locally: `make check` (ruff + mypy + pytest).
4. Commit using clear, imperative messages (Conventional Commits welcome:
   `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
5. Open a PR against `main`. CI must be green.

## Quality standards

- **Style/format**: `ruff format` + `ruff check` (config in `pyproject.toml`).
- **Types**: fully typed public APIs; `mypy src` must pass. The package ships
  `py.typed`.
- **Tests**: `pytest`, and coverage must stay **≥ 95%** (`--cov-fail-under=95` in
  CI). Prefer behavioral tests over line-chasing. Use the synthetic `fake`
  dataset and CPU so tests are hermetic and fast.
- **Docs**: update the relevant file under `docs/` and any docstrings.

## Adding features — where things go

| You want to add… | Do this |
|------------------|---------|
| A new architecture variant | Add a preset to `config.PRESETS`, or extend `VisionTransformer`. |
| A new dataset | Extend `DataModule._setup_*` and the `dataset` Literal in `config.py`. |
| A new optimizer/scheduler | Add a branch in `training/optim.py` / `training/scheduler.py`. |
| A new endpoint | Add it under `serving/app.py` with a schema in `serving/schemas.py` and tests. |
| A new CLI command | Add a subparser + `_cmd_*` in `cli.py`. |

## Commit hooks

Pre-commit runs whitespace/eol fixers, ruff (lint + format), and mypy on `src/`.
Run manually with `pre-commit run --all-files`.

## Reporting bugs

Open an issue with: `vit info` output, a minimal repro (config + command), the
full JSON logs (including `request_id` for serving issues), and expected vs.
actual behavior. For security issues, use a private advisory (see
[docs/security.md](docs/security.md)).

## Code of Conduct

Be respectful and constructive. We follow the spirit of the Contributor
Covenant.
