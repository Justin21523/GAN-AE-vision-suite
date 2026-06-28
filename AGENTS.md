# Repository Guidelines

## Project Structure & Module Organization

- `src/`: Python package (models, training loops, data loading, metrics, API/UI entrypoints).
  - `src/models/`: `ae/` (autoencoders) and `gan/` (generators/discriminators/EMA).
  - `src/training/`: trainer implementations used by scripts.
- `configs/`: YAML configs for datasets + GAN/AE training (see `configs/dataset_*.yaml`, `configs/gan/*.yaml`).
- `scripts/`: shell helpers (API server + fullstack dev).
- `src/scripts/`: Python utilities (training, sampling, evaluation).
- `tests/`: `pytest` tests (`test_*.py`) intended to stay CPU-friendly and fast.
- `gan-ui/`: React + Vite frontend (dev server, build, ESLint).

## Build, Test, and Development Commands

- Python deps: `python -m pip install -r requirements.txt`
- Run tests: `pytest -q`
- Train a GAN run (example): `python -m src.scripts.train_gan --config configs/gan/wgangp_celeba128.yaml`
- Validate dataset + PSNR/SSIM (example): `python -m src.validate_data --config configs/dataset_celeba.yaml --use_ae`
- FastAPI (dev): `uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`
- UI dev server: `cd gan-ui && npm ci && npm run dev`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes.
- Formatting/linting: run `black .` and `ruff check .` before opening a PR.
- Config-first: prefer adding options to `configs/*.yaml` over hardcoding paths/hyperparams.
- React: components `PascalCase` (`GridViewer.jsx`), hooks `useSomething`, lint via `npm run lint`.

## Testing Guidelines

- Framework: `pytest` (files: `tests/test_*.py`, tests: `test_*`).
- Keep tests deterministic (set seeds), lightweight (small tensors), and runnable on CPU.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits seen in history (e.g., `feat(gan): ...`, `fix(api): ...`, `chore(ui): ...`).
- PRs include: what/why, how to run/verify (commands + configs), and UI screenshots when changing `gan-ui/`.
- Do not commit large artifacts (checkpoints, logs, datasets); keep them under ignored dirs like `logs/`, `outputs/`, `data/`.

## Security & Configuration Tips

- Set `AI_CACHE_ROOT` to a writable “warehouse” directory when running services/scripts that materialize models, logs, or metrics.
