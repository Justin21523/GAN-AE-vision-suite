# GAN-AE Vision Suite

Portfolio-ready computer vision lab for dataset inspection, AE/VAE reconstruction, GAN training, checkpoint sampling, metrics tracking, and artifact review.

The project is intentionally local-first: the public demo can run as a static portfolio page, while the full ML workflow runs through a local FastAPI backend so training jobs and filesystem access are not exposed on the open internet.

## Highlights

- Config-driven PyTorch pipelines for AE/VAE and GAN experiments.
- Torchvision-free dataset, transform, and image-grid utilities for better environment compatibility.
- FastAPI backend with checkpoint sampling, job runner, config validation/overrides, filesystem artifact browser, run registry, and run comparison.
- React/Vite dashboard for demo overview, workflow launch, logs, metrics, sampling, configs, files, and runs.
- CPU-friendly smoke tests and demo assets for reliable interview walkthroughs.

## Tech Stack

- ML: Python, PyTorch, PIL, NumPy, YAML configs.
- API: FastAPI, Pydantic, Uvicorn, Server-Sent Events for live job logs/metrics.
- UI: React, Vite, plain CSS.
- Artifacts: filesystem-based `logs/`, `outputs/`, `.ai_cache/`; no database required.
- Deployment: static public demo plus local or protected fullstack backend.

## Main Features

- Data QA: scan image folders, bad files, image modes/sizes, tensor statistics, sample grids.
- AE/VAE: train reconstruction models, save input/recon grids, PSNR/SSIM, checkpoints.
- GAN: DCGAN/WGAN-GP/SNGAN-style configs, checkpoints, EMA-aware sampling, FID/KID hooks.
- Local jobs: run whitelisted CLI tasks from the UI and stream logs/metrics.
- Runs: scan `logs/**/meta.json`, inspect metrics, tag notes, clone configs, compare runs.
- Configs: validate YAML, apply overrides/overlays, save local experiment configs.

## Project Structure

```text
src/                 Python package: API, data, models, metrics, scripts, services
configs/             Dataset/model/GAN YAML configs and split lists
gan-ui/              React + Vite dashboard
demo/                Safe-to-commit demo scenario and sample JSON artifacts
portfolio-web/       Static portfolio landing page
scripts/             Local API/fullstack shell helpers
tests/               CPU-friendly pytest suite
```

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
cd gan-ui && npm ci
```

Run the full local demo:

```bash
scripts/dev_fullstack.sh
```

Open:

- UI: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000`

## Useful Commands

```bash
# Tests
pytest -q

# Frontend checks
cd gan-ui
npm run lint
npm run build

# Create deterministic screenshot artifacts
python -m src.scripts.make_demo_assets --out ./outputs/demo

# Create a tiny demo image folder for smoke runs
python -m src.scripts.prepare_data --create-demo-imagefolder ./data/demo_images --num-images 32 --img-size 64

# Data report
python -m src.scripts.data_report --config configs/dataset_celeba.yaml --out ./outputs/data_report

# GAN training example
python -m src.scripts.train_gan --config configs/gan/wgangp_celeba128.yaml

# API only
scripts/serve_api.sh
```

## Demo Strategy

Public deployment should use the static overview/portfolio pages and sample artifacts in `demo/`. The full FastAPI backend exposes local job execution and filesystem browsing, so it should be run locally or behind authentication.

Recommended interview flow:

1. Show the Overview page and explain the ML pipeline.
2. Run Data Report on a small dataset and open generated grids/report JSON.
3. Start a CPU smoke training job and show streaming logs/metrics.
4. Open Runs to inspect `meta.json`, `metrics.jsonl`, tags, and artifacts.
5. Load a checkpoint in Sampler and generate a deterministic grid.

## Deployment

- Static demo: GitHub Pages, Vercel, or Netlify.
- GitHub Pages URL after deployment: `https://justin21523.github.io/GAN-AE-vision-suite/`
- Full local demo: `scripts/dev_fullstack.sh`.
- Full remote backend: only behind authentication; do not publicly expose job runner or filesystem write endpoints.

See [DEPLOYMENT.md](DEPLOYMENT.md) for recommended options.

## Current Verification

- `pytest -q`
- `cd gan-ui && npm run lint`
- `cd gan-ui && npm run build`

The test suite is designed to remain CPU-friendly and avoids external dataset downloads.
