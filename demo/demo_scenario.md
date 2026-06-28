# Demo Scenario

## Goal

Show that the project is more than a model script: it is a local ML operations surface for inspecting data, running experiments, sampling checkpoints, and reviewing artifacts.

## Recording Flow

1. Open the Overview page and explain the pipeline: dataset QA, AE/VAE reconstruction, GAN training, sampling, metrics, and run comparison.
2. Start the local stack with `scripts/dev_fullstack.sh`.
3. Use Data Tools to create a tiny demo imagefolder or point to an existing dataset.
4. Run Data Report and open `report.json` plus sample grids.
5. Run a CPU smoke GAN or AE job and watch logs/metrics stream in the Jobs panel.
6. Open Runs, inspect `meta.json`, `metrics.jsonl`, notes/tags, and artifact files.
7. Load a GAN checkpoint in Sampler and generate a deterministic grid.

## Screenshot States

- Overview dashboard with API in demo mode.
- Data Report summary with green quality pills.
- Job panel streaming logs and metrics.
- Runs comparison with metric chart.
- Config editor showing safe overrides before launching a job.
