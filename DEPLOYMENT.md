# Deployment

## Recommended Setup

Use a split deployment:

1. Public static demo: deploy `portfolio-web/` or the built `gan-ui/dist/` overview/demo mode to GitHub Pages, Vercel, or Netlify.
2. Full ML workflow: run FastAPI + Vite locally with `scripts/dev_fullstack.sh`, or deploy behind authentication on a private service.

This split is intentional. The backend includes a local job runner and filesystem browser, which are useful for demos but should not be exposed publicly without access control.

Expected GitHub Pages URL:

```text
https://justin21523.github.io/GAN-AE-vision-suite/
```

## Static Portfolio Page

The simplest public deployment is `portfolio-web/`:

```bash
docker build -f docker/portfolio.Dockerfile -t gan-ae-vision-suite-portfolio .
docker run --rm -p 8080:80 gan-ae-vision-suite-portfolio
```

Then open `http://127.0.0.1:8080`.

## Static React Demo

```bash
cd gan-ui
npm ci
npm run build
```

Deploy `gan-ui/dist/` to GitHub Pages, Vercel, or Netlify. Without a backend, the Overview page still provides a stable demo state. Backend-driven pages will show API connection errors until a FastAPI endpoint is configured.

This repository also commits a ready-to-serve `docs/` site. The workflow in `.github/workflows/pages.yml` publishes `docs/` directly, including:

- `/` portfolio demo page,
- `/app/` static React overview demo,
- `/assets/` screenshots and demo video.

## Local Fullstack Demo

```bash
python -m pip install -r requirements.txt
cd gan-ui && npm ci
cd ..
scripts/dev_fullstack.sh
```

- UI: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000`

## Protected Backend Option

Render, Railway, Fly.io, or a private VM can host the API if you add:

- authentication in front of all `/api/jobs/*` and `/api/fs/*` routes,
- strict `AI_CACHE_ROOT` outside the repo,
- CPU/GPU resource limits,
- persistent storage for `.ai_cache`, logs, and outputs,
- CORS locked to the deployed frontend origin.

Do not expose the current backend as-is on the public internet.

## Environment Variables

- `AI_CACHE_ROOT`: artifact/cache root for jobs, logs, model outputs. Defaults to `./.ai_cache`.
- `MAX_JOBS`: max concurrent local jobs. Defaults to `2`.
- `HOST` / `PORT`: API bind host/port for `scripts/serve_api.sh`.
- `VITE_API_BASE_URL`: optional frontend API base, e.g. `https://example.com/api`.

## Release Checklist

```bash
pytest -q
cd gan-ui && npm run lint && npm run build
python -m src.scripts.make_demo_assets --out ./outputs/demo
```

Confirm no large artifacts are staged:

```bash
git status --short
```

Keep checkpoints, datasets, logs, and generated outputs out of git unless they are tiny intentional fixtures under `demo/`.
