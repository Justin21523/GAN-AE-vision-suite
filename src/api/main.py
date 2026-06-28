"""
FastAPI server for local GAN checkpoint sampling.

This repo snapshot contains a working GAN sampling core in `src/service/gan_infer.py`.
Some older/experimental API code referenced a larger "warehouse/registry/jobs"
stack that is not included here. This module provides a minimal, working API
that powers:
- `gan-ui/` (React frontend)
- simple curl/postman usage

Endpoints (all aliases of the same actions):
- POST `/api/load`, `/api/gan/load`, `/load`, `/gan/load`
- POST `/api/generate`, `/api/gan/generate`, `/generate`, `/gan/generate`
"""

from __future__ import annotations

import io
import os
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from src.api.capabilities import as_dict as capabilities_dict
from src.service.gan_infer import GANService, GenerateParams
from src.service.jobs import JobManager
from src.service.fs import SafeFS
from src.service.runs import get_run, load_run_detail, scan_runs
from src.service.run_notes import read_notes as _read_run_notes, write_notes as _write_run_notes
from src.utils.checkpoint import load_checkpoint as _load_checkpoint_file
from src.utils.config import load_config


app = FastAPI(
    title="GAN-AE-VISION-SUITE API",
    description="Minimal API for loading a GAN checkpoint and generating sample grids.",
    version="0.1.0",
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=int(exc.status_code),
        content={"error": {"code": "http_error", "message": str(exc.detail), "detail": exc.detail}, "detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "message": "Invalid request", "detail": exc.errors()}, "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Internal server error", "detail": str(exc)}},
    )

# Allow local UI/dev servers to call the API (Vite proxy usually avoids CORS, but this
# makes direct access easier during development).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Single-process global state (fine for local development; not intended for multi-worker deployments).
_service: Optional[GANService] = None
_loaded_ckpt: Optional[str] = None
_jobs: Optional[JobManager] = None
_fs: Optional[SafeFS] = None


def _repo_root() -> Path:
    # src/api/main.py -> parents[0]=api, [1]=src, [2]=repo root
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str) -> str:
    """Resolve user-provided paths (expand `~`, allow repo-relative paths)."""
    p = os.path.expanduser(p)
    path = Path(p)
    if not path.is_absolute():
        path = _repo_root() / path
    return str(path.resolve())


def _get_service() -> GANService:
    global _service
    if _service is None:
        _service = GANService()
    return _service


def _get_jobs() -> JobManager:
    global _jobs
    if _jobs is None:
        _jobs = JobManager(repo_root=_repo_root())
    return _jobs


def _get_fs() -> SafeFS:
    global _fs
    if _fs is None:
        repo = _repo_root()
        cache = Path(os.getenv("AI_CACHE_ROOT", str(repo / ".ai_cache"))).resolve()
        _fs = SafeFS([repo, cache])
    return _fs


class LoadRequest(BaseModel):
    """Request body for loading a checkpoint."""

    ckpt: str = Field(..., description="Path to a GAN checkpoint (.pt).")
    device: Optional[str] = Field(
        None, description="Optional device override (e.g., 'cuda', 'cpu', 'cuda:0')."
    )


class LoadResponse(BaseModel):
    """Response returned after successfully loading a checkpoint."""

    success: bool
    message: str
    ckpt: str
    device: str
    img_size: Optional[int] = None
    img_channels: Optional[int] = None
    latent_dim: Optional[int] = None
    has_ema_shadow: bool = False


class GenerateRequest(BaseModel):
    """Request body for generating a sample grid image."""

    n: int = Field(64, ge=1, le=4096, description="Number of samples to generate.")
    nrow: int = Field(8, ge=1, le=256, description="Grid columns (images per row).")
    seed: int = Field(42, ge=0, le=2**31 - 1, description="Random seed.")
    use_ema: bool = Field(
        False, description="Use EMA weights if the checkpoint contains them."
    )


class JobStartRequest(BaseModel):
    """Start a whitelisted CLI-backed job."""

    type: str = Field(..., description="Job type (e.g., train_gan, train_ae, data_report).")
    args: dict = Field(default_factory=dict, description="Job-specific arguments.")


class JobInfo(BaseModel):
    id: str
    name: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    return_code: Optional[int] = None
    log_path: str
    job_dir: str
    artifacts: list[str] = []
    pid: Optional[int] = None
    cmd: list[str]
    manifest_path: Optional[str] = None


def _job_to_info(j) -> JobInfo:
    manifest = str(Path(j.job_dir) / "manifest.json")
    if not Path(manifest).exists():
        manifest = None
    return JobInfo(
        id=j.id,
        name=j.name,
        status=j.status,
        created_at=float(j.created_at),
        started_at=float(j.started_at) if j.started_at is not None else None,
        finished_at=float(j.finished_at) if j.finished_at is not None else None,
        return_code=j.return_code,
        log_path=j.log_path,
        job_dir=j.job_dir,
        artifacts=list(getattr(j, "artifacts", []) or []),
        pid=j.pid,
        cmd=list(j.cmd),
        manifest_path=manifest,
    )


def _build_job_command(job_type: str, args: dict) -> tuple[str, list[str]]:
    """
    Map a job type + args to a concrete `python -m ...` command (no shell).

    This API is local-dev focused; it intentionally does not accept arbitrary commands.
    """
    job_type = str(job_type).strip().lower()
    python = "python"

    def _get(key: str, default=None):
        return args.get(key, default)

    def _path(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        # Validate path stays within allowed roots.
        fs = _get_fs()
        try:
            return str(fs.resolve(s))
        except Exception as e:
            raise ValueError(f"Invalid path for '{s}': {e}") from e

    if job_type == "train_gan":
        config = _path(_get("config"))
        if not config and not _get("resume"):
            raise ValueError("train_gan requires args.config or args.resume")
        cmd = [python, "-m", "src.scripts.train_gan"]
        if config:
            cmd += ["--config", str(config)]
        if _get("device"):
            cmd += ["--device", str(_get("device"))]
        if _get("run_name"):
            cmd += ["--run-name", str(_get("run_name"))]
        if _get("run_dir"):
            cmd += ["--run-dir", str(_path(_get("run_dir")))]
        if _get("resume"):
            cmd += ["--resume", str(_path(_get("resume")))]
        if bool(_get("finetune", False)):
            cmd += ["--finetune"]
        return "Train GAN", cmd

    if job_type == "train_ae":
        config = _path(_get("config"))
        if not config and not _get("resume"):
            raise ValueError("train_ae requires args.config or args.resume")
        cmd = [python, "-m", "src.scripts.train_ae"]
        if config:
            cmd += ["--config", str(config)]
        if _get("device"):
            cmd += ["--device", str(_get("device"))]
        if _get("epochs") is not None:
            cmd += ["--epochs", str(int(_get("epochs")))]
        if _get("run_name"):
            cmd += ["--run-name", str(_get("run_name"))]
        if _get("run_dir"):
            cmd += ["--run-dir", str(_path(_get("run_dir")))]
        if _get("resume"):
            cmd += ["--resume", str(_path(_get("resume")))]
        if bool(_get("finetune", False)):
            cmd += ["--finetune"]
        return "Train AE/VAE", cmd

    if job_type == "data_report":
        config = _path(_get("config"))
        if not config:
            raise ValueError("data_report requires args.config")
        cmd = [python, "-m", "src.scripts.data_report", "--config", str(config)]
        if _get("out"):
            cmd += ["--out", str(_path(_get("out")))]
        if bool(_get("hash_duplicates", False)):
            cmd += ["--hash-duplicates"]
        if _get("hash_max_files") is not None:
            cmd += ["--hash-max-files", str(int(_get("hash_max_files")))]
        return "Data Report", cmd

    if job_type == "prepare_data":
        cmd = [python, "-m", "src.scripts.prepare_data"]
        if _get("create_demo_imagefolder"):
            cmd += ["--create-demo-imagefolder", str(_path(_get("create_demo_imagefolder")))]
            cmd += ["--num-images", str(int(_get("num_images", 32)))]
            cmd += ["--img-size", str(int(_get("img_size", 64)))]
            return "Prepare Data (Demo)", cmd
        dataset = _get("dataset")
        if not dataset:
            raise ValueError("prepare_data requires args.dataset or args.create_demo_imagefolder")
        cmd += ["--dataset", str(dataset)]
        if _get("root"):
            cmd += ["--root", str(_path(_get("root")))]
        return "Prepare Data (Check)", cmd

    if job_type == "prepare_demo":
        cmd = [python, "-m", "src.scripts.prepare_data"]
        out = _get("create_demo_imagefolder")
        if not out:
            raise ValueError("prepare_demo requires args.create_demo_imagefolder")
        cmd += ["--create-demo-imagefolder", str(_path(out))]
        cmd += ["--num-images", str(int(_get("num_images", 32)))]
        cmd += ["--img-size", str(int(_get("img_size", 64)))]
        return "Prepare Data (Demo)", cmd

    if job_type == "validate_data":
        config = _path(_get("config"))
        if not config:
            raise ValueError("validate_data requires args.config")
        cmd = [python, "-m", "src.validate_data", "--config", str(config)]
        if bool(_get("use_ae", False)):
            cmd += ["--use_ae"]
        return "Validate Data", cmd

    if job_type == "sample_gan":
        checkpoint = _path(_get("checkpoint"))
        if not checkpoint:
            raise ValueError("sample_gan requires args.checkpoint")
        cmd = [python, "-m", "src.scripts.sample_gan", "--checkpoint", str(checkpoint)]
        cmd += ["--n", str(int(_get("n", 64)))]
        cmd += ["--seed", str(int(_get("seed", 42)))]
        if _get("out"):
            cmd += ["--out", str(_path(_get("out")))]
        if bool(_get("ema", False)):
            cmd += ["--ema"]
        return "Sample GAN (to file)", cmd

    if job_type == "eval_fid":
        config = _path(_get("config"))
        gen_dir = _path(_get("gen_dir"))
        if not config or not gen_dir:
            raise ValueError("eval_fid requires args.config and args.gen_dir")
        cmd = [python, "-m", "src.scripts.eval_fid", "--config", str(config), "--gen_dir", str(gen_dir)]
        if _get("max_samples") is not None:
            cmd += ["--max_samples", str(int(_get("max_samples")))]
        return "Eval FID/KID", cmd

    if job_type == "eval_gan_pipeline":
        run_dir = _path(_get("run_dir"))
        if not run_dir:
            raise ValueError("eval_gan_pipeline requires args.run_dir")
        cmd = [python, "-m", "src.scripts.eval_gan_pipeline", "--run-dir", str(run_dir)]
        if _get("checkpoint"):
            cmd += ["--checkpoint", str(_path(_get("checkpoint")))]
        if _get("out_dir"):
            cmd += ["--out-dir", str(_path(_get("out_dir")))]
        if _get("device"):
            cmd += ["--device", str(_get("device"))]
        if _get("n_images") is not None:
            cmd += ["--n-images", str(int(_get("n_images")))]
        if _get("batch_size") is not None:
            cmd += ["--batch-size", str(int(_get("batch_size")))]
        if _get("seed") is not None:
            cmd += ["--seed", str(int(_get("seed")))]
        if bool(_get("use_ema", False)):
            cmd += ["--use-ema"]
        if _get("max_samples") is not None:
            cmd += ["--max-samples", str(int(_get("max_samples")))]
        return "Eval GAN Pipeline", cmd

    raise ValueError(f"Unsupported job type: {job_type}")


def _compute_job_artifacts(job_type: str, args: dict) -> list[str]:
    """
    Best-effort list of artifact paths for the UI to browse.

    Returns absolute paths (must be within allowed roots to be accessible).
    """
    job_type = str(job_type).strip().lower()
    repo = _repo_root()

    def _abs(p: str) -> str:
        pp = Path(os.path.expanduser(str(p)))
        if not pp.is_absolute():
            pp = (repo / pp).resolve()
        else:
            pp = pp.resolve()
        return str(pp)

    out: list[str] = []

    if job_type in {"data_report"}:
        cfg = load_config(args.get("config"))
        out_dir = args.get("out") or (cfg.get("save", {}) or {}).get("out_dir") or "./outputs/data_report"
        out.append(_abs(str(out_dir)))
        return out

    if job_type in {"validate_data"}:
        cfg = load_config(args.get("config"))
        out_dir = (cfg.get("save", {}) or {}).get("out_dir") or "./outputs/validation"
        out.append(_abs(str(out_dir)))
        return out

    if job_type in {"prepare_data"}:
        if args.get("create_demo_imagefolder"):
            out.append(_abs(str(args.get("create_demo_imagefolder"))))
        return out

    if job_type in {"prepare_demo"}:
        if args.get("create_demo_imagefolder"):
            out.append(_abs(str(args.get("create_demo_imagefolder"))))
        return out

    if job_type in {"sample_gan"}:
        if args.get("out"):
            out.append(_abs(str(args.get("out"))))
        return out

    if job_type in {"train_gan"}:
        base = None
        if args.get("run_dir"):
            base = args.get("run_dir")
        else:
            cfg = None
            if args.get("config"):
                cfg = load_config(args.get("config"))
            elif args.get("resume"):
                # If resuming without a config, assume artifacts go to the config's logdir stored in ckpt.
                try:
                    import torch

                    ckpt = _load_checkpoint_file(args.get("resume"), map_location="cpu")
                    cfg = ckpt.get("cfg")
                except Exception:
                    cfg = None
            if cfg:
                base = (cfg.get("training", {}) or {}).get("logdir")
        if base:
            run_name = args.get("run_name")
            out.append(_abs(str(Path(str(base)) / str(run_name))) if run_name else _abs(str(base)))
        return out

    if job_type in {"train_ae"}:
        base = None
        if args.get("run_dir"):
            base = args.get("run_dir")
        else:
            cfg = None
            if args.get("config"):
                cfg = load_config(args.get("config"))
            elif args.get("resume"):
                try:
                    import torch

                    ckpt = _load_checkpoint_file(args.get("resume"), map_location="cpu")
                    cfg = ckpt.get("cfg")
                except Exception:
                    cfg = None
            if cfg:
                base = ((cfg.get("logging", {}) or {}).get("log_dir")) or "./logs"
        if base:
            run_name = args.get("run_name")
            out.append(_abs(str(Path(str(base)) / str(run_name))) if run_name else _abs(str(base)))
        return out

    if job_type in {"eval_fid"}:
        # Outputs are printed to logs; gen_dir is still useful to open.
        if args.get("gen_dir"):
            out.append(_abs(str(args.get("gen_dir"))))
        return out

    if job_type in {"eval_gan_pipeline"}:
        if args.get("out_dir"):
            out.append(_abs(str(args.get("out_dir"))))
        if args.get("run_dir"):
            out.append(_abs(str(args.get("run_dir"))))
        return out

    return out


@app.get("/")
async def root():
    """Basic API info and current loaded checkpoint state."""
    return {
        "service": app.title,
        "version": app.version,
        "loaded_checkpoint": _loaded_ckpt,
        "endpoints": {
            "load": ["/api/load", "/api/gan/load", "/load", "/gan/load"],
            "generate": ["/api/generate", "/api/gan/generate", "/generate", "/gan/generate"],
        },
    }


@app.get("/api/capabilities")
async def capabilities() -> dict:
    """Return supported jobs and UI form schema."""
    return capabilities_dict()


@app.get("/health")
async def health():
    """Health check."""
    svc = _service
    return {
        "status": "ok",
        "checkpoint_loaded": bool(_loaded_ckpt),
        "generator_ready": bool(svc and svc.G is not None),
    }


@app.post("/api/load", response_model=LoadResponse)
@app.post("/api/gan/load", response_model=LoadResponse)
@app.post("/load", response_model=LoadResponse)
@app.post("/gan/load", response_model=LoadResponse)
async def load_checkpoint(req: LoadRequest) -> LoadResponse:
    """
    Load a GAN checkpoint on the server.

    The client typically calls this once, then calls `/generate` repeatedly.
    """
    global _service, _loaded_ckpt

    ckpt_path = _resolve_path(req.ckpt)
    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt_path}")

    # Re-create the service if a device override is provided.
    if req.device:
        _service = GANService(device=req.device)
    svc = _get_service()

    try:
        svc.load_checkpoint(ckpt_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {e}") from e

    _loaded_ckpt = ckpt_path
    cfg = svc.cfg or {}

    return LoadResponse(
        success=True,
        message="Checkpoint loaded",
        ckpt=_loaded_ckpt,
        device=str(svc.device),
        img_size=int(cfg.get("img_size")) if cfg.get("img_size") is not None else None,
        img_channels=int(cfg.get("img_channels"))
        if cfg.get("img_channels") is not None
        else None,
        latent_dim=int(cfg.get("latent_dim"))
        if cfg.get("latent_dim") is not None
        else None,
        has_ema_shadow=bool(svc.has_ema_shadow),
    )


@app.post("/api/generate")
@app.post("/api/gan/generate")
@app.post("/generate")
@app.post("/gan/generate")
async def generate(req: GenerateRequest):
    """
    Generate a PNG image grid from the currently loaded checkpoint.

    Returns:
        `image/png` bytes.
    """
    svc = _service
    if svc is None or svc.G is None:
        raise HTTPException(
            status_code=400, detail="No checkpoint loaded. Call POST /api/load first."
        )

    try:
        img = svc.generate_grid(
            GenerateParams(
                n=req.n,
                seed=req.seed,
                nrow=req.nrow,
                use_ema_shadow=req.use_ema,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/jobs", response_model=list[JobInfo])
async def list_jobs() -> list[JobInfo]:
    jm = _get_jobs()
    return [_job_to_info(j) for j in jm.list_jobs()]


@app.post("/api/jobs/start", response_model=JobInfo)
async def start_job(req: JobStartRequest) -> JobInfo:
    jm = _get_jobs()
    try:
        name, cmd = _build_job_command(req.type, dict(req.args))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        job = jm.start(name=name, cmd=cmd, cwd=_repo_root())
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    try:
        job.artifacts = _compute_job_artifacts(req.type, dict(req.args))  # type: ignore[attr-defined]
    except Exception:
        job.artifacts = []  # type: ignore[attr-defined]
    return _job_to_info(job)


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str) -> JobInfo:
    jm = _get_jobs()
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_info(job)


@app.get("/api/jobs/{job_id}/manifest")
async def get_job_manifest(job_id: str) -> dict:
    jm = _get_jobs()
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    p = Path(job.job_dir) / "manifest.json"
    if not p.exists():
        return {"artifacts": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read manifest: {e}") from e


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """
    Server-sent events stream for a job.

    Events:
    - `job`: current JobInfo
    - `log`: {"lines": [...]} incremental log lines
    - `manifest`: manifest.json content (if present)
    """
    from starlette.responses import StreamingResponse
    import asyncio

    jm = _get_jobs()
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def gen():
        last_pos = 0
        last_sent_status = None
        metrics_pos = 0
        metrics_path = None
        while True:
            j = jm.get(job_id)
            if j is None:
                break

            info = _job_to_info(j).model_dump()
            st = info.get("status")
            if st != last_sent_status:
                last_sent_status = st
                yield f"event: job\ndata: {json.dumps(info, ensure_ascii=False)}\n\n"

            # tail log
            try:
                lp = Path(j.log_path)
                if lp.exists():
                    with open(lp, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_pos)
                        new = f.read()
                        last_pos = f.tell()
                    if new:
                        lines = new.splitlines()
                        payload = {"lines": lines[-200:]}
                        yield f"event: log\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception:
                pass

            # tail metrics.jsonl (best-effort, for training runs)
            try:
                if metrics_path is None:
                    # Heuristic: search a few artifact directories for metrics.jsonl
                    candidates = []
                    for p in list(getattr(j, "artifacts", []) or []):
                        candidates.append(Path(p) / "metrics.jsonl")
                    # also check job dir (rare)
                    candidates.append(Path(j.job_dir) / "metrics.jsonl")
                    for c in candidates:
                        if c.exists() and c.is_file():
                            metrics_path = c
                            break

                if metrics_path is not None and metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(metrics_pos)
                        new = f.read()
                        metrics_pos = f.tell()
                    if new:
                        entries = []
                        for raw in new.splitlines():
                            raw = raw.strip()
                            if not raw:
                                continue
                            try:
                                entries.append(json.loads(raw))
                            except Exception:
                                continue
                        if entries:
                            yield f"event: metrics\ndata: {json.dumps({'entries': entries[-50:]}, ensure_ascii=False)}\n\n"
            except Exception:
                pass

            # manifest when done
            if st in {"succeeded", "failed", "canceled"}:
                try:
                    mp = Path(j.job_dir) / "manifest.json"
                    if mp.exists():
                        payload = json.loads(mp.read_text(encoding="utf-8"))
                        yield f"event: manifest\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                except Exception:
                    pass
                yield f"event: done\ndata: {json.dumps({'status': st}, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, tail: int = 200) -> dict:
    jm = _get_jobs()
    if jm.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"lines": jm.read_logs(job_id, tail=int(tail))}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    jm = _get_jobs()
    ok = jm.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Job not cancelable")
    return {"success": True}


@app.get("/api/fs/list")
async def fs_list(path: str = ".") -> dict:
    fs = _get_fs()
    try:
        return fs.list_dir(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/fs/read")
async def fs_read(path: str, max_bytes: int = 200_000) -> dict:
    fs = _get_fs()
    try:
        return fs.read_text(path, max_bytes=int(max_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/fs/file")
async def fs_file(path: str) -> Response:
    fs = _get_fs()
    try:
        p = fs.resolve(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")
        media_type = fs.guess_mime(str(p))
        return FileResponse(str(p), media_type=media_type, filename=p.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class FSWriteRequest(BaseModel):
    path: str
    text: str
    overwrite: bool = False


class FSMkdirRequest(BaseModel):
    path: str
    parents: bool = True
    exist_ok: bool = True


@app.post("/api/fs/write")
async def fs_write(req: FSWriteRequest) -> dict:
    fs = _get_fs()
    try:
        # Only allow editing text-y files by extension (local safety).
        p = Path(req.path)
        if p.suffix.lower() not in {".yaml", ".yml", ".json", ".jsonl", ".txt", ".md", ".log"}:
            raise ValueError("Only text config/log formats are allowed to be written.")
        return fs.write_text(req.path, req.text, overwrite=bool(req.overwrite))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/fs/mkdir")
async def fs_mkdir(req: FSMkdirRequest) -> dict:
    fs = _get_fs()
    try:
        return fs.mkdir(req.path, parents=bool(req.parents), exist_ok=bool(req.exist_ok))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class ConfigValidateRequest(BaseModel):
    """Validate YAML config text (syntax + basic expected keys)."""

    kind: str = Field("auto", description="auto | gan | ae")
    text: str = Field(..., description="YAML text to validate.")


@app.post("/api/config/validate")
async def validate_config(req: ConfigValidateRequest) -> dict:
    import yaml

    try:
        obj = yaml.safe_load(req.text) or {}
        if not isinstance(obj, dict):
            raise ValueError("Top-level YAML must be a mapping/dict.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YAML parse error: {e}") from e

    kind = str(req.kind).strip().lower()
    if kind == "auto":
        # Heuristic: configs/gan typically have training.lr_g/lr_d; dataset configs have train/training.lr.
        training = obj.get("training") or obj.get("train") or {}
        if isinstance(training, dict) and ("lr_g" in training or "lr_d" in training):
            kind = "gan"
        else:
            kind = "ae"

    warnings: list[str] = []
    model_type = None

    def _require_keys(section: str, keys: list[str]) -> None:
        val = obj.get(section)
        if not isinstance(val, dict):
            raise ValueError(f"Missing or invalid '{section}' section (expected mapping).")
        missing = [k for k in keys if k not in val]
        if missing:
            raise ValueError(f"Missing keys in '{section}': {missing}")

    try:
        if kind == "gan":
            _require_keys("model", ["type", "img_size", "img_channels", "latent_dim", "g_channels", "d_channels"])
            _require_keys("data", ["dataset", "root", "batch_size"])
            _require_keys("training", ["epochs", "logdir", "lr_g", "lr_d"])
            model_type = str((obj.get("model") or {}).get("type", "")).lower()
        elif kind == "ae":
            _require_keys("data", ["dataset", "root", "batch_size"])
            _require_keys("model", ["type"])
            # allow either `training` or legacy `train`
            if "training" not in obj and "train" not in obj:
                raise ValueError("Missing 'training' section (or legacy 'train').")
            model_type = str((obj.get("model") or {}).get("type", "")).lower()
        else:
            raise ValueError(f"Unknown kind: {kind}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if "train" in obj and "training" not in obj:
        warnings.append("Config uses legacy `train:`; prefer `training:`.")

    return {
        "ok": True,
        "kind": kind,
        "model_type": model_type,
        "is_wgan_gp": bool(kind == "gan" and model_type == "wgan-gp"),
        "top_level_keys": sorted(list(obj.keys())),
        "warnings": warnings,
    }


class ConfigOverrideItem(BaseModel):
    path: str = Field(..., description="Dot-path like 'training.epochs' or 'data.batch_size'")
    value: str = Field(..., description="Value as string; interpreted by `type`.")
    type: str = Field("auto", description="auto | string | int | float | bool | json")


class ConfigApplyOverridesRequest(BaseModel):
    text: str = Field(..., description="Base YAML config text.")
    overrides: list[ConfigOverrideItem] = Field(default_factory=list)


@app.post("/api/config/apply_overrides")
async def apply_overrides(req: ConfigApplyOverridesRequest) -> dict:
    """
    Apply a set of dot-path overrides to YAML text and return patched YAML.

    This is intended for the Configs UI so users can change common knobs without
    hand-editing YAML.
    """
    import json as _json
    import yaml

    try:
        obj = yaml.safe_load(req.text) or {}
        if not isinstance(obj, dict):
            raise ValueError("Top-level YAML must be a mapping/dict.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YAML parse error: {e}") from e

    def _parse_value(item: ConfigOverrideItem):
        t = str(item.type).strip().lower()
        raw = item.value
        if t == "string":
            return str(raw)
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            s = str(raw).strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"Invalid bool: {raw}")
        if t == "json":
            return _json.loads(raw)
        # auto: try yaml scalar parsing
        try:
            v = yaml.safe_load(raw)
            return v
        except Exception:
            return raw

    def _set_path(d: dict, path: str, value):
        parts = [p for p in str(path).split(".") if p]
        if not parts:
            raise ValueError("Empty override path.")
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    applied = []
    for item in req.overrides:
        try:
            val = _parse_value(item)
            _set_path(obj, item.path, val)
            applied.append({"path": item.path, "value": val})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed override {item.path}: {e}") from e

    patched = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    return {"ok": True, "patched": patched, "applied": applied}


class ConfigApplyOverlayRequest(BaseModel):
    base_text: str
    overlay_text: str


class ConfigApplyOverlayPathRequest(BaseModel):
    base_text: str
    overlay_path: str


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v


@app.post("/api/config/apply_overlay")
async def apply_overlay(req: ConfigApplyOverlayRequest) -> dict:
    import yaml

    try:
        base = yaml.safe_load(req.base_text) or {}
        overlay = yaml.safe_load(req.overlay_text) or {}
        if not isinstance(base, dict) or not isinstance(overlay, dict):
            raise ValueError("Both base_text and overlay_text must be YAML dicts.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YAML parse error: {e}") from e

    _deep_update(base, overlay)
    patched = yaml.safe_dump(base, sort_keys=False, allow_unicode=True)
    return {"ok": True, "patched": patched}


@app.post("/api/config/apply_overlay_path")
async def apply_overlay_path(req: ConfigApplyOverlayPathRequest) -> dict:
    fs = _get_fs()
    try:
        overlay = fs.read_text(req.overlay_path, max_bytes=500_000)["text"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read overlay: {e}") from e
    return await apply_overlay(ConfigApplyOverlayRequest(base_text=req.base_text, overlay_text=overlay))


@app.get("/api/config/overlays")
async def list_overlays(dir: str = "./.ai_cache/configs/overrides") -> dict:
    """
    List overlay files under a directory (default: ./.ai_cache/configs/overrides).
    """
    fs = _get_fs()
    listing = fs.list_dir(dir)
    overlays = [
        e
        for e in listing["entries"]
        if e.get("type") == "file"
        and str(e.get("name", "")).lower().endswith((".yaml", ".yml", ".json"))
    ]
    return {"dir": listing["path"], "overlays": overlays}


@app.get("/api/runs")
async def list_runs(limit: int = 200) -> dict:
    runs = scan_runs(_repo_root(), max_runs=int(limit))
    return {
        "runs": [
            {
                "id": r.id,
                "path": r.path,
                "created_at": r.created_at,
                "script": r.script,
                "metrics_path": r.metrics_path,
                "config_path": r.config_path,
                "notes": _read_run_notes(Path(r.path)),
            }
            for r in runs
        ]
    }


@app.get("/api/runs/{run_id:path}")
async def run_detail(run_id: str, tail_metrics: int = 200) -> dict:
    rd = get_run(_repo_root(), run_id)
    if rd is None:
        raise HTTPException(status_code=404, detail="Run not found")
    out = load_run_detail(rd, tail_metrics=int(tail_metrics))
    out["notes"] = _read_run_notes(rd)
    return out


class RunNotesRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)
    note: str = ""


@app.get("/api/runs/{run_id:path}/notes")
async def get_run_notes(run_id: str) -> dict:
    rd = get_run(_repo_root(), run_id)
    if rd is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _read_run_notes(rd)


@app.post("/api/runs/{run_id:path}/notes")
async def set_run_notes(run_id: str, req: RunNotesRequest) -> dict:
    rd = get_run(_repo_root(), run_id)
    if rd is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _write_run_notes(rd, tags=list(req.tags), note=str(req.note))


@app.post("/api/runs/{run_id:path}/clone_config")
async def clone_run_config(run_id: str, dest: Optional[str] = None) -> dict:
    """
    Copy a run's config_resolved.yaml into AI cache for easy re-experimentation.
    """
    rd = get_run(_repo_root(), run_id)
    if rd is None:
        raise HTTPException(status_code=404, detail="Run not found")
    cfg_path = rd / "config_resolved.yaml"
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="config_resolved.yaml not found for this run")
    fs = _get_fs()
    base_name = run_id.replace("/", "_").replace("\\", "_")
    target = dest or f"./.ai_cache/configs/clone_{base_name}.yaml"
    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    fs.mkdir("./.ai_cache/configs", parents=True, exist_ok=True)
    return fs.write_text(target, text, overwrite=True)


@app.get("/api/runs/compare")
async def compare_runs(run1: str, run2: str, format: str = "json", tail_metrics: int = 500) -> Response:
    """
    Compare two runs and return either JSON or Markdown.
    """
    r1 = get_run(_repo_root(), run1)
    r2 = get_run(_repo_root(), run2)
    if r1 is None or r2 is None:
        raise HTTPException(status_code=404, detail="Run not found")

    d1 = load_run_detail(r1, tail_metrics=int(tail_metrics))
    d2 = load_run_detail(r2, tail_metrics=int(tail_metrics))
    n1 = _read_run_notes(r1)
    n2 = _read_run_notes(r2)

    def _last_numeric(metrics: list[dict], key: str):
        for m in reversed(metrics):
            v = m.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    keys = ["fid", "kid_mean", "g_loss", "d_loss", "train_loss", "val_loss", "val_psnr", "val_ssim"]
    summary = {
        "run1": {"id": run1, "path": str(r1), "notes": n1},
        "run2": {"id": run2, "path": str(r2), "notes": n2},
        "metrics": {
            k: {"run1": _last_numeric(d1.get("metrics_tail", []), k), "run2": _last_numeric(d2.get("metrics_tail", []), k)}
            for k in keys
        },
    }

    if str(format).lower() in {"md", "markdown"}:
        lines = []
        lines.append(f"# Run Compare\n")
        lines.append(f"- run1: `{run1}`")
        lines.append(f"- run2: `{run2}`\n")
        lines.append("## Notes\n")
        lines.append(f"- run1 tags: {', '.join(n1.get('tags', [])) or '(none)'}")
        lines.append(f"- run2 tags: {', '.join(n2.get('tags', [])) or '(none)'}\n")
        lines.append("## Metrics (last seen)\n")
        lines.append("| key | run1 | run2 |")
        lines.append("| --- | ---: | ---: |")
        for k in keys:
            a = summary["metrics"][k]["run1"]
            b = summary["metrics"][k]["run2"]
            a_s = "" if a is None else f"{a:.6g}"
            b_s = "" if b is None else f"{b:.6g}"
            lines.append(f"| {k} | {a_s} | {b_s} |")
        return Response(content="\n".join(lines), media_type="text/markdown")

    return JSONResponse(content=summary)
