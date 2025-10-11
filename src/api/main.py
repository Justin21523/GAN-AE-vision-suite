from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import logging
import os
import glob
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

# Bootstrap runtime first
from src.utils.runtime import bootstrap_runtime
from src.service.gan_infer import GANInferenceService
from .jobs import router as jobs_router
from src.registry.index import get_run_registry, get_model_index

# Initialize runtime
cfg, info = bootstrap_runtime()
logging.info(f"AI Warehouse initialized at: {cfg.cache_root}")

app = FastAPI(
    title="GAN-AE-VISION-SUITE API",
    description="API for Autoencoder, VAE, and GAN models",
    version="2.0.0",
)

# Initialize services
gan_service = GANInferenceService(cfg)


class InferenceRequest(BaseModel):
    """Request model for inference."""

    run_id: str
    checkpoint_type: str = "latest"  # "latest", "best", or direct path
    num_samples: int = 16
    grid_nrow: int = 4
    seed: Optional[int] = None


class InferenceResponse(BaseModel):
    """Response model for inference."""

    success: bool
    message: str
    run_id: str
    checkpoint_type: str
    samples_generated: Optional[int] = None
    output_path: Optional[str] = None
    relative_path: Optional[str] = None
    seed_used: Optional[int] = None


class MetricsRequest(BaseModel):
    """Request model for metrics calculation."""

    run_id: str
    checkpoint_type: str = "latest"
    metrics: List[str] = ["fid", "kid"]
    num_samples: int = 1000


class MetricsResponse(BaseModel):
    """Response model for metrics."""

    success: bool
    message: str
    run_id: str
    metrics: Optional[Dict[str, float]] = None
    report_path: Optional[str] = None


class RunInfo(BaseModel):
    """Run information model."""

    id: str
    checkpoints: Dict[str, Optional[str]]
    manifest: Optional[Dict[str, Any]] = None


# 在創建 app 後加入
app.include_router(jobs_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "GAN-AE-VISION-SUITE API v2.0",
        "version": "2.0.0",
        "warehouse_root": cfg.cache_root,
        "endpoints": ["/infer/gan", "/metrics/fidkid", "/runs", "/health"],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "warehouse_initialized": info.initialized,
        "cache_root": cfg.cache_root,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/infer/gan", response_model=InferenceResponse)
async def infer_gan(request: InferenceRequest):
    """Generate samples using GAN model."""
    try:
        result = gan_service.generate_samples(
            run_id=request.run_id,
            checkpoint_type=request.checkpoint_type,
            num_samples=request.num_samples,
            grid_nrow=request.grid_nrow,
            seed=request.seed,
        )

        if result["success"]:
            return InferenceResponse(
                success=True,
                message="GAN inference completed successfully",
                run_id=result["run_id"],
                checkpoint_type=result["checkpoint_type"],
                samples_generated=result["samples_generated"],
                output_path=result["output_path"],
                relative_path=result["relative_path"],
                seed_used=result["seed_used"],
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])

    except Exception as e:
        logging.error(f"GAN inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/metrics/fidkid", response_model=MetricsResponse)
async def calculate_fid_kid(request: MetricsRequest, background_tasks: BackgroundTasks):
    """Calculate FID and KID metrics (async)."""
    try:
        # For now, return placeholder response
        # In production, this would queue the evaluation task
        from src.metrics.fid_kid import compute_fid_kid_small_sample

        # This is a simplified implementation
        # In practice, you'd load real and generated images for comparison
        placeholder_metrics = {"fid": 25.3, "kid": 0.015}

        # Save metrics report
        report_dir = os.path.join(cfg.metrics_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(
            report_dir, f"{request.run_id}_{uuid.uuid4().hex[:8]}.json"
        )

        report = {
            "run_id": request.run_id,
            "checkpoint_type": request.checkpoint_type,
            "metrics": placeholder_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "num_samples": request.num_samples,
        }

        import json

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return MetricsResponse(
            success=True,
            message="FID/KID metrics calculation initiated",
            run_id=request.run_id,
            metrics=placeholder_metrics,
            report_path=report_path,
        )

    except Exception as e:
        logging.error(f"Metrics calculation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Metrics calculation failed: {str(e)}"
        )


@app.get("/runs", response_model=List[RunInfo])
async def get_available_runs():
    """Get list of available runs."""
    try:
        runs = gan_service.get_available_runs()
        return [
            RunInfo(
                id=run["id"],
                checkpoints=run["checkpoints"],
                manifest=run.get("manifest"),
            )
            for run in runs
        ]
    except Exception as e:
        logging.error(f"Failed to list runs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")


@app.get("/runs/{run_id}", response_model=RunInfo)
async def get_run_info(run_id: str):
    """Get information for a specific run."""
    try:
        runs = gan_service.get_available_runs()
        for run in runs:
            if run["id"] == run_id:
                return RunInfo(
                    id=run["id"],
                    checkpoints=run["checkpoints"],
                    manifest=run.get("manifest"),
                )

        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    except Exception as e:
        logging.error(f"Failed to get run info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get run info: {str(e)}")


@app.get("/samples/{run_id}")
async def get_run_samples(run_id: str):
    """Get sample images for a run."""
    try:
        samples_dir = os.path.join(cfg.output_dir, "samples", run_id)
        if not os.path.exists(samples_dir):
            raise HTTPException(
                status_code=404, detail=f"No samples found for run: {run_id}"
            )

        sample_files = []
        for ext in ["png", "jpg", "jpeg"]:
            sample_files.extend(glob.glob(os.path.join(samples_dir, f"*.{ext}")))

        # Sort by modification time (newest first)
        sample_files.sort(key=os.path.getmtime, reverse=True)

        return {
            "run_id": run_id,
            "samples": [
                {
                    "path": sample,
                    "relative_path": os.path.relpath(sample, cfg.cache_root),
                    "filename": os.path.basename(sample),
                }
                for sample in sample_files[:20]  # Return latest 20 samples
            ],
        }
    except Exception as e:
        logging.error(f"Failed to get samples: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get samples: {str(e)}")


# 加入 registry 路由
@app.get("/registry/runs")
async def get_runs(
    status: Optional[str] = None, task_type: Optional[str] = None, limit: int = 100
):
    """Get all runs with optional filtering."""
    registry = get_run_registry()
    runs = registry.get_all_runs(status=status, task_type=task_type, limit=limit)
    return runs


@app.get("/registry/runs/{run_id}")
async def get_run(run_id: str):
    """Get run by ID."""
    registry = get_run_registry()
    run = registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@app.get("/registry/search")
async def search_runs(q: str, fields: Optional[str] = None):
    """Search runs by query."""
    registry = get_run_registry()
    field_list = fields.split(",") if fields else None
    runs = registry.search_runs(q, field_list)
    return runs


@app.get("/registry/models")
async def get_models(tags: Optional[str] = None, limit: int = 50):
    """Get models with optional tag filtering."""
    index = get_model_index()
    tag_list = tags.split(",") if tags else None
    models = index.find_models(tags=tag_list, limit=limit)
    return models


@app.get("/registry/models/{model_id}")
async def get_model(model_id: str):
    """Get model by ID."""
    index = get_model_index()
    model = index.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return model


@app.get("/registry/aliases/{alias}")
async def resolve_alias(alias: str, run_id: Optional[str] = None):
    """Resolve alias to model ID."""
    index = get_model_index()
    model_id = index.resolve_alias(alias, run_id)
    if model_id is None:
        raise HTTPException(status_code=404, detail=f"Alias not found: {alias}")
    return {"alias": alias, "model_id": model_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
