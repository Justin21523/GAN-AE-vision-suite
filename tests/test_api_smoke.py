"""
FastAPI smoke tests.

These tests validate that the minimal API imports and responds to basic routes
without requiring any checkpoints or GPU.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch
from fastapi import HTTPException

from src.api.main import _resolve_path, generate, health, GenerateRequest


def test_health_route():
    body = asyncio.run(health())
    assert body["status"] == "ok"
    assert body["checkpoint_loaded"] is False


def test_generate_requires_load():
    try:
        asyncio.run(
            generate(GenerateRequest(n=4, nrow=2, seed=123, use_ema=False))
        )
        raise AssertionError("Expected HTTPException")
    except HTTPException as e:
        assert e.status_code == 400
        assert "No checkpoint loaded" in str(e.detail)


def test_resolve_path_repo_relative():
    out = _resolve_path("logs/example.pt")
    p = Path(out)
    assert p.is_absolute()
    assert p.as_posix().endswith("logs/example.pt")


def test_job_command_validation():
    from src.api.main import _build_job_command

    name, cmd = _build_job_command("data_report", {"config": "configs/dataset_mnist.yaml"})
    assert "Data Report" in name
    assert cmd[:3] == ["python", "-m", "src.scripts.data_report"]


def test_capabilities_has_jobs():
    from src.api.capabilities import as_dict

    caps = as_dict()
    types = {j["type"] for j in caps["jobs"]}
    assert "train_gan" in types
    assert "data_report" in types
    assert "eval_gan_pipeline" in types


def test_validate_config_includes_model_flags():
    import asyncio

    from src.api.main import validate_config, ConfigValidateRequest

    txt = """
model:
  type: wgan-gp
  img_size: 32
  img_channels: 3
  latent_dim: 16
  g_channels: [64, 32]
  d_channels: [32, 64]
data:
  dataset: imagefolder
  root: ./data
  batch_size: 4
training:
  epochs: 1
  logdir: ./logs/x
  lr_g: 1.0e-4
  lr_d: 1.0e-4
"""
    out = asyncio.run(validate_config(ConfigValidateRequest(kind="gan", text=txt)))
    assert out["model_type"] == "wgan-gp"
    assert out["is_wgan_gp"] is True


def test_config_validate_ok():
    import asyncio

    from src.api.main import validate_config, ConfigValidateRequest

    # minimal GAN config (required keys only)
    txt = """
model:
  type: wgan-gp
  img_size: 32
  img_channels: 3
  latent_dim: 16
  g_channels: [64, 32]
  d_channels: [32, 64]
data:
  dataset: imagefolder
  root: ./data
  batch_size: 4
training:
  epochs: 1
  logdir: ./logs/x
  lr_g: 1.0e-4
  lr_d: 1.0e-4
"""
    out = asyncio.run(validate_config(ConfigValidateRequest(kind="gan", text=txt)))
    assert out["ok"] is True
    assert out["kind"] == "gan"


def test_config_validate_auto_detects_ae():
    import asyncio

    from src.api.main import validate_config, ConfigValidateRequest

    txt = """
data:
  dataset: imagefolder
  root: ./data
  batch_size: 4
model:
  type: ae
training:
  epochs: 1
  lr: 1.0e-3
  save_dir: ./logs/checkpoints
"""
    out = asyncio.run(validate_config(ConfigValidateRequest(kind="auto", text=txt)))
    assert out["ok"] is True
    assert out["kind"] == "ae"


def test_config_apply_overrides():
    import asyncio

    from src.api.main import apply_overrides, ConfigApplyOverridesRequest, ConfigOverrideItem

    base = "data:\n  batch_size: 4\ntraining:\n  epochs: 1\n"
    req = ConfigApplyOverridesRequest(
        text=base,
        overrides=[
            ConfigOverrideItem(path="data.batch_size", value="8", type="int"),
            ConfigOverrideItem(path="training.epochs", value="3", type="int"),
        ],
    )
    out = asyncio.run(apply_overrides(req))
    assert out["ok"] is True
    assert "batch_size: 8" in out["patched"]
    assert "epochs: 3" in out["patched"]


def test_config_apply_overlay():
    import asyncio

    from src.api.main import apply_overlay, ConfigApplyOverlayRequest

    base = "a:\n  b: 1\n"
    overlay = "a:\n  c: 2\n"
    out = asyncio.run(apply_overlay(ConfigApplyOverlayRequest(base_text=base, overlay_text=overlay)))
    assert out["ok"] is True
    assert "b: 1" in out["patched"]
    assert "c: 2" in out["patched"]


def test_fs_resolve_rejects_outside(tmp_path: Path):
    from src.service.fs import SafeFS

    fs = SafeFS([tmp_path])
    try:
        fs.resolve("/etc/passwd")
        raise AssertionError("Expected PermissionError")
    except PermissionError:
        pass


def test_fs_list_dir(tmp_path: Path):
    from src.service.fs import SafeFS

    (tmp_path / "a").mkdir()
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    fs = SafeFS([tmp_path])
    out = fs.list_dir(str(tmp_path))
    names = [e["name"] for e in out["entries"]]
    assert "a" in names
    assert "hello.txt" in names


def test_fs_write_text(tmp_path: Path):
    from src.service.fs import SafeFS

    fs = SafeFS([tmp_path])
    out = fs.write_text(str(tmp_path / "x.yaml"), "a: 1\n", overwrite=True)
    assert (tmp_path / "x.yaml").exists()
    assert out["size"] > 0


def test_run_notes_compare_and_clone(tmp_path: Path, monkeypatch):
    import json

    import src.api.main as api_main

    # Use an isolated repo root for this test.
    monkeypatch.setattr(api_main, "_repo_root", lambda: tmp_path)
    monkeypatch.setenv("AI_CACHE_ROOT", str(tmp_path / ".ai_cache"))
    api_main._fs = None
    api_main._jobs = None

    (tmp_path / "logs" / "r1").mkdir(parents=True)
    (tmp_path / "logs" / "r2").mkdir(parents=True)

    (tmp_path / "logs" / "r1" / "meta.json").write_text(
        json.dumps({"created_at": "20250101-000000", "script": "train_gan"}),
        encoding="utf-8",
    )
    (tmp_path / "logs" / "r2" / "meta.json").write_text(
        json.dumps({"created_at": "20250101-000001", "script": "train_gan"}),
        encoding="utf-8",
    )

    (tmp_path / "logs" / "r1" / "config_resolved.yaml").write_text("a: 1\n", encoding="utf-8")
    (tmp_path / "logs" / "r2" / "config_resolved.yaml").write_text("a: 2\n", encoding="utf-8")

    (tmp_path / "logs" / "r1" / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "train_loss": 1.0}) + "\n" + json.dumps({"step": 2, "train_loss": 0.5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "logs" / "r2" / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "train_loss": 1.2}) + "\n" + json.dumps({"step": 2, "train_loss": 0.6}) + "\n",
        encoding="utf-8",
    )

    runs = asyncio.run(api_main.list_runs(limit=10))
    assert runs["runs"]
    assert runs["runs"][0]["notes"]["tags"] == []

    saved = asyncio.run(
        api_main.set_run_notes("logs/r1", api_main.RunNotesRequest(tags=["baseline", "wgangp"], note="ok"))
    )
    assert "updated_at" in saved
    loaded = asyncio.run(api_main.get_run_notes("logs/r1"))
    assert loaded["tags"] == ["baseline", "wgangp"]
    assert loaded["note"] == "ok"

    clone = asyncio.run(api_main.clone_run_config("logs/r1", dest=None))
    assert "path" in clone
    assert (tmp_path / ".ai_cache" / "configs" / "clone_logs_r1.yaml").exists()

    md = asyncio.run(api_main.compare_runs(run1="logs/r1", run2="logs/r2", format="markdown", tail_metrics=50))
    assert md.media_type == "text/markdown"
    body = md.body.decode("utf-8", errors="replace")
    assert "| train_loss |" in body


def test_gan_checkpoint_api_roundtrip(tmp_path: Path):
    import src.api.main as api_main
    from src.api.main import LoadRequest
    from src.models.gan.generator import DCGANGenerator
    from src.service.gan_infer import GANService, GenerateParams
    from src.utils.checkpoint import checkpoint_payload, load_checkpoint

    cfg = {
        "model": {
            "type": "wgan-gp",
            "arch": "dcgan",
            "img_size": 32,
            "img_channels": 3,
            "latent_dim": 8,
            "g_channels": [32, 16, 8],
            "d_channels": [8, 16, 32],
        },
        "data": {"dataset": "imagefolder", "root": str(tmp_path), "batch_size": 2},
        "training": {"epochs": 1, "logdir": str(tmp_path), "lr_g": 1e-4, "lr_d": 1e-4},
    }
    g = DCGANGenerator(latent_dim=8, img_channels=3, channels=(32, 16, 8), img_size=32)
    ckpt_path = tmp_path / "ckpt_epoch1.pt"
    torch.save(checkpoint_payload({"G": g.state_dict(), "cfg": cfg}), ckpt_path)

    loaded = load_checkpoint(ckpt_path)
    assert type(loaded["cfg"]) is dict

    svc = GANService(device="cpu")
    svc.load_checkpoint(str(ckpt_path))
    img = svc.generate_grid(GenerateParams(n=4, nrow=2, seed=1))
    assert img.size == (64, 64)

    async def _load():
        return await api_main.load_checkpoint(LoadRequest(ckpt=str(ckpt_path), device="cpu"))

    resp = asyncio.run(_load())
    assert resp.success is True
    assert resp.img_size == 32
