"""
End-to-end evaluation pipeline for a trained GAN run.

What it does:
- Loads a training checkpoint (optionally EMA weights if present).
- Generates N images into an output directory (one file per image).
- Computes FID/KID vs the real validation split using the run's resolved config.
- Writes `eval_result.json` and appends a record to the run's `metrics.jsonl`.

This is intended for local UI usage (jobs runner) to make "evaluate this run"
one click.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.datasets import build_dataset
from src.data.transforms import build_transforms
from src.metrics.fidkid import FIDKID
from src.models.gan.factory import build_generator_from_cfg
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.run import JSONLMetricsWriter
from src.utils.vision import save_image_grid, to_pil_rgb


def _find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    ckpts = list(run_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        return None

    def _epoch(p: Path) -> int:
        m = re.search(r"ckpt_epoch(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_epoch)
    return ckpts[-1]


@torch.no_grad()
def _generate_images(
    ckpt_path: Path,
    out_dir: Path,
    n_images: int,
    batch_size: int,
    seed: int,
    device: str,
    use_ema: bool,
) -> None:
    ckpt = load_checkpoint(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("cfg") or {}
    mcfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}
    if not mcfg:
        raise SystemExit(f"Checkpoint missing cfg.model: {ckpt_path}")

    G = build_generator_from_cfg(mcfg).to(device)
    G.load_state_dict(ckpt["G"])
    if bool(use_ema) and isinstance(ckpt.get("ema_shadow"), dict) and ckpt.get("ema_shadow"):
        G.load_state_dict(ckpt["ema_shadow"], strict=False)
    G.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))

    bs = max(1, int(batch_size))
    total = int(n_images)
    latent_dim = int(mcfg["latent_dim"])

    written = 0
    for i0 in range(0, total, bs):
        cur = min(bs, total - i0)
        z = torch.randn(cur, latent_dim, device=device)
        x = G(z)  # NCHW in [-1,1]
        x01 = (x * 0.5 + 0.5).clamp(0, 1).detach().cpu()
        for j in range(cur):
            idx = written
            written += 1
            img = to_pil_rgb(x01[j])
            img.save(str(out_dir / f"{idx:06d}.png"), format="PNG")

    # Convenience preview
    grid_n = min(64, total)
    if grid_n > 0:
        torch.manual_seed(int(seed))
        z = torch.randn(grid_n, latent_dim, device=device)
        x = G(z)
        save_image_grid(x.detach().cpu(), out_dir / "generated_grid.png", nrow=8, value_range=(-1, 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True, help="run directory (e.g. logs/gan_xxx)")
    p.add_argument("--checkpoint", type=str, default="", help="optional explicit checkpoint path")
    p.add_argument("--out-dir", type=str, default="", help="optional output directory (defaults under run-dir)")
    p.add_argument("--n-images", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--max-samples", type=int, default=10000, help="max real/fake samples used for FID/KID")
    args = p.parse_args()

    run_dir = Path(os.path.expanduser(args.run_dir)).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config_resolved.yaml in run dir: {run_dir}")
    cfg = load_config(str(cfg_path))

    if args.checkpoint:
        ckpt_path = Path(os.path.expanduser(args.checkpoint)).resolve()
    else:
        ckpt_path = _find_latest_ckpt(run_dir) or Path()
    if not ckpt_path.exists():
        raise SystemExit("No checkpoint found (provide --checkpoint or ensure ckpt_epoch*.pt exists)")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve() if args.out_dir else (run_dir / "eval" / f"gen_{ts}")

    _generate_images(
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        n_images=int(args.n_images),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=str(device),
        use_ema=bool(args.use_ema),
    )

    metric = FIDKID(device=str(device))
    if not metric.enabled:
        result = {
            "ok": False,
            "reason": "torchmetrics not installed; skip FID/KID",
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "out_dir": str(out_dir),
            "created_at": ts,
        }
        (out_dir / "eval_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(result)
        return

    # Real features (val split)
    val_ds = build_dataset(cfg, train=False)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4)
    with torch.no_grad():
        for b in val_loader:
            metric.update_real(b[0].to(device))

    # Fake features (read generated images back through the same val transforms)
    to_tensor = build_transforms(cfg, train=False)
    files = sorted([p for p in out_dir.glob("*.png") if p.name != "generated_grid.png"])
    files = files[: int(args.max_samples)]
    for fp in files:
        img = Image.open(fp).convert("RGB")
        t = to_tensor(img).unsqueeze(0).to(device)  # type: ignore
        metric.update_fake(t)

    scores = metric.compute()
    result = {
        "ok": True,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "out_dir": str(out_dir),
        "n_images": int(args.n_images),
        "max_samples": int(args.max_samples),
        "use_ema": bool(args.use_ema),
        "device": str(device),
        "fid": float(scores["fid"]),
        "kid_mean": float(scores["kid_mean"]),
        "created_at": ts,
    }
    (out_dir / "eval_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Append to run metrics for UI charts/compare.
    writer = JSONLMetricsWriter(run_dir / "metrics.jsonl")
    writer.write(
        {
            "event": "eval_fid_kid",
            "time_s": float(time.time()),
            "fid": float(scores["fid"]),
            "kid_mean": float(scores["kid_mean"]),
            "eval_out_dir": str(out_dir),
            "eval_checkpoint": str(ckpt_path),
            "eval_use_ema": bool(args.use_ema),
        }
    )

    print(result)


if __name__ == "__main__":
    main()
