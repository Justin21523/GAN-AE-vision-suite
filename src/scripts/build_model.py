"""
Model factory for AE/VAE-style models.

Training scripts in `src/scripts/` commonly rely on a config-driven model builder
to keep experimentation simple. This file maps `cfg["model"]["type"]` to a
concrete PyTorch module.

Supported types (case-insensitive):
- "ae"
- "conv-ae" / "convae"
- "vae" (configurable ConvVAE)
- "vae-fixed" (simple fixed-arch VAE)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from src.models.ae.ae import AutoEncoder
from src.models.ae import ConvAE, ConvVAE, VariationalAutoEncoder


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Small helper for dict/AddictDict-like configs."""
    try:
        return cfg.get(key, default)  # type: ignore[attr-defined]
    except Exception:
        return default


def _infer_input_size(cfg: Mapping[str, Any], mcfg: Mapping[str, Any]) -> Optional[int]:
    """Infer model input size from model/data sections if not explicitly provided."""
    if _get(mcfg, "input_size") is not None:
        return int(_get(mcfg, "input_size"))
    data = _get(cfg, "data", {}) or {}
    if _get(data, "image_size") is not None:
        return int(_get(data, "image_size"))
    if _get(data, "img_size") is not None:
        return int(_get(data, "img_size"))
    return None


def build_model(cfg: Mapping[str, Any]):
    """
    Build model from configuration.
    Expected cfg["model"] keys:
      - type: one of {"ae", "conv-ae", "convae", "vae"}
      - img_channels: int (default 3)
      - latent_dim: int (default 128)
      - hidden_dims: List[int] (default [32, 64, 128])
      - activation: str (default "relu")
      - input_size: Optional[int or tuple] (needed by some VAE impls)
    """
    mcfg = _get(cfg, "model", None) or {}

    # Backward-compat: some configs use top-level `model_type` + model fields.
    if not mcfg and _get(cfg, "model_type", None) is not None:
        mcfg = {
            "type": _get(cfg, "model_type"),
            "img_channels": _get(cfg, "img_channels", _get(cfg, "channels", 3)),
            "latent_dim": _get(cfg, "latent_dim", 128),
            "input_size": _get(cfg, "input_size", None),
        }

    mtype = str(_get(mcfg, "type", _get(cfg, "model_type", "ae"))).lower()

    img_ch = int(_get(mcfg, "img_channels", _get(mcfg, "channels", 3)))
    latent = int(_get(mcfg, "latent_dim", _get(mcfg, "z_dim", 128)))
    hidden = list(_get(mcfg, "hidden_dims", [32, 64, 128]))
    act = str(_get(mcfg, "activation", "relu"))
    input_size: Optional[int] = _infer_input_size(cfg, mcfg)

    if mtype in {"ae"}:
        model = AutoEncoder(
            img_channels=img_ch,
            latent_dim=latent,
            hidden_dims=hidden,
            activation=act,
            input_size=(int(input_size), int(input_size)) if input_size else (28, 28),
        )

    elif mtype in {"conv-ae", "conv_ae", "convae"}:
        model = ConvAE(
            input_channels=img_ch,
            latent_dim=latent,
            hidden_dims=hidden,
            activation=act,
            input_size=int(input_size) if input_size else 64,
        )

    elif mtype in {"vae", "convvae"}:
        model = ConvVAE(
            input_channels=img_ch,
            latent_dim=latent,
            hidden_dims=hidden,
            activation=act,
            input_size=int(input_size) if input_size else 64,
        )

    elif mtype in {"vae-fixed", "vae_fixed", "simple-vae", "simple_vae"}:
        model = VariationalAutoEncoder(
            img_channels=img_ch,
            latent_dim=latent,
            input_size=int(input_size) if input_size else 64,
        )

    else:
        raise ValueError(f"Unknown model type: {mtype}")

    return model
