# src/scripts/build_model.py  (或你現有的 build_model 定義處)
from typing import Optional, List
from src.models.ae.ae import AutoEncoder
from src.models.ae import ConvAE
from src.models.ae import VariationalAutoEncoder


def build_model(cfg):
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
    mcfg = cfg["model"]
    mtype = str(mcfg.get("type", "ae")).lower()

    img_ch = int(mcfg.get("img_channels", 3))
    latent = int(mcfg.get("latent_dim", 128))
    hidden = mcfg.get("hidden_dims", [32, 64, 128])
    act = mcfg.get("activation", "relu")
    input_size: Optional[int] = mcfg.get("input_size", None)

    if mtype in {"ae"}:
        model = AutoEncoder(
            img_channels=img_ch,
            latent_dim=latent,
            hidden_dims=hidden,
            activation=act,
        )

    elif mtype in {"conv-ae", "convae"}:
        model = ConvAE(
            input_channels=img_ch,
            latent_dim=latent,
            hidden_dims=hidden,
            activation=act,
        )

    elif mtype == "vae":
        vae_kwargs = dict(
            img_channels=img_ch,
            latent_dim=latent,
            hidden_dims=hidden,
            activation=act,
        )
        if input_size is not None:
            vae_kwargs["input_size"] = input_size
        model = VariationalAutoEncoder(**vae_kwargs)

    else:
        raise ValueError(f"Unknown model type: {mtype}")

    return model
