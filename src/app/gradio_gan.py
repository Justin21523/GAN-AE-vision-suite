"""
Gradio UI for sampling a GAN checkpoint (optional).

This UI depends on the local Python environment having a compatible `gradio` +
`gradio_client` installation. If those packages are missing or mismatched, this
module will exit with a clear message (instead of crashing on import).

Usage:
  python -m src.app.gradio_gan

Alternative:
  If Gradio isn't available, you can still use the FastAPI + React UI:
  - Start API: `scripts/serve_api.sh`
  - Start UI:  `cd gan-ui && npm ci && npm run dev`
"""

from __future__ import annotations

import os

from src.service.gan_infer import GANService, GenerateParams


def main() -> None:
    """Launch the Gradio UI if dependencies are available."""
    try:
        import gradio as gr  # type: ignore
    except Exception as e:
        raise SystemExit(
            "❌ 無法匯入 gradio（可能是 gradio / gradio_client 版本不相容）。\n"
            "建議做法：\n"
            "  1) 重新安裝依賴：python -m pip install -r requirements.txt\n"
            "  2) 或改用 FastAPI + React UI：scripts/serve_api.sh + (cd gan-ui && npm run dev)\n\n"
            f"Original import error: {e}"
        ) from e

    service = GANService()

    def load_checkpoint(ckpt_path: str, device: str) -> str:
        if not ckpt_path:
            raise gr.Error("Please provide a checkpoint path.")

        # Recreate service if device changes.
        nonlocal service
        dev = None if device == "auto" else device
        service = GANService(device=dev)

        try:
            service.load_checkpoint(ckpt_path)
        except Exception as err:
            raise gr.Error(f"Failed to load checkpoint: {err}")

        cfg = service.cfg or {}
        return (
            f"Loaded: device={service.device}, img_size={cfg.get('img_size')}, "
            f"img_channels={cfg.get('img_channels')}, latent_dim={cfg.get('latent_dim')}, "
            f"has_ema_shadow={service.has_ema_shadow}"
        )

    def generate_grid(n: int, nrow: int, seed: int, use_ema: bool):
        if service.G is None:
            raise gr.Error("No checkpoint loaded yet. Click 'Load checkpoint' first.")

        return service.generate_grid(
            GenerateParams(
                n=int(n),
                nrow=int(nrow),
                seed=int(seed),
                use_ema_shadow=bool(use_ema),
            )
        )

    with gr.Blocks(title="GAN Sampler") as demo:
        gr.Markdown("# GAN Sampler")
        gr.Markdown("Load a GAN checkpoint (`ckpt_epoch*.pt`) and generate a sample grid.")

        with gr.Row():
            ckpt = gr.Textbox(
                label="Checkpoint path",
                value="logs/stage3_wgangp/ckpt_epoch1.pt",
                placeholder="e.g., logs/stage3_wgangp/ckpt_epoch10.pt",
            )
            device = gr.Dropdown(
                label="Device",
                choices=["auto", "cuda", "cpu", "cuda:0", "cuda:1"],
                value="auto",
            )

        load_btn = gr.Button("Load checkpoint", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("## Sampling")
        with gr.Row():
            n = gr.Slider(1, 256, value=64, step=1, label="n (num samples)")
            nrow = gr.Slider(1, 32, value=8, step=1, label="nrow (grid columns)")
            seed = gr.Number(value=42, precision=0, label="seed")
            use_ema = gr.Checkbox(value=False, label="Use EMA (if available)")

        gen_btn = gr.Button("Generate", variant="primary")
        out = gr.Image(label="Output grid", type="pil")

        load_btn.click(load_checkpoint, inputs=[ckpt, device], outputs=[status])
        gen_btn.click(generate_grid, inputs=[n, nrow, seed, use_ema], outputs=[out])

    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name=host, server_port=port, share=False, show_error=True)


if __name__ == "__main__":
    main()

