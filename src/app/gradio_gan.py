# src/app/gradio_gan.py
"""
Gradio UI for Stage-3 GAN sampling.
- Manual checkpoint path (editable)
- Optional: upload a .pt checkpoint instead
- Robust error messages (no bare "Error")
- Status panel always updates
"""
import torch
import traceback
from typing import Optional
import os, gradio as gr
from src.service.gan_infer import GANService, GenerateParams


def _pick_ckpt_path(text_path: str, file_obj) -> Optional[str]:
    """
    Choose which checkpoint to use:
    1) If text_path exists on server, use it.
    2) Else if file_obj is uploaded, use the uploaded temp path.
    3) Otherwise return None.
    """
    text_path = (text_path or "").strip()
    if text_path and os.path.exists(text_path):
        return text_path
    if file_obj is not None:
        # gr.File returns a tempfile-like object; .name is the local path
        up = getattr(file_obj, "name", None)
        if up and os.path.exists(up):
            return up
    return None


def build_ui():
    # Keep a single service instance for the whole app
    svc = GANService()

    def on_load(ckpt_text: str, ckpt_file, device: str):
        try:
            ckpt = _pick_ckpt_path(ckpt_text, ckpt_file)
            if not ckpt:
                return f"Error: checkpoint not found. Please type a valid path or upload a .pt file."

            # Re-init device and load weights
            svc.__init__(device=device or None)
            svc.load_checkpoint(ckpt)
            msg = (
                f"✅ Loaded on {svc.device}, img_size={svc.cfg['img_size']} "  # type: ignore
                f"({os.path.basename(ckpt)})"
            )
            return msg
        except Exception as e:
            tb = traceback.format_exc(limit=1)
            return f"❌ Load failed: {e.__class__.__name__}: {e}\n{tb}"

    def on_generate(n: int, seed: int, nrow: int, use_ema: bool):
        try:
            if svc.G is None:
                return None, "Error: no model loaded yet. Click 'Load' first."
            params = GenerateParams(n=n, seed=seed, nrow=nrow, use_ema_shadow=use_ema)
            img = svc.generate_grid(params)
            return img, "Done."
        except Exception as e:
            tb = traceback.format_exc(limit=1)
            return None, f"❌ Generate failed: {e.__class__.__name__}: {e}\n{tb}"

    with gr.Blocks(title="GAN Sampler (Stage-3)") as demo:
        gr.Markdown("### GAN Sampler (DCGAN/WGAN-GP)")
        gr.Markdown(
            "Use a trained Stage-3 checkpoint. You can type a server path or upload a `.pt` file."
        )

        with gr.Row():
            ckpt_text = gr.Textbox(
                label="Checkpoint path (server)",
                value="logs/stage3_wgangp/ckpt_epoch1.pt",  # you can change it freely
                lines=1,
                interactive=True,
                placeholder="e.g., logs/stage3_wgangp/ckpt_epoch10.pt",
            )
            ckpt_file = gr.File(
                label="or Upload checkpoint (.pt)",
                file_count="single",
                file_types=[".pt"],
                interactive=True,
            )
            device = gr.Dropdown(
                ["cuda", "cpu"],
                value="cuda" if torch.cuda.is_available() else "cpu",
                label="Device",
            )
            load_btn = gr.Button("Load", variant="primary")

        status = gr.Textbox(
            label="Status",
            value="Idle",
            interactive=False,
            lines=2,
        )

        with gr.Row():
            n = gr.Slider(4, 128, value=64, step=4, label="Samples (n)")
            nrow = gr.Slider(1, 16, value=8, step=1, label="Grid nrow")

        with gr.Row():
            seed = gr.Number(value=42, precision=0, label="Seed")
            use_ema = gr.Checkbox(value=False, label="Use EMA shadow (if available)")
            gen_btn = gr.Button("Generate", variant="primary")

        out = gr.Image(type="pil", label="Output grid", show_download_button=True)

        # Wiring
        load_btn.click(on_load, inputs=[ckpt_text, ckpt_file, device], outputs=status)
        gen_btn.click(
            on_generate, inputs=[n, seed, nrow, use_ema], outputs=[out, status]
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
