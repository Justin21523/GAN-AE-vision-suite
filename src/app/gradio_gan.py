# src/app/gradio_gan.py
import gradio as gr
from src.service.gan_infer import GANService, GenerateParams
import torch


def build_ui():
    svc = GANService()

    def on_load(ckpt: str, device: str):
        svc.__init__(device=device or None)  # re-init with device choice
        svc.load_checkpoint(ckpt)
        return f"Loaded: {ckpt} on {svc.device.type}. model={svc.cfg['type']} size={svc.cfg['img_size']}"  # type: ignore

    def on_generate(n, seed, nrow, use_ema):
        img = svc.generate_grid(
            GenerateParams(n=n, seed=seed, nrow=nrow, use_ema_shadow=use_ema)
        )
        return img

    with gr.Blocks(title="GAN Sampler") as demo:
        gr.Markdown(
            "## GAN Sampler (DCGAN/WGAN-GP)\nUse any Stage-3 checkpoint to generate sample grids."
        )
        with gr.Row():
            ckpt = gr.Textbox(
                label="Checkpoint path", placeholder="logs/stage3_wgangp/ckpt_epoch1.pt"
            )
            device = gr.Dropdown(
                ["cuda", "cpu"],
                value="cuda" if torch.cuda.is_available() else "cpu",
                label="Device",
            )
            load_btn = gr.Button("Load")
        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            n = gr.Slider(4, 128, value=64, step=4, label="Samples (n)")
            nrow = gr.Slider(1, 16, value=8, step=1, label="Grid nrow")
        with gr.Row():
            seed = gr.Number(value=42, precision=0, label="Seed")
            use_ema = gr.Checkbox(value=False, label="Use EMA shadow (if available)")
            gen_btn = gr.Button("Generate")

        out = gr.Image(type="pil", label="Output grid", show_download_button=True)

        load_btn.click(on_load, inputs=[ckpt, device], outputs=status)
        gen_btn.click(on_generate, inputs=[n, seed, nrow, use_ema], outputs=out)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
