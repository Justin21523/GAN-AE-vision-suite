import gradio as gr
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Bootstrap runtime
from src.utils.runtime import bootstrap_runtime
from src.service.gan_infer import GANInferenceService
from src.registry.index import get_run_registry, get_model_index
from src.jobs.engine import get_job_engine

# Initialize runtime and services
cfg, info = bootstrap_runtime()
gan_service = GANInferenceService(cfg)
registry = get_run_registry()
model_index = get_model_index()
job_engine = get_job_engine()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGradioGANUI:
    """Enhanced Gradio UI with run search and evaluation features."""

    def __init__(self):
        self.cfg = cfg
        self.gan_service = gan_service
        self.registry = registry
        self.model_index = model_index
        self.job_engine = job_engine

    def get_available_runs(self) -> List[str]:
        """Get list of available run IDs."""
        runs = self.registry.get_all_runs()
        return [run["id"] for run in runs]

    def search_runs(
        self, query: str, status_filter: str, task_type_filter: str
    ) -> List[List]:
        """Search runs with filters."""
        runs = self.registry.search_runs(query)

        # Apply additional filters
        if status_filter != "all":
            runs = [run for run in runs if run.get("status") == status_filter]

        if task_type_filter != "all":
            runs = [
                run
                for run in runs
                if run.get("config", {}).get("task_type") == task_type_filter
            ]

        # Format results for dataframe
        results = []
        for run in runs[:20]:  # Limit to 20 results
            config = run.get("config", {})
            data_config = config.get("data", {})
            metrics = run.get("metrics", {})

            # Get best FID if available
            best_fid = "N/A"
            if "fid" in metrics:
                best_fid = f"{metrics['fid']:.2f}"
            elif "best_metrics" in run and "fid" in run["best_metrics"]:
                best_fid = f"{run['best_metrics']['fid']:.2f}"

            results.append(
                [
                    run.get("id", "N/A"),
                    run.get("status", "N/A"),
                    config.get("task_type", "N/A"),
                    data_config.get("name", "N/A"),
                    best_fid,
                    run.get("created_at", "N/A")[:10],  # Just date
                ]
            )

        return results

    def get_run_details(self, run_id: str) -> str:
        """Get detailed information about a run."""
        if not run_id:
            return "Select a run to see details"

        run = self.registry.get_run(run_id)
        if not run:
            return f"Run '{run_id}' not found"

        details = f"# Run: {run_id}\n\n"

        # Basic info
        details += f"**Status**: {run.get('status', 'N/A')}\n"
        details += f"**Created**: {run.get('created_at', 'N/A')}\n"
        details += f"**Updated**: {run.get('updated_at', 'N/A')}\n\n"

        # Configuration
        config = run.get("config", {})
        details += "## Configuration\n"
        details += f"- **Task Type**: {config.get('task_type', 'N/A')}\n"
        details += f"- **Dataset**: {config.get('data', {}).get('name', 'N/A')}\n"
        details += (
            f"- **Image Size**: {config.get('data', {}).get('image_size', 'N/A')}\n\n"
        )

        # Metrics
        metrics = run.get("metrics", {})
        best_metrics = run.get("best_metrics", {})

        if metrics or best_metrics:
            details += "## Metrics\n"

            if metrics:
                details += "**Latest**:\n"
                for metric, value in list(metrics.items())[:5]:  # Show top 5
                    details += f"- {metric}: {value:.4f}\n"

            if best_metrics:
                details += "\n**Best**:\n"
                for metric, value in list(best_metrics.items())[:5]:
                    details += f"- {metric}: {value:.4f}\n"

        return details

    def schedule_evaluation(
        self, run_id: str, checkpoint_type: str, force_recompute: bool
    ) -> str:
        """Schedule FID/KID evaluation for a run."""
        if not run_id:
            return "❌ Please select a run first"

        try:
            job_id = self.job_engine.submit(
                eval_fidkid_task,
                run_id=run_id,
                checkpoint_type=checkpoint_type,
                force_recompute=force_recompute,
            )

            return f"✅ Evaluation job submitted: `{job_id}`\n\nYou can check the status in the Jobs tab."

        except Exception as e:
            return f"❌ Evaluation scheduling failed: {str(e)}"

    def get_job_status(self) -> List[List]:
        """Get current job status."""
        jobs = self.job_engine.list_jobs(limit=20)

        results = []
        for job in jobs:
            results.append(
                [
                    job["id"][:8] + "...",  # Shorten ID
                    job["status"],
                    (
                        job.get("started_at", "N/A")[:19]
                        if job.get("started_at")
                        else "Pending"
                    ),
                    f"{job.get('progress', 0) * 100:.1f}%",
                    job.get("error", "")[:50] + "..." if job.get("error") else "",
                ]
            )

        return results

    def create_enhanced_interface(self):
        """Create enhanced Gradio interface with tabs."""
        with gr.Blocks(
            title="GAN-AE-VISION-SUITE",
            theme=gr.themes.Soft(),
            css="""
            .message-box {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
            }
            .success { border-color: #4caf50; background: #f1f8e9; }
            .error { border-color: #f44336; background: #ffebee; }
            .info { border-color: #2196f3; background: #e3f2fd; }
            .run-table { font-size: 0.9em; }
            """,
        ) as interface:
            gr.Markdown("# 🎨 GAN-AE-VISION-SUITE")
            gr.Markdown(f"**AI Warehouse**: `{self.cfg.cache_root}`")

            with gr.Tabs():
                # Tab 1: Generation
                with gr.TabItem("🎲 Generation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Run selection
                            run_dropdown = gr.Dropdown(
                                choices=self.get_available_runs(),
                                label="Select Run",
                                info="Choose a trained model run",
                            )

                            # Checkpoint type
                            checkpoint_radio = gr.Radio(
                                choices=["latest", "best"],
                                value="latest",
                                label="Checkpoint Type",
                                info="Use latest or best checkpoint",
                            )

                            # Generation parameters
                            num_samples_slider = gr.Slider(
                                minimum=1,
                                maximum=64,
                                value=16,
                                step=1,
                                label="Number of Samples",
                                info="How many samples to generate",
                            )

                            grid_nrow_slider = gr.Slider(
                                minimum=1,
                                maximum=8,
                                value=4,
                                step=1,
                                label="Grid Columns",
                                info="Number of images per row",
                            )

                            seed_number = gr.Number(
                                label="Random Seed",
                                info="Leave empty for random",
                                precision=0,
                            )

                            generate_btn = gr.Button(
                                "🎲 Generate Samples", variant="primary"
                            )

                        with gr.Column(scale=2):
                            # Run information
                            run_info = gr.Markdown("Select a run to see details")

                            # Output image
                            output_image = gr.Image(
                                label="Generated Samples", type="filepath", height=400
                            )

                            # Output message
                            output_message = gr.HTML(
                                value="<div class='message-box info'>Ready to generate samples</div>"
                            )

                # Tab 2: Run Management
                with gr.TabItem("📊 Run Management"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_query = gr.Textbox(
                                label="Search Runs",
                                placeholder="Enter search query...",
                                info="Search by run ID, status, or dataset",
                            )

                            with gr.Row():
                                status_filter = gr.Dropdown(
                                    choices=[
                                        "all",
                                        "queued",
                                        "running",
                                        "completed",
                                        "failed",
                                    ],
                                    value="all",
                                    label="Status Filter",
                                )

                                task_type_filter = gr.Dropdown(
                                    choices=[
                                        "all",
                                        "gan",
                                        "ae",
                                        "vae",
                                        "pix2pix",
                                        "cyclegan",
                                    ],
                                    value="all",
                                    label="Task Type Filter",
                                )

                            search_btn = gr.Button("🔍 Search", variant="secondary")

                            # Selected run for operations
                            selected_run = gr.Textbox(
                                label="Selected Run ID", interactive=False
                            )

                            with gr.Row():
                                eval_checkpoint = gr.Radio(
                                    choices=["latest", "best"],
                                    value="best",
                                    label="Evaluate Checkpoint",
                                )

                                force_recompute = gr.Checkbox(
                                    label="Force Recompute",
                                    value=False,
                                    info="Ignore cached results",
                                )

                            eval_btn = gr.Button(
                                "📈 Evaluate FID/KID", variant="primary"
                            )

                            eval_status = gr.HTML(
                                value="<div class='message-box info'>Select a run to evaluate</div>"
                            )

                        with gr.Column(scale=2):
                            search_results = gr.Dataframe(
                                headers=[
                                    "ID",
                                    "Status",
                                    "Task",
                                    "Dataset",
                                    "Best FID",
                                    "Created",
                                ],
                                label="Search Results",
                                interactive=True,
                                wrap=True,
                                elem_classes="run-table",
                            )

                            run_details = gr.Markdown(
                                value="Select a run from the table to see details"
                            )

                # Tab 3: Job Management
                with gr.TabItem("⚙️ Job Management"):
                    with gr.Row():
                        with gr.Column():
                            refresh_btn = gr.Button(
                                "🔄 Refresh Job Status", variant="secondary"
                            )

                            job_status_table = gr.Dataframe(
                                headers=[
                                    "Job ID",
                                    "Status",
                                    "Started",
                                    "Progress",
                                    "Error",
                                ],
                                label="Current Jobs",
                                interactive=False,
                                wrap=True,
                            )

                        with gr.Column():
                            gr.Markdown("### Scheduled Tasks")
                            scheduled_tasks = gr.JSON(
                                label="Scheduled Tasks Status", value={}
                            )

            # Event handlers for Generation tab
            run_dropdown.change(
                fn=self.get_run_details, inputs=[run_dropdown], outputs=[run_info]
            )

            generate_btn.click(
                fn=self._generate_samples,
                inputs=[
                    run_dropdown,
                    checkpoint_radio,
                    num_samples_slider,
                    grid_nrow_slider,
                    seed_number,
                ],
                outputs=[output_image, output_message],
            )

            # Event handlers for Run Management tab
            search_btn.click(
                fn=self.search_runs,
                inputs=[search_query, status_filter, task_type_filter],
                outputs=[search_results],
            )

            search_results.select(
                fn=self._select_run_from_table,
                inputs=[search_results],
                outputs=[selected_run, run_details],
            )

            eval_btn.click(
                fn=self.schedule_evaluation,
                inputs=[selected_run, eval_checkpoint, force_recompute],
                outputs=[eval_status],
            )

            # Event handlers for Job Management tab
            refresh_btn.click(fn=self.get_job_status, outputs=[job_status_table])

            refresh_btn.click(fn=self._get_scheduled_tasks, outputs=[scheduled_tasks])

        return interface

    def _generate_samples(self, run_id, checkpoint_type, num_samples, grid_nrow, seed):
        """Generate samples wrapper for Gradio."""
        result = self.gan_service.generate_samples(
            run_id=run_id,
            checkpoint_type=checkpoint_type,
            num_samples=num_samples,
            grid_nrow=grid_nrow,
            seed=seed,
        )

        if result["success"]:
            css_class = "success"
            message = f"✅ Generated {num_samples} samples\nRun: {run_id}\nCheckpoint: {checkpoint_type}\nSaved to: {result['output_path']}"
        else:
            css_class = "error"
            message = f"❌ Generation failed: {result['error']}"

        return (
            result.get("output_path"),
            f'<div class="message-box {css_class}">{message}</div>',
        )

    def _select_run_from_table(self, evt: gr.SelectData, dataframe):
        """Handle run selection from search results table."""
        if evt.index[0] < len(dataframe):
            run_id = dataframe[evt.index[0]][0]
            details = self.get_run_details(run_id)
            return run_id, details
        return "", "Select a run from the table"

    def _get_scheduled_tasks(self):
        """Get scheduled tasks status."""
        from src.jobs.scheduler import get_scheduler

        scheduler = get_scheduler()
        return scheduler.get_scheduled_tasks_status()

    def get_checkpoint_types(self) -> List[str]:
        """Get available checkpoint types."""
        return ["latest", "best"]

    def generate_samples(
        self,
        run_id: str,
        checkpoint_type: str,
        num_samples: int,
        grid_nrow: int,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Generate samples and return results for UI."""
        try:
            result = self.gan_service.generate_samples(
                run_id=run_id,
                checkpoint_type=checkpoint_type,
                num_samples=num_samples,
                grid_nrow=grid_nrow,
                seed=seed,
            )

            if result["success"]:
                return {
                    "image": result["output_path"],
                    "message": f"✅ Generated {num_samples} samples\n"
                    f"Run: {run_id}\n"
                    f"Checkpoint: {checkpoint_type}\n"
                    f"Seed: {result['seed_used'] or 'random'}\n"
                    f"Saved to: {result['output_path']}",
                    "success": True,
                }
            else:
                return {
                    "image": None,
                    "message": f"❌ Generation failed: {result['error']}",
                    "success": False,
                }

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "image": None,
                "message": f"❌ Generation failed: {str(e)}",
                "success": False,
            }

    def get_run_info(self, run_id: str) -> str:
        """Get information about a run."""
        if not run_id:
            return "Select a run to see details"

        runs = self.gan_service.get_available_runs()
        for run in runs:
            if run["id"] == run_id:
                info_text = f"# Run: {run_id}\n\n"

                if run.get("manifest"):
                    manifest = run["manifest"]
                    info_text += f"**Task Type**: {manifest.get('task_type', 'N/A')}\n"
                    info_text += f"**Dataset**: {manifest.get('dataset', 'N/A')}\n"
                    info_text += f"**Status**: {manifest.get('status', 'N/A')}\n"
                    if "metrics" in manifest:
                        info_text += "\n**Best Metrics**:\n"
                        for metric, value in manifest["metrics"].items():
                            info_text += f"- {metric}: {value:.4f}\n"

                info_text += f"\n**Checkpoints**:\n"
                if run["checkpoints"]["latest"]:
                    info_text += f"- Latest: ✓\n"
                if run["checkpoints"]["best"]:
                    info_text += f"- Best: ✓\n"

                return info_text

        return f"Run '{run_id}' not found or has no manifest"

    def get_recent_samples(self, run_id: str) -> List[str]:
        """Get recent sample images for a run."""
        if not run_id:
            return []

        samples_dir = os.path.join(self.cfg.output_dir, "samples", run_id)
        api_samples_dir = os.path.join(self.cfg.output_dir, "api_samples", run_id)

        sample_paths = []

        # Check training samples
        if os.path.exists(samples_dir):
            for ext in ["png", "jpg", "jpeg"]:
                sample_paths.extend(Path(samples_dir).glob(f"*.{ext}"))

        # Check API-generated samples
        if os.path.exists(api_samples_dir):
            for ext in ["png", "jpg", "jpeg"]:
                sample_paths.extend(Path(api_samples_dir).glob(f"*.{ext}"))

        # Sort by modification time (newest first) and get latest 10
        sample_paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(path) for path in sample_paths[:10]]

    def create_interface(self):
        with gr.Blocks(
            title="GAN-AE-VISION-SUITE",
            theme=gr.themes.Soft(),
            css="""
            .message-box {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
            }
            .success { border-color: #4caf50; background: #f1f8e9; }
            .error { border-color: #f44336; background: #ffebee; }
            """,
        ) as interface:
            gr.Markdown("# 🎨 GAN-AE-VISION-SUITE")
            gr.Markdown(f"**AI Warehouse**: `{self.cfg.cache_root}`")

            with gr.Row():
                with gr.Column(scale=1):
                    # Run selection
                    run_dropdown = gr.Dropdown(
                        choices=self.get_available_runs(),
                        label="Select Run",
                        info="Choose a trained model run",
                    )

                    # Checkpoint type
                    checkpoint_radio = gr.Radio(
                        choices=self.get_checkpoint_types(),
                        value="latest",
                        label="Checkpoint Type",
                        info="Use latest or best checkpoint",
                    )

                    # Generation parameters
                    num_samples_slider = gr.Slider(
                        minimum=1,
                        maximum=64,
                        value=16,
                        step=1,
                        label="Number of Samples",
                        info="How many samples to generate",
                    )

                    grid_nrow_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Grid Columns",
                        info="Number of images per row",
                    )

                    seed_number = gr.Number(
                        label="Random Seed", info="Leave empty for random", precision=0
                    )

                    generate_btn = gr.Button("🎲 Generate Samples", variant="primary")

                    gr.Markdown("## 🔍 Run Search & Evaluation")

                    search_query = gr.Textbox(
                        label="Search Runs",
                        placeholder="Enter search query...",
                        info="Search by run ID, status, or dataset",
                    )

                    search_results = gr.Dataframe(
                        headers=["ID", "Status", "Task Type", "Dataset"],
                        label="Search Results",
                        interactive=False,
                    )

                    selected_run_id = gr.Textbox(
                        label="Selected Run ID", interactive=False
                    )

                    with gr.Row():
                        search_btn = gr.Button("Search", variant="secondary")
                        eval_btn = gr.Button("Evaluate FID/KID", variant="primary")

                    eval_status = gr.HTML(
                        value="<div class='message-box'>Select a run to evaluate</div>"
                    )
                with gr.Column(scale=2):
                    # Run information
                    run_info = gr.Markdown("Select a run to see details")

                    # Output image
                    output_image = gr.Image(
                        label="Generated Samples", type="filepath", height=400
                    )

                    # Output message
                    output_message = gr.HTML(
                        value="<div class='message-box'>Ready to generate samples</div>"
                    )

            # Recent samples gallery
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Recent Samples")
                    sample_gallery = gr.Gallery(
                        label="Recent Samples from Selected Run",
                        show_label=True,
                        elem_id="gallery",
                    ).style(columns=4, rows=2, height="auto")

            # Event handlers
            run_dropdown.change(
                fn=self.get_run_info, inputs=[run_dropdown], outputs=[run_info]
            )

            run_dropdown.change(
                fn=self.get_recent_samples,
                inputs=[run_dropdown],
                outputs=[sample_gallery],
            )

            generate_btn.click(
                fn=self.generate_samples,
                inputs=[
                    run_dropdown,
                    checkpoint_radio,
                    num_samples_slider,
                    grid_nrow_slider,
                    seed_number,
                ],
                outputs=[output_image, output_message],
            )

            # 事件處理
            search_btn.click(
                fn=self.search_runs, inputs=[search_query], outputs=[search_results]
            )

            search_results.select(
                fn=self.select_run, inputs=[search_results], outputs=[selected_run_id]
            )

            eval_btn.click(
                fn=self.evaluate_run, inputs=[selected_run_id], outputs=[eval_status]
            )

            # Update message display based on success
            def update_message_display(result):
                css_class = "success" if result["success"] else "error"
                return f'<div class="message-box {css_class}">{result["message"]}</div>'

            # Additional handler for message formatting
            generate_btn.click(
                fn=update_message_display,
                inputs=[output_message],
                outputs=[output_message],
            )

        return interface

    def select_run(self, evt: gr.SelectData, dataframe):
        """Handle run selection from search results."""
        if evt.index[0] < len(dataframe):
            return dataframe[evt.index[0]][0]
        return ""

    def evaluate_run(self, run_id: str):
        """Schedule FID/KID evaluation for a run."""
        if not run_id:
            return "<div class='message-box error'>Please select a run first</div>"

        try:
            job_engine = get_job_engine()
            job_id = job_engine.submit(
                eval_fidkid_task, run_id=run_id, checkpoint_type="best"
            )

            return f"<div class='message-box success'>Evaluation job submitted: {job_id}</div>"

        except Exception as e:
            return f"<div class='message-box error'>Evaluation failed: {str(e)}</div>"


def main():
    """Launch enhanced Gradio UI."""
    ui = EnhancedGradioGANUI()
    interface = ui.create_enhanced_interface()

    # Get host and port from environment or use defaults
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    port = int(os.getenv("GRADIO_PORT", "7860"))

    logger.info(f"Starting enhanced Gradio UI on {host}:{port}")
    logger.info(f"AI Warehouse: {cfg.cache_root}")

    interface.launch(server_name=host, server_port=port, share=False, show_error=True)


if __name__ == "__main__":
    main()
