"""
API capabilities and job specs.

This centralizes the UI<->backend contract so the React UI can render dynamic
forms and stay in sync with supported CLI commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    type: str  # string|number|boolean|path
    required: bool = False
    default: Any = None
    placeholder: Optional[str] = None
    help: Optional[str] = None
    choices: Optional[List[Any]] = None


@dataclass(frozen=True)
class JobSpec:
    type: str
    label: str
    description: str
    args: Sequence[FieldSpec] = field(default_factory=list)


def job_specs() -> List[JobSpec]:
    return [
        JobSpec(
            type="train_gan",
            label="Train GAN",
            description="Runs `python -m src.scripts.train_gan` as a background job.",
            args=[
                FieldSpec(
                    key="config",
                    label="Config",
                    type="path",
                    required=False,
                    default="configs/gan/wgangp_celeba128.yaml",
                    help="Path to a GAN training config YAML. Required unless resuming.",
                ),
                FieldSpec(
                    key="device",
                    label="Device",
                    type="string",
                    required=False,
                    default="cuda",
                    choices=["cuda", "cuda:0", "cpu"],
                ),
                FieldSpec(
                    key="run_name",
                    label="Run name",
                    type="string",
                    required=False,
                    default="exp01",
                    help="Creates a subdir under training.logdir for this run.",
                ),
                FieldSpec(
                    key="resume",
                    label="Resume checkpoint",
                    type="path",
                    required=False,
                    default="",
                    placeholder="logs/.../ckpt_epochX.pt",
                ),
                FieldSpec(key="finetune", label="Finetune", type="boolean", required=False, default=False),
            ],
        ),
        JobSpec(
            type="train_ae",
            label="Train AE/VAE",
            description="Runs `python -m src.scripts.train_ae` as a background job.",
            args=[
                FieldSpec(
                    key="config",
                    label="Config",
                    type="path",
                    required=False,
                    default="configs/dataset_mnist.yaml",
                    help="Path to a dataset/model config YAML. Required unless resuming.",
                ),
                FieldSpec(
                    key="device",
                    label="Device",
                    type="string",
                    required=False,
                    default="cuda",
                    choices=["cuda", "cuda:0", "cpu"],
                ),
                FieldSpec(
                    key="epochs",
                    label="Epochs override",
                    type="number",
                    required=False,
                    default=None,
                    help="Optional override of training.epochs",
                ),
                FieldSpec(
                    key="run_name",
                    label="Run name",
                    type="string",
                    required=False,
                    default="exp01",
                    help="Creates a subdir under logging.log_dir for this run.",
                ),
                FieldSpec(
                    key="resume",
                    label="Resume checkpoint",
                    type="path",
                    required=False,
                    default="",
                    placeholder="logs/checkpoints/ckpt_best.pt",
                ),
                FieldSpec(key="finetune", label="Finetune", type="boolean", required=False, default=False),
            ],
        ),
        JobSpec(
            type="data_report",
            label="Data Report",
            description="Runs `python -m src.scripts.data_report` and writes report.json + sample grids.",
            args=[
                FieldSpec(
                    key="config",
                    label="Config",
                    type="path",
                    required=True,
                    default="configs/dataset_celeba.yaml",
                ),
                FieldSpec(
                    key="out",
                    label="Out dir",
                    type="path",
                    required=False,
                    default="./outputs/data_report",
                ),
                FieldSpec(
                    key="hash_duplicates",
                    label="Hash duplicates (slow)",
                    type="boolean",
                    required=False,
                    default=False,
                ),
                FieldSpec(
                    key="hash_max_files",
                    label="Hash max files",
                    type="number",
                    required=False,
                    default=5000,
                ),
            ],
        ),
        JobSpec(
            type="validate_data",
            label="Validate Data",
            description="Runs `python -m src.validate_data` to write sample grids + quick metric check.",
            args=[
                FieldSpec(
                    key="config",
                    label="Config",
                    type="path",
                    required=True,
                    default="configs/dataset_celeba.yaml",
                ),
                FieldSpec(key="use_ae", label="Use AE forward", type="boolean", required=False, default=False),
            ],
        ),
        JobSpec(
            type="prepare_data",
            label="Prepare Data (Check)",
            description="Runs `python -m src.scripts.prepare_data` to validate local dataset layouts.",
            args=[
                FieldSpec(
                    key="dataset",
                    label="Dataset",
                    type="string",
                    required=True,
                    default="celeba",
                    choices=["celeba", "mnist", "cifar10", "imagefolder"],
                ),
                FieldSpec(key="root", label="Root", type="path", required=False, default="./data"),
            ],
        ),
        JobSpec(
            type="prepare_demo",
            label="Prepare Data (Create Demo ImageFolder)",
            description="Runs `python -m src.scripts.prepare_data --create-demo-imagefolder ...`.",
            args=[
                FieldSpec(
                    key="create_demo_imagefolder",
                    label="Out path",
                    type="path",
                    required=True,
                    default="./data/demo_images",
                ),
                FieldSpec(key="num_images", label="# images", type="number", required=False, default=32),
                FieldSpec(key="img_size", label="Image size", type="number", required=False, default=64),
            ],
        ),
        JobSpec(
            type="sample_gan",
            label="Sample GAN (to file)",
            description="Runs `python -m src.scripts.sample_gan` to write a grid image to disk.",
            args=[
                FieldSpec(key="checkpoint", label="Checkpoint", type="path", required=True, default=""),
                FieldSpec(key="out", label="Out path", type="path", required=False, default="./outputs/sample_gan.png"),
                FieldSpec(key="n", label="n", type="number", required=False, default=64),
                FieldSpec(key="seed", label="seed", type="number", required=False, default=42),
                FieldSpec(key="ema", label="Use EMA", type="boolean", required=False, default=True),
            ],
        ),
        JobSpec(
            type="eval_fid",
            label="Eval FID/KID",
            description="Runs `python -m src.scripts.eval_fid` against a directory of generated images.",
            args=[
                FieldSpec(key="config", label="Data config", type="path", required=True, default="configs/dataset_celeba.yaml"),
                FieldSpec(key="gen_dir", label="Generated images dir", type="path", required=True, default="./outputs/gen_dir"),
                FieldSpec(key="max_samples", label="Max samples", type="number", required=False, default=10000),
            ],
        ),
        JobSpec(
            type="eval_gan_pipeline",
            label="Eval GAN (Generate → FID/KID → Append Metrics)",
            description="Runs `python -m src.scripts.eval_gan_pipeline` to evaluate an existing run directory.",
            args=[
                FieldSpec(
                    key="run_dir",
                    label="Run dir",
                    type="path",
                    required=True,
                    default="logs/gan_...",
                    help="A run directory containing config_resolved.yaml and ckpt_epoch*.pt.",
                ),
                FieldSpec(
                    key="checkpoint",
                    label="Checkpoint (optional)",
                    type="path",
                    required=False,
                    default="",
                    help="If empty, uses the latest ckpt_epoch*.pt in the run dir.",
                ),
                FieldSpec(key="device", label="Device", type="string", required=False, default="cuda", choices=["cuda", "cuda:0", "cpu"]),
                FieldSpec(key="n_images", label="# images", type="number", required=False, default=2000),
                FieldSpec(key="batch_size", label="Batch size", type="number", required=False, default=64),
                FieldSpec(key="seed", label="Seed", type="number", required=False, default=123),
                FieldSpec(key="use_ema", label="Use EMA", type="boolean", required=False, default=True),
                FieldSpec(key="max_samples", label="Max samples", type="number", required=False, default=10000),
                FieldSpec(
                    key="out_dir",
                    label="Out dir (optional)",
                    type="path",
                    required=False,
                    default="",
                    help="If empty, writes under run_dir/eval/gen_<timestamp>.",
                ),
            ],
        ),
    ]


def as_dict() -> Dict[str, Any]:
    specs = job_specs()
    return {
        "jobs": [
            {
                "type": j.type,
                "label": j.label,
                "description": j.description,
                "args": [
                    {
                        "key": f.key,
                        "label": f.label,
                        "type": f.type,
                        "required": bool(f.required),
                        "default": f.default,
                        "placeholder": f.placeholder,
                        "help": f.help,
                        "choices": f.choices,
                    }
                    for f in j.args
                ],
            }
            for j in specs
        ]
    }
