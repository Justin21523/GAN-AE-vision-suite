"""
Run/artifact utility tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.run import JSONLMetricsWriter, data_fingerprint, prepare_run_dir, write_config_yaml


def test_prepare_run_dir_creates(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, run_name="abc", prefix="x")
    assert run_dir.exists()
    assert run_dir.name == "abc"


def test_write_config_yaml(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    write_config_yaml({"a": 1, "b": {"c": 2}}, p)
    assert p.exists()
    assert "a" in p.read_text(encoding="utf-8")


def test_jsonl_metrics_writer(tmp_path: Path) -> None:
    p = tmp_path / "metrics.jsonl"
    w = JSONLMetricsWriter(p)
    w.write({"step": 1, "loss": 0.5})
    w.write({"step": 2, "loss": 0.4})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1


def test_data_fingerprint_includes_split_hash(tmp_path: Path) -> None:
    train_list = tmp_path / "train.txt"
    train_list.write_text("a.png\nb.png\n", encoding="utf-8")
    cfg = {"data": {"dataset": "imagefolder", "root": "/data", "train_list": str(train_list)}}
    fp = data_fingerprint(cfg)
    assert fp["dataset"] == "imagefolder"
    assert "train_list" in fp
    assert "sha256" in fp["train_list"]

