"""
Local job runner for CLI-backed tasks.

This is designed for *local development* only:
- in-memory registry (lost on restart)
- runs whitelisted `python -m ...` commands via subprocess
- stores logs under `AI_CACHE_ROOT/jobs/<job_id>/job.log`
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class Job:
    id: str
    name: str
    cmd: List[str]
    cwd: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    return_code: Optional[int] = None
    status: str = "queued"  # queued|running|succeeded|failed|canceled
    log_path: str = ""
    job_dir: str = ""
    artifacts: List[str] = field(default_factory=list)
    pid: Optional[int] = None
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)


class JobManager:
    def __init__(self, repo_root: Path, cache_root: Optional[Path] = None):
        self.repo_root = Path(repo_root).resolve()
        self.cache_root = (
            Path(cache_root)
            if cache_root is not None
            else Path(os.getenv("AI_CACHE_ROOT", str(self.repo_root / ".ai_cache")))
        )
        self.jobs_root = (self.cache_root / "jobs").resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.max_running = int(os.getenv("MAX_JOBS", "2"))

    def list_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, name: str, cmd: Sequence[str], cwd: Optional[Path] = None) -> Job:
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == "running")
        if running >= self.max_running:
            raise RuntimeError(f"Too many running jobs ({running}); limit is {self.max_running}.")

        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "job.log"

        job = Job(
            id=job_id,
            name=name,
            cmd=list(cmd),
            cwd=str((cwd or self.repo_root).resolve()),
            created_at=time.time(),
            status="queued",
            log_path=str(log_path),
            job_dir=str(job_dir),
        )

        with self._lock:
            self._jobs[job_id] = job

        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        t.start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        proc = job._proc
        if proc is None or job.status not in {"running", "queued"}:
            return False
        try:
            if os.name == "posix" and job.pid is not None:
                try:
                    os.killpg(job.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            job.status = "canceled"
            job.finished_at = time.time()
            return True
        except Exception:
            return False

    def read_logs(self, job_id: str, tail: int = 200) -> List[str]:
        job = self.get(job_id)
        if job is None:
            return []
        p = Path(job.log_path)
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-int(tail) :]

    def _run_job(self, job: Job) -> None:
        job.started_at = time.time()
        job.status = "running"

        with open(job.log_path, "a", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(job.cmd)}\n")
            logf.flush()

            try:
                proc = subprocess.Popen(
                    job.cmd,
                    cwd=job.cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    start_new_session=(os.name == "posix"),
                )
            except Exception as e:
                job.status = "failed"
                job.finished_at = time.time()
                job.return_code = 1
                logf.write(f"[job] Failed to start: {e}\n")
                logf.flush()
                return

            job._proc = proc
            job.pid = proc.pid

            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    logf.write(line)
                    logf.flush()
            finally:
                rc = proc.wait()
                job.return_code = int(rc)
                job.finished_at = time.time()
                if job.status != "canceled":
                    job.status = "succeeded" if rc == 0 else "failed"

        # Write a minimal manifest after completion for UI consumption.
        try:
            self._write_manifest(job)
        except Exception:
            pass

    def _write_manifest(self, job: Job) -> None:
        import json

        job_dir = Path(job.job_dir)
        artifacts = []

        def _add(path: str, kind: str, display_name: str, preview: Optional[str] = None):
            artifacts.append(
                {"path": path, "type": kind, "display_name": display_name, "preview": preview}
            )

        _add(job.log_path, "text", "Job log")

        for p in list(job.artifacts or []):
            _add(p, "dir", "Output")

        # Heuristic: scan a few common output files under artifact dirs.
        wanted = [
            ("report.json", "json", "Report"),
            ("meta.json", "json", "Run metadata"),
            ("config_resolved.yaml", "yaml", "Resolved config"),
            ("metrics.jsonl", "jsonl", "Metrics"),
            ("eval_result.json", "json", "Eval result"),
            ("generated_grid.png", "image", "Generated grid"),
            ("samples_train.png", "image", "Samples (train)"),
            ("samples_val.png", "image", "Samples (val)"),
            ("samples_input.png", "image", "Samples (input)"),
            ("samples_recon.png", "image", "Samples (recon)"),
        ]

        for base in list(job.artifacts or []):
            b = Path(base)
            if not b.exists() or not b.is_dir():
                continue
            for fname, kind, disp in wanted:
                fp = b / fname
                if fp.exists() and fp.is_file():
                    _add(str(fp), kind, disp, preview=str(fp) if kind == "image" else None)

        manifest = {
            "job": {
                "id": job.id,
                "name": job.name,
                "status": job.status,
                "return_code": job.return_code,
                "cmd": job.cmd,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            },
            "artifacts": artifacts,
        }
        (job_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
