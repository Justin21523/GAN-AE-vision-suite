"""
Job manager tests.

These tests run a tiny subprocess and ensure status/log capture works.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.service.jobs import JobManager


def test_job_manager_runs_and_logs(tmp_path: Path) -> None:
    jm = JobManager(repo_root=tmp_path, cache_root=tmp_path / ".cache")
    job = jm.start(name="hello", cmd=[sys.executable, "-c", "print('hello')"], cwd=tmp_path)

    # wait up to 2s
    t0 = time.time()
    while True:
        j = jm.get(job.id)
        assert j is not None
        if j.status in {"succeeded", "failed"}:
            break
        if time.time() - t0 > 2.0:
            raise AssertionError("job did not finish in time")
        time.sleep(0.05)

    lines = jm.read_logs(job.id, tail=50)
    joined = "\n".join(lines)
    assert "hello" in joined

