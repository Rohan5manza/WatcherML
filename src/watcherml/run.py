from __future__ import annotations

import sys
import time
import uuid
from typing import Optional

from . import collectors
from .capsule import build_failure_capsule, format_capsule_report
from .storage import Storage


def _short_id(project: str) -> str:
    return f"{project}-{uuid.uuid4().hex[:6]}"


class Run:
    """A single WatcherML experiment run.

    Usage:
        with watcherml.init(project="tomato-disease", config={...}) as run:
            run.log_metric("val_accuracy", 0.91)
            run.set_dataset("./data/tomato")
    """

    def __init__(self, project: str, config: Optional[dict] = None,
                 run_id: Optional[str] = None, storage: Optional[Storage] = None,
                 sample_interval: float = 2.0):
        self.project = project
        self.config = config or {}
        self.run_id = run_id or _short_id(project)
        self.storage = storage or Storage()
        self.sample_interval = sample_interval

        self.started_at: Optional[float] = None
        self.git_info: dict = {}
        self.env_info: dict = {}
        self.gpu_info: dict = {}
        self.dataset_fingerprint: Optional[str] = None
        self.sampler: Optional[collectors.SystemSampler] = None
        self.warnings: list = []
        self._notebook_cells: list = []  # populated by watcherml.notebook, if loaded
        self._finished = False

    # -- lifecycle -----------------------------------------------------
    def start(self) -> "Run":
        self.started_at = time.time()
        self.git_info = collectors.collect_git_info(".")
        self.env_info = collectors.collect_env_info()
        self.gpu_info = collectors.collect_gpu_info()

        if self.git_info.get("dirty"):
            self.warnings.append("Notebook/repo contains uncommitted changes.")
        if not self.gpu_info.get("available"):
            self.warnings.append("No GPU detected; running in CPU-only / no-GPU mode.")

        self.sampler = collectors.SystemSampler(
            interval_seconds=self.sample_interval, on_sample=self._flush_sample)
        self.sampler.start()

        self.storage.upsert_run(
            self.run_id,
            project=self.project,
            config_json=self.config,
            started_at=self.started_at,
            exit_status="running",
            git_json=self.git_info,
            env_json=self.env_info,
            gpu_json=self.gpu_info,
        )
        return self

    def _flush_sample(self, sample: dict):
        """Called from the sampler's background thread after every sample --
        this is what makes system telemetry visible in the UI *during*
        training, not only after the run finishes. Best-effort: a storage
        hiccup here must never interrupt the actual training loop."""
        try:
            self.storage.save_resource_samples(self.run_id, [sample])
        except Exception:
            pass

    def set_dataset(self, path: str):
        self.dataset_fingerprint = collectors.dataset_fingerprint(path)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        self.storage.log_metric(self.run_id, name, float(value), step, time.time())

    def log(self, metrics: dict, step: Optional[int] = None):
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, path: str):
        import hashlib
        import os
        checksum = ""
        size = 0
        try:
            with open(path, "rb") as f:
                data = f.read()
                checksum = hashlib.sha256(data).hexdigest()[:16]
                size = len(data)
        except OSError:
            pass
        self.storage.log_artifact(self.run_id, path, checksum, size)

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "Run":
        if self.started_at is None:
            self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        try:
            if exc_type is not None:
                self._finish_failure(exc_type, exc_value, exc_tb)
                return False  # do not suppress — the user should still see the error
            self._finish_success()
            return False
        finally:
            self._clear_notebook_active_state()

    def finish(self):
        """Explicitly mark a notebook-style run (no 'with' block) as successful.

        Use this when you call watcher.init() directly in a cell rather than
        wrapping your work in `with watcher.init(...) as run:`. If a later
        cell raises instead, WatcherML notices automatically — you don't
        need to call anything in that case.
        """
        if self._finished:
            return
        self._finish_success()
        self._clear_notebook_active_state()

    def _notebook_auto_fail(self, exc_type, exc_value, exc_tb):
        """Called by watcherml.notebook when a cell raises while this run is active
        and the user never wrapped it in a 'with' block."""
        if self._finished:
            return
        self._finish_failure(exc_type, exc_value, exc_tb)

    def _clear_notebook_active_state(self):
        try:
            from . import notebook
            notebook.clear_active_run(self)
        except ImportError:
            pass

    # -- finishing -----------------------------------------------------
    def _finish_failure(self, exc_type, exc_value, exc_tb):
        if self._finished:
            return
        self._finished = True
        ended_at = time.time()
        resource_summary = self.sampler.stop() if self.sampler else {}
        capsule = build_failure_capsule(self, exc_type, exc_value, exc_tb)

        self.storage.upsert_run(
            self.run_id,
            ended_at=ended_at,
            duration_seconds=ended_at - self.started_at,
            exit_status="failed",
            resource_json=resource_summary,
            dataset_fingerprint=self.dataset_fingerprint,
            warnings_json=self.warnings,
        )
        # Samples are no longer batch-inserted here -- they were already
        # persisted live, one at a time, via _flush_sample() as they were taken.
        self.storage.save_failure(
            self.run_id, capsule["exception_type"], capsule["message"],
            capsule["traceback"], capsule["diagnosis"], capsule["evidence"],
        )
        print("\n" + format_capsule_report(capsule) + "\n", file=sys.stderr)

    def _finish_success(self):
        if self._finished:
            return
        self._finished = True
        ended_at = time.time()
        duration = ended_at - self.started_at
        resource_summary = self.sampler.stop() if self.sampler else {}
        score = self._reproduction_score()

        self.storage.upsert_run(
            self.run_id,
            ended_at=ended_at,
            duration_seconds=duration,
            exit_status="success",
            resource_json=resource_summary,
            dataset_fingerprint=self.dataset_fingerprint,
            reproduction_score=score,
            warnings_json=self.warnings,
        )
        # Samples are no longer batch-inserted here -- they were already
        # persisted live, one at a time, via _flush_sample() as they were taken.
        self._print_receipt(duration, resource_summary, score)

    def _reproduction_score(self) -> float:
        score = 0
        if self.git_info.get("available") and not self.git_info.get("dirty"):
            score += 3
        elif self.git_info.get("available"):
            score += 1
        if self.config:
            score += 2
        if self.dataset_fingerprint:
            score += 2
        if self.env_info.get("package_count"):
            score += 2
        if self.gpu_info.get("available"):
            score += 1
        return min(score, 10)

    def _print_receipt(self, duration: float, resource_summary: dict, score: float):
        mins, secs = divmod(int(duration), 60)
        final_metrics = self.storage.final_metrics(self.run_id)

        lines = [f"WatcherML run completed: {self.run_id}", ""]
        for name, value in final_metrics.items():
            lines.append(f"{name:<18} {value}")
        lines.append(f"{'duration':<18} {mins}m {secs}s")
        if resource_summary.get("vram_used_mib_peak") is not None:
            lines.append(f"{'peak_vram':<18} {resource_summary['vram_used_mib_peak']:.1f} MiB")
        if resource_summary.get("gpu_util", {}):
            lines.append(f"{'mean_gpu_util':<18} {resource_summary['gpu_util']['mean']:.0f}%")
        if self.git_info.get("available"):
            git_state = "dirty" if self.git_info.get("dirty") else "clean"
        else:
            git_state = "no_git_repo"
        lines.append(f"{'git_state':<18} {git_state}")
        if self.dataset_fingerprint:
            lines.append(f"{'dataset_fingerprint':<18} {self.dataset_fingerprint}")
        lines.append(f"{'reproduction_score':<18} {int(score)}/10")

        if resource_summary.get("gpu_low_utilization_fraction") is not None and \
                resource_summary["gpu_low_utilization_fraction"] > 0.2:
            self.warnings.append(
                f"GPU utilization fell below 50% for "
                f"{resource_summary['gpu_low_utilization_fraction']*100:.0f}% of the run"
            )

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"- {w}")

        lines.append("")
        lines.append(f"Inspect: watcher inspect {self.run_id}")
        print("\n" + "\n".join(lines) + "\n")


def init(project: Optional[str] = None, config: Optional[dict] = None,
         storage: Optional[Storage] = None) -> Run:
    """Start recording a new WatcherML run.

    Recommended usage as a context manager so failures are captured automatically:
        with watcherml.init(project="tomato-disease", config={...}) as run:
            ...

    In a notebook, `project` can be omitted after `%watcher project <name>` —
    see `watcherml.notebook`.
    """
    if project is None:
        try:
            from . import notebook
            project = notebook.get_active_project_default()
        except ImportError:
            project = None
        if project is None:
            raise ValueError(
                "project is required. Pass project=..., or in a notebook run "
                "`%watcher project <name>` first."
            )

    run = Run(project=project, config=config, storage=storage)
    run.start()
    try:
        from . import notebook
        notebook.set_active_run(run)
    except ImportError:
        pass
    return run