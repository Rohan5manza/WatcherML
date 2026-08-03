"""Collectors: everything WatcherML gathers automatically about a run's context."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------

def _run(cmd, cwd=None):
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def collect_git_info(path: str = ".") -> dict:
    """Capture commit, branch, and dirty-diff state. Degrades gracefully if not a git repo."""
    if _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path) != "true":
        return {"available": False}

    commit = _run(["git", "rev-parse", "HEAD"], cwd=path)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
    status = _run(["git", "status", "--porcelain"], cwd=path) or ""
    dirty = bool(status.strip())
    diff_stat = _run(["git", "diff", "--stat"], cwd=path) or ""
    diff_patch = _run(["git", "diff"], cwd=path) or ""

    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "changed_files": [line[3:] for line in status.splitlines() if line.strip()],
        "diff_stat": diff_stat,
        "diff_patch": diff_patch,  # used for reproduction capsules; not shown in receipts
    }


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def collect_env_info() -> dict:
    """Python version + installed package snapshot."""
    packages = {}
    try:
        from importlib import metadata as importlib_metadata
        for dist in importlib_metadata.distributions():
            try:
                name = dist.metadata["Name"]
                if name:
                    packages[name] = dist.version
            except Exception:
                continue
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "packages": packages,
        "package_count": len(packages),
    }


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

def collect_gpu_info() -> dict:
    """Static GPU/driver/CUDA info via nvidia-smi. Fully optional — absence is not an error."""
    query = "name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu"
    out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not out:
        return {"available": False}

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append({
            "name": parts[0],
            "driver_version": parts[1],
            "memory_total_mib": _to_num(parts[2]),
            "memory_used_mib": _to_num(parts[3]),
            "utilization_pct": _to_num(parts[4]),
            "temperature_c": _to_num(parts[5]),
        })
    cuda_version = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {"available": True, "gpus": gpus, "driver_version": cuda_version}


def _to_num(s):
    try:
        return float(s)
    except Exception:
        return None


def _sample_gpu_once() -> Optional[dict]:
    out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    return {
        "util_pct": _to_num(parts[0]),
        "mem_used_mib": _to_num(parts[1]),
        "temp_c": _to_num(parts[2]),
    }


# --------------------------------------------------------------------------
# System sampler (background thread: CPU / RAM / GPU over time)
# --------------------------------------------------------------------------

@dataclass
class SamplerStats:
    samples: list = field(default_factory=list)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        cpu_vals = [s["cpu_pct"] for s in self.samples if s.get("cpu_pct") is not None]
        ram_vals = [s["ram_pct"] for s in self.samples if s.get("ram_pct") is not None]
        gpu_vals = [s["gpu_util_pct"] for s in self.samples if s.get("gpu_util_pct") is not None]
        vram_vals = [s["gpu_mem_used_mib"] for s in self.samples if s.get("gpu_mem_used_mib") is not None]

        def agg(vals):
            return {"mean": sum(vals) / len(vals), "peak": max(vals)} if vals else None

        gpu_agg = agg(gpu_vals)
        low_util_samples = [v for v in gpu_vals if v < 50]
        return {
            "cpu": agg(cpu_vals),
            "ram": agg(ram_vals),
            "gpu_util": gpu_agg,
            "vram_used_mib_peak": max(vram_vals) if vram_vals else None,
            "gpu_low_utilization_fraction": (
                len(low_util_samples) / len(gpu_vals) if gpu_vals else None
            ),
            "sample_count": len(self.samples),
        }


class SystemSampler:
    """Samples CPU/RAM/GPU on a background thread at a fixed interval."""

    def __init__(self, interval_seconds: float = 2.0):
        self.interval = interval_seconds
        self.stats = SamplerStats()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop_event.is_set():
            sample = {"t": time.time()}
            if psutil is not None:
                sample["cpu_pct"] = psutil.cpu_percent(interval=None)
                sample["ram_pct"] = psutil.virtual_memory().percent
            gpu = _sample_gpu_once()
            if gpu:
                sample["gpu_util_pct"] = gpu["util_pct"]
                sample["gpu_mem_used_mib"] = gpu["mem_used_mib"]
                sample["gpu_temp_c"] = gpu["temp_c"]
            self.stats.samples.append(sample)
            self._stop_event.wait(self.interval)

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
        return self.stats.summary()


# --------------------------------------------------------------------------
# Dataset fingerprint (lightweight — listing + sizes + mtimes, not full hashing)
# --------------------------------------------------------------------------

def dataset_fingerprint(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    if os.path.isfile(path):
        stat = os.stat(path)
        hasher.update(f"{path}:{stat.st_size}:{stat.st_mtime}".encode())
    else:
        for root, _dirs, files in os.walk(path):
            for f in sorted(files):
                fp = os.path.join(root, f)
                try:
                    stat = os.stat(fp)
                    hasher.update(f"{fp}:{stat.st_size}:{stat.st_mtime}".encode())
                except OSError:
                    continue
    return hasher.hexdigest()[:12]
