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
    """Samples CPU/RAM/GPU on a background thread at a fixed interval.

    on_sample, if given, is called with each raw sample dict as it's taken --
    this is what lets a run's telemetry be persisted (and therefore visible
    in the UI) *while training is still in progress*, not only after the run
    ends. A callback failure never kills the sampling thread or the run.
    """

    def __init__(self, interval_seconds: float = 2.0, on_sample=None):
        self.interval = interval_seconds
        self.stats = SamplerStats()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_sample = on_sample
        self._prev_disk_io = None  # psutil counters are cumulative -- we report rates
        self._prev_net_io = None   # (MB/s), computed as the delta between consecutive samples
        self._prev_t = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            sample = {"t": now}
            if psutil is not None:
                sample["cpu_pct"] = psutil.cpu_percent(interval=None)
                sample["ram_pct"] = psutil.virtual_memory().percent
                self._sample_io_rates(sample, now)
            gpu = _sample_gpu_once()
            if gpu:
                sample["gpu_util_pct"] = gpu["util_pct"]
                sample["gpu_mem_used_mib"] = gpu["mem_used_mib"]
                sample["gpu_temp_c"] = gpu["temp_c"]
            self.stats.samples.append(sample)
            if self._on_sample is not None:
                try:
                    self._on_sample(sample)
                except Exception:
                    pass  # live telemetry is best-effort -- never let it disrupt training
            self._stop_event.wait(self.interval)

    def _sample_io_rates(self, sample: dict, now: float):
        """Disk/network throughput, in MB/s, computed as the delta between
        consecutive cumulative-counter reads. First sample in a run has
        nothing to diff against, so it's skipped (no rate yet, not zero)."""
        try:
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
        except Exception:
            return  # not available on this platform/permissions -- degrade silently
        if disk_io is None or net_io is None:
            return
        if self._prev_t is not None and now > self._prev_t:
            dt = now - self._prev_t
            sample["disk_read_mbps"] = (disk_io.read_bytes - self._prev_disk_io.read_bytes) / dt / 1e6
            sample["disk_write_mbps"] = (disk_io.write_bytes - self._prev_disk_io.write_bytes) / dt / 1e6
            sample["net_sent_mbps"] = (net_io.bytes_sent - self._prev_net_io.bytes_sent) / dt / 1e6
            sample["net_recv_mbps"] = (net_io.bytes_recv - self._prev_net_io.bytes_recv) / dt / 1e6
        self._prev_disk_io = disk_io
        self._prev_net_io = net_io
        self._prev_t = now

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