"""Versioned, deterministic failure capsules."""
from __future__ import annotations

import os
import sys
import time
import traceback as tb_module

from . import collectors, failures, similarity
from .capsule_schema import (
    CAPSULE_SCHEMA_NAME,
    CAPSULE_SCHEMA_VERSION,
    calculate_capture_completeness,
    validate_capsule,
)


_EVIDENCE_CATEGORIES = (
    ("EV-1", "config", "Run configuration"),
    ("EV-2", "training_state", "Last recorded training state"),
    ("EV-3", "runtime", "Process/runtime context"),
    ("EV-4", "resource_state_at_failure", "CPU/RAM/GPU sampler summary"),
    ("EV-5", "gpu", "GPU hardware and driver information"),
    ("EV-6", "framework", "Framework and CUDA allocator state"),
    ("EV-7", "git", "Git state"),
    ("EV-8", "environment", "Python/package environment fingerprint"),
    ("EV-9", "dataset", "Dataset fingerprint"),
    ("EV-10", "recent_metrics", "Recent metric history"),
    ("EV-11", "notebook_cells_executed", "Notebook execution history"),
)


def build_failure_capsule(run, exc_type, exc_value, exc_tb) -> dict:
    """Build a v1 capsule without network calls, models, or generated claims."""
    tb_str = "".join(tb_module.format_exception(exc_type, exc_value, exc_tb))
    message = str(exc_value)
    type_name = exc_type.__name__
    classification = failures.diagnose(type_name, message, tb_str)

    recent_rows = run.storage.get_metrics(run.run_id)[-10:]
    recent_metrics = [
        {"name": row["name"], "value": row["value"], "step": row["step"],
         "timestamp": row["timestamp"]}
        for row in recent_rows
    ]
    resource_summary = run.sampler.stats.summary() if run.sampler else {}
    torch_state = collectors.collect_torch_cuda_state()
    framework = {
        "python_version": run.env_info.get("python_version"),
        "platform": run.env_info.get("platform"),
        **torch_state,
    }

    environment_fingerprint = run.env_info.get("fingerprint")
    if not environment_fingerprint:
        environment_fingerprint = collectors.environment_fingerprint(run.env_info)

    notebook_cells = getattr(run, "_notebook_cells", None) or []
    evidence = {
        "config": run.config,
        "training_state": _training_state(run.config, recent_metrics),
        "runtime": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "working_directory": os.getcwd(),
        },
        "resource_state_at_failure": resource_summary,
        "gpu": run.gpu_info or {"available": False},
        "framework": framework,
        "git": {
            key: value for key, value in (run.git_info or {}).items()
            if key != "diff_patch"
        },
        "environment": {
            "python_version": run.env_info.get("python_version"),
            "platform": run.env_info.get("platform"),
            "package_count": run.env_info.get("package_count"),
            "fingerprint": environment_fingerprint,
        },
        "dataset": {"fingerprint": run.dataset_fingerprint},
        "recent_metrics": recent_metrics,
        # Source code previews are intentionally excluded from v1 to reduce
        # accidental secret capture. Only execution order and status remain.
        "notebook_cells_executed": [
            {"execution_count": cell.get("execution_count"),
             "success": cell.get("success")}
            for cell in notebook_cells
        ] or None,
    }

    evidence_index = build_evidence_index(evidence)
    classification["evidence_ids"] = _evidence_ids_for_categories(
        evidence_index, classification.get("evidence_categories", []))
    failure = {
        "class": classification["rule"],
        "exception_type": type_name,
        "message": message,
        "traceback": tb_str,
        "classification": classification,
    }
    capture = calculate_capture_completeness(failure, evidence)

    target_snapshot = {
        "config": run.config,
        "dataset_fingerprint": run.dataset_fingerprint,
        "git": run.git_info or {},
        "gpu": run.gpu_info or {},
        "env": run.env_info or {},
        "started_at": run.started_at,
    }
    nearest_success = similarity.find_nearest_successful_run(
        run.storage, run.project, run.run_id, target_snapshot)
    similar = find_similar_failures(
        run.storage, run.project, run.run_id, classification["rule"])

    capsule = {
        "schema": {"name": CAPSULE_SCHEMA_NAME, "version": CAPSULE_SCHEMA_VERSION},
        "capsule_schema_version": CAPSULE_SCHEMA_VERSION,
        "run_id": run.run_id,
        "project": run.project,
        "captured_at": time.time(),
        "failure": failure,
        "failure_class": classification["rule"],
        "evidence": evidence,
        "evidence_index": evidence_index,
        "capture": capture,
        "capture_completeness": capture["score"],
        "similar_previous_failures": similar,
        "nearest_successful_run": nearest_success,
        # Compatibility aliases for the v0.1 CLI/UI. Remove only in a future
        # major schema migration after all consumers use capsule["failure"].
        "exception_type": type_name,
        "message": message,
        "traceback": tb_str,
        "classification": classification,
        "diagnosis": classification,
        "comparison_to_last_success": nearest_success,
    }
    errors = validate_capsule(capsule)
    if errors:  # This is a WatcherML bug, not a user-training failure.
        raise ValueError("Invalid WatcherML failure capsule: " + "; ".join(errors))
    return capsule


def _training_state(config: dict, recent_metrics: list[dict]) -> dict:
    def first(*keys):
        for key in keys:
            if key in config:
                return config[key]
        return None

    logged_steps = [m["step"] for m in recent_metrics if m.get("step") is not None]
    batch_size = first("batch_size", "per_device_train_batch_size", "train_batch_size")
    grad_accum = first("gradient_accumulation_steps", "grad_accum_steps")
    effective_batch = None
    if isinstance(batch_size, (int, float)) and isinstance(grad_accum, (int, float)):
        effective_batch = batch_size * grad_accum
    return {
        "last_logged_step": max(logged_steps) if logged_steps else None,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size_per_process": effective_batch,
        "precision": first("precision", "mixed_precision", "dtype"),
        "gradient_checkpointing": first("gradient_checkpointing"),
        "sequence_length": first("sequence_length", "max_seq_length", "max_length"),
        "image_resolution": first("image_resolution", "resolution", "image_size"),
    }


def build_evidence_index(evidence: dict) -> list[dict]:
    """Return present evidence categories with permanently assigned IDs."""
    index = []
    for evidence_id, category, label in _EVIDENCE_CATEGORIES:
        if evidence.get(category) in (None, {}, [], ""):
            continue
        index.append({"id": evidence_id, "category": category, "label": label})
    return index


def _evidence_ids_for_categories(evidence_index: list, categories: list) -> list:
    return [item["id"] for item in evidence_index if item["category"] in categories]


def find_similar_failures(storage, project: str, exclude_run_id: str,
                          rule_name: str, limit: int = 3):
    out = []
    for row in storage.list_failures(project=project):
        if row["run_id"] == exclude_run_id:
            continue
        diagnosis = _safe_json(row["diagnosis_json"])
        if diagnosis and diagnosis.get("rule") == rule_name:
            out.append({"run_id": row["run_id"], "message": row["message"]})
        if len(out) >= limit:
            break
    return out


def compare_to_last_success(storage, project: str, exclude_run_id: str):
    """Backward-compatible name for nearest-success comparison."""
    row = storage.get_run(exclude_run_id)
    if row is None:
        return None
    target_snapshot = similarity.run_snapshot_from_row(row)
    return similarity.find_nearest_successful_run(
        storage, project, exclude_run_id, target_snapshot)


def _safe_json(value):
    if not value:
        return None
    import json
    try:
        return json.loads(value)
    except Exception:
        return None


def format_capsule_report(capsule: dict) -> str:
    classification = capsule.get("classification") or capsule["failure"]["classification"]
    failure = capsule.get("failure") or capsule
    lines = [
        f"WatcherML failure capsule v{capsule.get('capsule_schema_version', 'legacy')}: "
        f"{capsule['run_id']}",
        "",
        f"Exception:   {failure['exception_type']}: {failure['message']}",
        f"Diagnosis:   {classification['rule']} (deterministic rule)",
        f"  {classification['summary']}",
    ]
    if classification.get("likely_cause"):
        lines.append(f"  Likely cause: {classification['likely_cause']}")
    if classification.get("evidence_ids"):
        lines.append(f"  Based on: {', '.join(classification['evidence_ids'])}")

    capture = capsule.get("capture") or {}
    if capture:
        lines.extend([
            "",
            f"Capture completeness: {capture['score']}/{capture['maximum']}",
        ])
        if capture.get("missing"):
            lines.append(f"  Missing: {', '.join(capture['missing'])}")

    state = (capsule.get("evidence") or {}).get("training_state") or {}
    if any(value is not None for value in state.values()):
        lines.extend(["", "Last recorded training state:"])
        for key, value in state.items():
            if value is not None:
                lines.append(f"   {key}: {value}")

    allocator = (capsule.get("evidence") or {}).get("framework") or {}
    if allocator.get("cuda_available"):
        lines.extend(["", "CUDA allocator state at failure:"])
        for key in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes",
                    "max_reserved_bytes", "free_bytes", "total_bytes", "oom_count"):
            if allocator.get(key) is not None:
                lines.append(f"   {key}: {allocator[key]}")

    nearest = capsule.get("nearest_successful_run") or capsule.get("comparison_to_last_success")
    if nearest:
        score = nearest.get("similarity_score")
        suffix = f" (similarity {score * 100:.0f}%)" if score is not None else ""
        lines.extend(["", f"Nearest successful run: {nearest['run_id']}{suffix}"])

    lines.extend(["", f"Inspect: watcher inspect {capsule['run_id']}"])
    return "\n".join(lines)
