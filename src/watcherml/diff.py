"""Structured run diff: 'what changed' and 'what improved', computed deterministically.

Any AI-generated explanation is layered on top and clearly labeled — the
factual diff never depends on it.
"""
from __future__ import annotations

import json


def _load_json(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def compare_runs(storage, run_id_a: str, run_id_b: str) -> dict:
    """Compare run A (older/baseline) vs run B (newer)."""
    row_a = storage.get_run(run_id_a)
    row_b = storage.get_run(run_id_b)
    if row_a is None or row_b is None:
        raise ValueError("One or both run IDs were not found.")

    config_a, config_b = _load_json(row_a["config_json"]), _load_json(row_b["config_json"])
    config_diff = _dict_diff(config_a, config_b)

    env_a, env_b = _load_json(row_a["env_json"]), _load_json(row_b["env_json"])
    pkg_diff = _package_diff(env_a.get("packages", {}), env_b.get("packages", {}))

    git_a, git_b = _load_json(row_a["git_json"]), _load_json(row_b["git_json"])
    git_diff = {
        "commit_changed": git_a.get("commit") != git_b.get("commit"),
        "commit_a": git_a.get("commit"),
        "commit_b": git_b.get("commit"),
        "dirty_a": git_a.get("dirty"),
        "dirty_b": git_b.get("dirty"),
    }

    dataset_changed = row_a["dataset_fingerprint"] != row_b["dataset_fingerprint"]

    metrics_a = storage.final_metrics(run_id_a)
    metrics_b = storage.final_metrics(run_id_b)
    metric_diff = _metric_diff(metrics_a, metrics_b)

    res_a, res_b = _load_json(row_a["resource_json"]), _load_json(row_b["resource_json"])
    resource_diff = _resource_diff(res_a, res_b)

    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "config_diff": config_diff,
        "package_diff": pkg_diff,
        "git_diff": git_diff,
        "dataset_changed": dataset_changed,
        "metric_diff": metric_diff,
        "resource_diff": resource_diff,
        "exit_status_a": row_a["exit_status"],
        "exit_status_b": row_b["exit_status"],
    }


def _dict_diff(a: dict, b: dict) -> list:
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        if a.get(k) != b.get(k):
            out.append({"key": k, "from": a.get(k), "to": b.get(k)})
    return out


def _package_diff(a: dict, b: dict) -> list:
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        if a.get(k) != b.get(k):
            out.append({"package": k, "from": a.get(k), "to": b.get(k)})
    return out


def _metric_diff(a: dict, b: dict) -> list:
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        entry = {"metric": k, "from": va, "to": vb}
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            entry["delta"] = vb - va
        out.append(entry)
    return out


def _resource_diff(a: dict, b: dict) -> dict:
    def get(d, *path):
        cur = d
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    return {
        "vram_peak_from": get(a, "vram_used_mib_peak"),
        "vram_peak_to": get(b, "vram_used_mib_peak"),
        "gpu_util_mean_from": get(a, "gpu_util", "mean"),
        "gpu_util_mean_to": get(b, "gpu_util", "mean"),
    }


def format_diff_report(diff: dict) -> str:
    lines = [f"Comparing {diff['run_a']}  ->  {diff['run_b']}", "", "What changed?"]
    any_change = False

    for c in diff["config_diff"]:
        any_change = True
        lines.append(f"  - {c['key']}: {c['from']} -> {c['to']}")

    if diff["dataset_changed"]:
        any_change = True
        lines.append("  - dataset fingerprint changed")

    if diff["package_diff"]:
        any_change = True
        for p in diff["package_diff"][:8]:
            lines.append(f"  - {p['package']}: {p['from']} -> {p['to']}")
        if len(diff["package_diff"]) > 8:
            lines.append(f"  - ...and {len(diff['package_diff']) - 8} more package changes")

    if diff["git_diff"]["commit_changed"]:
        any_change = True
        lines.append(f"  - git commit: {diff['git_diff']['commit_a']} -> {diff['git_diff']['commit_b']}")

    if not any_change:
        lines.append("  - no tracked configuration, dataset, package, or git changes detected")

    lines.append("")
    lines.append("What changed in results?")
    for m in diff["metric_diff"]:
        if "delta" in m:
            sign = "+" if m["delta"] >= 0 else ""
            lines.append(f"  - {m['metric']}: {m['from']} -> {m['to']}  ({sign}{m['delta']:.4g})")
        else:
            lines.append(f"  - {m['metric']}: {m['from']} -> {m['to']}")

    rd = diff["resource_diff"]
    if rd.get("vram_peak_from") is not None or rd.get("vram_peak_to") is not None:
        lines.append(f"  - peak VRAM: {rd.get('vram_peak_from')} -> {rd.get('vram_peak_to')} MiB")
    if rd.get("gpu_util_mean_from") is not None or rd.get("gpu_util_mean_to") is not None:
        lines.append(f"  - mean GPU utilization: {rd.get('gpu_util_mean_from')} -> {rd.get('gpu_util_mean_to')}%")

    lines.append("")
    lines.append(f"Exit status: {diff['exit_status_a']} -> {diff['exit_status_b']}")
    return "\n".join(lines)
