"""Failure capsules: everything needed to understand *why* a run crashed."""
from __future__ import annotations

import time
import traceback as tb_module
from typing import Optional

from . import failures, similarity


def build_failure_capsule(run, exc_type, exc_value, exc_tb) -> dict:
    tb_str = "".join(tb_module.format_exception(exc_type, exc_value, exc_tb))
    message = str(exc_value)
    type_name = exc_type.__name__

    diagnosis = failures.diagnose(type_name, message, tb_str)

    recent_metrics = run.storage.get_metrics(run.run_id)[-10:]
    resource_summary = run.sampler.stats.summary() if run.sampler else {}

    notebook_cells = getattr(run, "_notebook_cells", None)

    evidence = {
        "config": run.config,
        "recent_metrics": [
            {"name": r["name"], "value": r["value"], "step": r["step"]} for r in recent_metrics
        ],
        "resource_state_at_failure": resource_summary,
        "gpu": run.gpu_info,
        "git": {k: v for k, v in (run.git_info or {}).items() if k != "diff_patch"},
        "env": {"python_version": run.env_info.get("python_version"),
                "package_count": run.env_info.get("package_count")},
        "notebook_cells_executed": [
            {"execution_count": c["execution_count"], "success": c["success"],
             "source_preview": (c["source"] or "")[:200]}
            for c in (notebook_cells or [])
        ] or None,
        "timestamp": time.time(),
    }
    evidence_index = build_evidence_index(evidence)
    diagnosis["evidence_ids"] = _evidence_ids_for_categories(
        evidence_index, diagnosis.get("evidence_categories", []))

    similar = find_similar_failures(run.storage, run.project, run.run_id, diagnosis["rule"])
    # NOTE: this runs before the failing run's own row is fully persisted
    # (dataset_fingerprint in particular is written later, in Run._finish_failure),
    # so we build the comparison target from the live Run object's attributes
    # rather than reading it back from storage -- reading storage here would
    # silently compare against a stale/incomplete snapshot of THIS run.
    target_snapshot = {
        "config": run.config,
        "dataset_fingerprint": run.dataset_fingerprint,
        "git": run.git_info or {},
        "gpu": run.gpu_info or {},
        "env": run.env_info or {},
        "started_at": run.started_at,
    }
    comparison = similarity.find_nearest_successful_run(
        run.storage, run.project, run.run_id, target_snapshot)

    return {
        "run_id": run.run_id,
        "exception_type": type_name,
        "message": message,
        "traceback": tb_str,
        "diagnosis": diagnosis,
        "evidence": evidence,
        "evidence_index": evidence_index,
        "similar_previous_failures": similar,
        "comparison_to_last_success": comparison,
    }


_EVIDENCE_CATEGORY_LABELS = {
    "config": "Run configuration",
    "resource_state_at_failure": "GPU/CPU resource state at failure",
    "git": "Git state",
    "env": "Environment / package snapshot",
    "recent_metrics": "Recent metric history",
    "gpu": "GPU hardware info",
    "notebook_cells_executed": "Notebook cell execution history",
}

# Order determines EV-N numbering -- stable across capsules so "EV-2" always
# means the same category, not whatever happened to be present that run.
_EVIDENCE_CATEGORY_ORDER = [
    "config", "resource_state_at_failure", "git", "env",
    "recent_metrics", "gpu", "notebook_cells_executed",
]


def build_evidence_index(evidence: dict) -> list:
    """Assigns a stable EV-N id to each present, non-empty evidence category.
    This is category-level granularity (e.g. EV-2 = 'the whole resource
    snapshot'), not per-fact -- a deliberate simplification over per-value
    evidence IDs, documented here rather than silently implied to be finer
    grained than it is."""
    index = []
    n = 1
    for category in _EVIDENCE_CATEGORY_ORDER:
        value = evidence.get(category)
        if value in (None, {}, [], ""):
            continue
        index.append({
            "id": f"EV-{n}",
            "category": category,
            "label": _EVIDENCE_CATEGORY_LABELS.get(category, category),
        })
        n += 1
    return index


def _evidence_ids_for_categories(evidence_index: list, categories: list) -> list:
    return [e["id"] for e in evidence_index if e["category"] in categories]


def find_similar_failures(storage, project: str, exclude_run_id: str, rule_name: str, limit: int = 3):
    rows = storage.list_failures(project=project)
    out = []
    for row in rows:
        if row["run_id"] == exclude_run_id:
            continue
        diag = _safe_json(row["diagnosis_json"])
        if diag and diag.get("rule") == rule_name:
            out.append({"run_id": row["run_id"], "message": row["message"]})
        if len(out) >= limit:
            break
    return out


def compare_to_last_success(storage, project: str, exclude_run_id: str):
    """Finds the best baseline to compare a failure against.

    Despite the name (kept for backward compatibility with existing callers
    in recovery.py and webapp.py), this no longer picks 'the last successful
    run' -- it picks the MOST SIMILAR one, via similarity.py's documented
    weighting (dataset fingerprint, model, GPU, git ancestry, config
    distance, framework versions, temporal proximity). The old behavior was
    silently misleading: recency is not relevance.

    Returns the same shape as before ({"run_id", "config"}) plus extra keys
    (similarity_score, checklist, failure_relevant_differences) -- callers
    that only look at run_id/config keep working unchanged.
    """
    row = storage.get_run(exclude_run_id)
    if row is None:
        return None
    target_snapshot = similarity.run_snapshot_from_row(row)
    return similarity.find_nearest_successful_run(storage, project, exclude_run_id, target_snapshot)


def _safe_json(s):
    if not s:
        return None
    import json
    try:
        return json.loads(s)
    except Exception:
        return None


def format_capsule_report(capsule: dict) -> str:
    d = capsule["diagnosis"]
    lines = []
    lines.append(f"WatcherML failure capsule: {capsule['run_id']}")
    lines.append("")
    lines.append(f"Exception:   {capsule['exception_type']}: {capsule['message']}")
    lines.append(f"Diagnosis:   {d['rule']}")
    lines.append(f"  {d['summary']}")
    if d.get("likely_cause"):
        lines.append(f"  Likely cause: {d['likely_cause']}")
    if d.get("evidence_ids"):
        lines.append(f"  Based on: {', '.join(d['evidence_ids'])}")
    if d.get("suggested_actions"):
        lines.append("  Suggested actions:")
        for a in d["suggested_actions"]:
            lines.append(f"   - {a}")

    ev = capsule["evidence"]
    if ev.get("notebook_cells_executed"):
        cells = ev["notebook_cells_executed"]
        lines.append("")
        lines.append(f"Notebook cells executed this run: {len(cells)}")
        last = cells[-1]
        status = "failed here" if last["success"] is False else "ok"
        lines.append(f"   [{last['execution_count']}] ({status}): {last['source_preview'].splitlines()[0][:80]}")

    if ev.get("recent_metrics"):
        lines.append("")
        lines.append("Recent metrics before failure:")
        for m in ev["recent_metrics"][-5:]:
            lines.append(f"   {m['name']} = {m['value']} (step {m['step']})")

    res = ev.get("resource_state_at_failure") or {}
    if res:
        lines.append("")
        lines.append("Resource state at failure:")
        if res.get("gpu_util"):
            lines.append(f"   GPU utilization: mean {res['gpu_util']['mean']:.0f}%, "
                          f"peak {res['gpu_util']['peak']:.0f}%")
        if res.get("vram_used_mib_peak"):
            lines.append(f"   Peak VRAM used: {res['vram_used_mib_peak']:.0f} MiB")

    if capsule.get("comparison_to_last_success"):
        comp = capsule["comparison_to_last_success"]
        lines.append("")
        lines.append(f"Nearest similar successful run: {comp['run_id']} "
                      f"(similarity {comp['similarity_score']*100:.0f}%)")

    if capsule.get("similar_previous_failures"):
        lines.append("")
        lines.append("Similar previous failures:")
        for s in capsule["similar_previous_failures"]:
            lines.append(f"   {s['run_id']}: {s['message'][:80]}")

    lines.append("")
    lines.append(f"Full traceback saved. Inspect with: watcher inspect {capsule['run_id']}")
    return "\n".join(lines)