"""Nearest-successful-run selection via a real, documented similarity score --
replacing the naive 'just pick the most recently successful run' baseline.

This is a transparent heuristic, not a learned model. Weights live in one
place (SIMILARITY_WEIGHTS) rather than buried in scoring logic, because "why
was this baseline chosen" is exactly the kind of thing a person needs to be
able to audit, not just trust. If you disagree with a weight, change the
constant -- don't reverse-engineer the behavior from output.
"""
from __future__ import annotations

import json
import subprocess
from typing import Optional

SIMILARITY_WEIGHTS = {
    "dataset_fingerprint": 0.30,
    "model_key": 0.20,
    "gpu": 0.15,
    "git_ancestry": 0.15,
    "config_distance": 0.10,
    "framework_versions": 0.05,
    "temporal_proximity": 0.05,
}

# Config keys checked for "model architecture" equality, first match wins.
_MODEL_KEYS = ("model", "architecture", "model_name")
# Packages checked for "framework version" equality (most framework versions
# live in env.packages, not in config).
_FRAMEWORK_PACKAGES = ("torch", "tensorflow", "transformers", "jax")

# Config keys ranked as more likely to be failure-relevant when they differ --
# a documented heuristic (memory/throughput-sensitive knobs), not a learned
# importance model.
_HIGH_RELEVANCE_KEYS = {
    "batch_size", "image_size", "sequence_length", "resolution",
    "precision", "mixed_precision", "gradient_accumulation_steps",
    "gradient_checkpointing", "learning_rate", "lr",
}


def _is_git_ancestor(ancestor_commit: Optional[str], descendant_commit: Optional[str],
                      cwd: str = ".") -> bool:
    if not ancestor_commit or not descendant_commit:
        return False
    if ancestor_commit == descendant_commit:
        return True
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor_commit, descendant_commit],
            cwd=cwd, capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _config_distance(config_a: dict, config_b: dict) -> float:
    """Fraction of matching key/value pairs among the union of keys.
    1.0 = identical, 0.0 = nothing in common (or nothing to compare)."""
    keys = set(config_a) | set(config_b)
    if not keys:
        return 1.0
    matches = sum(
        1 for k in keys
        if k in config_a and k in config_b and config_a[k] == config_b[k]
    )
    return matches / len(keys)


def _gpu_name(gpu_info: dict) -> Optional[str]:
    gpus = (gpu_info or {}).get("gpus") or []
    return gpus[0].get("name") if gpus else None


def _framework_versions_match(env_a: dict, env_b: dict) -> bool:
    pkgs_a = (env_a or {}).get("packages", {}) or {}
    pkgs_b = (env_b or {}).get("packages", {}) or {}
    found_shared = False
    for pkg in _FRAMEWORK_PACKAGES:
        if pkg in pkgs_a and pkg in pkgs_b:
            found_shared = True
            if pkgs_a[pkg] != pkgs_b[pkg]:
                return False
    return found_shared  # never claim a match if we never found a shared package


def score_similarity(candidate: dict, target: dict) -> dict:
    """candidate, target: dicts with keys 'config', 'dataset_fingerprint',
    'git' ({'commit': ...}), 'gpu', 'env', 'started_at'.

    Returns {"similarity_score": 0-1, "checklist": [...], "config_distance": 0-1}.
    The checklist is the 'why selected' explanation -- always returned, even
    for a low-scoring pair, so a caller can show why a match was weak too.
    """
    checklist = []
    score = 0.0

    dataset_match = bool(candidate.get("dataset_fingerprint")) and \
        candidate.get("dataset_fingerprint") == target.get("dataset_fingerprint")
    checklist.append({"label": "Same dataset fingerprint", "matched": dataset_match})
    if dataset_match:
        score += SIMILARITY_WEIGHTS["dataset_fingerprint"]

    config_a, config_b = candidate.get("config") or {}, target.get("config") or {}
    model_match = False
    for key in _MODEL_KEYS:
        if key in config_a and key in config_b:
            model_match = config_a[key] == config_b[key]
            break
    checklist.append({"label": "Same model architecture", "matched": model_match})
    if model_match:
        score += SIMILARITY_WEIGHTS["model_key"]

    gpu_a, gpu_b = _gpu_name(candidate.get("gpu") or {}), _gpu_name(target.get("gpu") or {})
    gpu_match = bool(gpu_a) and gpu_a == gpu_b
    checklist.append({"label": "Same GPU", "matched": gpu_match})
    if gpu_match:
        score += SIMILARITY_WEIGHTS["gpu"]

    git_a, git_b = candidate.get("git") or {}, target.get("git") or {}
    ancestor = _is_git_ancestor(git_a.get("commit"), git_b.get("commit"))
    checklist.append({
        "label": "Direct Git ancestor" if ancestor else "Git ancestry",
        "matched": ancestor,
    })
    if ancestor:
        score += SIMILARITY_WEIGHTS["git_ancestry"]

    distance = _config_distance(config_a, config_b)
    total_keys = len(set(config_a) | set(config_b))
    matching_keys = round(distance * total_keys)
    checklist.append({
        "label": (f"{matching_keys} of {total_keys} configuration fields identical"
                  if total_keys else "No configuration fields to compare"),
        "matched": distance > 0.5,
        "detail": f"{matching_keys}/{total_keys}" if total_keys else None,
    })
    score += SIMILARITY_WEIGHTS["config_distance"] * distance

    framework_match = _framework_versions_match(candidate.get("env") or {}, target.get("env") or {})
    checklist.append({"label": "Same framework versions", "matched": framework_match})
    if framework_match:
        score += SIMILARITY_WEIGHTS["framework_versions"]

    ts_a, ts_b = candidate.get("started_at"), target.get("started_at")
    temporal_score = 0.0
    if ts_a and ts_b:
        days_apart = abs(ts_a - ts_b) / 86400
        temporal_score = 1 / (1 + days_apart)
    score += SIMILARITY_WEIGHTS["temporal_proximity"] * temporal_score

    return {
        "similarity_score": round(min(score, 1.0), 3),
        "checklist": checklist,
        "config_distance": round(distance, 3),
    }


def rank_relevant_differences(config_a: dict, config_b: dict) -> list:
    """Config diffs from a (baseline) to b (target), ranked by heuristic
    relevance -- keys commonly tied to OOM/throughput/accuracy sensitivity
    surface first. Documented heuristic, not a learned importance model."""
    keys = sorted(set(config_a) | set(config_b))
    diffs = []
    for k in keys:
        if config_a.get(k) != config_b.get(k):
            diffs.append({
                "key": k, "from": config_a.get(k), "to": config_b.get(k),
                "high_relevance": k in _HIGH_RELEVANCE_KEYS,
            })
    diffs.sort(key=lambda d: not d["high_relevance"])
    return diffs


def _safe_json(s, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def run_snapshot_from_row(row) -> dict:
    """Build a similarity-comparable snapshot from a storage row. Public --
    reused by capsule.py for post-hoc comparisons (once a run's own row is
    fully persisted)."""
    return {
        "config": _safe_json(row["config_json"], {}),
        "dataset_fingerprint": row["dataset_fingerprint"],
        "git": _safe_json(row["git_json"], {}),
        "gpu": _safe_json(row["gpu_json"], {}),
        "env": _safe_json(row["env_json"], {}),
        "started_at": row["started_at"],
    }


def find_nearest_successful_run(storage, project: str, exclude_run_id: str,
                                 target_snapshot: dict) -> Optional[dict]:
    """The real replacement for 'just pick the last successful run'. Scans
    every successful run in the project, scores each against target_snapshot,
    and returns the highest-similarity match with its full explanation.

    target_snapshot has the same shape _run_snapshot() produces -- callers
    comparing against a run that failed (not itself in a clean 'success'
    state) should build this from that run's failure-capsule evidence.

    Returns None if there are no successful runs to compare against.
    """
    rows = storage.list_runs(project=project)
    best = None
    best_score = -1.0
    for row in rows:
        if row["run_id"] == exclude_run_id or row["exit_status"] != "success":
            continue
        candidate_snapshot = run_snapshot_from_row(row)
        result = score_similarity(candidate_snapshot, target_snapshot)
        # list_runs is DESC by started_at, so the first candidate encountered
        # at a tied score is the more recent one -- a sensible tiebreaker.
        if result["similarity_score"] > best_score:
            best_score = result["similarity_score"]
            best = {
                "run_id": row["run_id"],
                "similarity_score": result["similarity_score"],
                "checklist": result["checklist"],
                "config": candidate_snapshot["config"],
                "failure_relevant_differences": rank_relevant_differences(
                    candidate_snapshot["config"], target_snapshot.get("config") or {}),
            }
    return best