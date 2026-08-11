"""Deterministic CUDA OOM recovery campaigns.

This module is deliberately narrow. It observes an already-recorded OOM
capsule, creates a bounded sequence of explainable configuration patches,
executes probe/full trials, and persists every outcome. It performs no network
calls and contains no model-generated diagnosis or planning( which is the plan for v2 of watcherml: to include any-llm and autopilot modes powered by agentic AI).

The current executor still invokes ``train_fn`` in the parent Python process.
That is an interim implementation, not process isolation. The v1 release must
replace ``_run_trial`` with the explicit-entrypoint subprocess trial runner
before claiming isolated or confirmation-verified recovery.

Only three changes are automatic in this interim policy: batch size, gradient
accumulation, and gradient checkpointing. Code, dependencies, datasets,
precision, sequence length, and worker count are never modified.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from .capsule import compare_to_last_success
from .run import Run
from .storage import Storage

HARD_TRIAL_CAP = 10  # enforced regardless of what a contract requests

ALLOWED_KEYS = {
    "batch_size",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
}

_KEY_SPECS = {
    "batch_size": {"type": int, "min": 1},
    "gradient_accumulation_steps": {"type": int, "min": 1},
    "gradient_checkpointing": {"type": bool},
}


@dataclass
class RecoveryContract:
    """The 'success contract' for a recovery campaign -- deliberately small
    for this MVP compared to the full YAML campaign spec in the product doc.
    Extend this, don't bypass it, when generalizing beyond OOM recovery."""
    goal_metric: Optional[str] = None          # e.g. "val_accuracy"; None = ignore in scoring
    goal_direction: str = "maximize"            # "maximize" or "minimize"
    target: Optional[float] = None              # acceptance threshold for goal_metric; if met, campaign stops early
    throughput_metric: Optional[str] = None     # e.g. "throughput_samples_per_sec"
    max_vram_gb: Optional[float] = None         # used for a headroom bonus in scoring, if known
    max_wall_minutes: Optional[float] = None    # summed trial wall-clock budget
    probe_steps: int = 30
    max_trials: int = 6
    max_candidates: int = 3


# Permission level per allowed key. Everything here is genuinely "automatic"
# today -- there is no approval workflow, no human-in-the-loop gate, and no
# code/dependency/dataset change capability at all yet. This dict exists so
# API consumers can see the true current state rather than assume a richer
# permissions model exists than actually does.
KEY_PERMISSIONS = {key: "automatic" for key in ALLOWED_KEYS}


# ============================================================================
# Policy engine. Nothing reaches a trial without passing here.
# ============================================================================

def validate_patch(patch: dict) -> tuple:
    """Filter a proposed patch down to allowed keys with valid values.

    Returns ``(cleaned_patch, rejected_keys)``. Never raises: a
    completely invalid patch just returns ({}, [all keys])."""
    cleaned: dict = {}
    rejected: list = []
    if not isinstance(patch, dict):
        return {}, ["<patch was not an object>"]
    for key, value in patch.items():
        spec = _KEY_SPECS.get(key)
        if spec is None:
            rejected.append(key)
            continue
        if spec["type"] is bool:
            if isinstance(value, bool):
                cleaned[key] = value
            else:
                rejected.append(key)
        elif spec["type"] is int:
            try:
                v = int(value)
            except (TypeError, ValueError):
                rejected.append(key)
                continue
            if v < spec.get("min", float("-inf")):
                rejected.append(key)
            else:
                cleaned[key] = v
        elif spec["type"] is str:
            if isinstance(value, str) and value in spec.get("choices", {value}):
                cleaned[key] = value
            else:
                rejected.append(key)
    return cleaned, rejected


# ============================================================================
# Observer -- recorded facts only
# ============================================================================

def observe(storage: Storage, run_id: str) -> dict:
    """Build a factual observation from a persisted failure capsule."""
    row = storage.get_run(run_id)
    if row is None:
        raise ValueError(f"Run '{run_id}' not found.")
    capsule = storage.get_failure_capsule(run_id)
    if capsule is None:
        raise ValueError(f"Run '{run_id}' did not fail -- nothing to recover.")

    classification = (
        capsule.get("classification")
        or capsule.get("diagnosis")
        or (capsule.get("failure") or {}).get("classification")
        or {}
    )
    evidence = capsule.get("evidence") or {}
    resource = evidence.get("resource_state_at_failure") or {}
    vram_peak_mib = resource.get("vram_used_mib_peak")
    recent_metrics = evidence.get("recent_metrics") or []
    training_state = evidence.get("training_state") or {}
    comparison = (
        capsule.get("nearest_successful_run")
        or capsule.get("comparison_to_last_success")
        or compare_to_last_success(storage, row["project"], run_id)
    )

    return {
        "run_id": run_id,
        "project": row["project"],
        "status": "failed",
        "failure_class": capsule.get("failure_class") or classification.get("rule"),
        "failure_step": (
            training_state.get("last_logged_step")
            if training_state.get("last_logged_step") is not None
            else recent_metrics[-1].get("step") if recent_metrics else None
        ),
        "peak_vram_gb": round(vram_peak_mib / 1024, 2) if vram_peak_mib else None,
        "config": evidence.get("config") or {},
        "gpu": evidence.get("gpu", {}),
        "nearest_successful_run": comparison["run_id"] if comparison else None,
        "nearest_successful_config": comparison["config"] if comparison else None,
    }


# ============================================================================
# Deterministic classification and intervention policy
# ============================================================================

def deterministic_hypotheses(observation: dict) -> list:
    """Explain the evidence-backed OOM mechanism without probabilistic claims."""
    config = observation.get("config") or {}
    details = []
    if config.get("batch_size") is not None:
        details.append(f"batch_size={config['batch_size']}")
    if config.get("gradient_accumulation_steps") is not None:
        details.append(
            "gradient_accumulation_steps="
            f"{config['gradient_accumulation_steps']}"
        )
    if observation.get("peak_vram_gb") is not None:
        details.append(f"captured_peak_vram_gb={observation['peak_vram_gb']}")
    suffix = f" Recorded evidence: {', '.join(details)}." if details else ""
    return [{
        "cause": "activation_memory_or_batch_size",
        "explanation": (
            "The deterministic CUDA OOM rule matched because the process "
            "exhausted available accelerator memory. The bounded v1 policy "
            "therefore tests changes that reduce stored activation memory while "
            "preserving effective batch size where possible."
            + suffix
        ),
        "confidence": None,
        "rule": "cuda_out_of_memory",
    }]


def deterministic_candidates(base_config: dict) -> list:
    """Return an ordered, bounded intervention ladder.

    Order is intentional: preserve effective batch size first, then trade
    compute for activation memory, then combine both only if budget permits.
    """
    candidates = []
    bs = base_config.get("batch_size")
    if isinstance(bs, (int, float)) and bs > 1:
        new_bs = max(1, int(bs) // 2)
        factor = max(1, int(bs) // new_bs)
        grad_accum = base_config.get("gradient_accumulation_steps", 1) or 1
        candidates.append({
            "patch": {"batch_size": new_bs, "gradient_accumulation_steps": int(grad_accum) * factor},
            "rationale": (
                "Reduce per-step activation memory by halving batch size while "
                "increasing gradient accumulation to preserve effective batch size."
            ),
            "confidence": None,
            "policy_rule": "halve_batch_preserve_effective_batch",
        })
    if not base_config.get("gradient_checkpointing"):
        candidates.append({
            "patch": {"gradient_checkpointing": True},
            "rationale": (
                "Enable gradient checkpointing to trade additional compute for "
                "lower retained activation memory."
            ),
            "confidence": None,
            "policy_rule": "enable_gradient_checkpointing",
        })
    if isinstance(bs, (int, float)) and bs > 1 and not base_config.get("gradient_checkpointing"):
        new_bs = max(1, int(bs) // 2)
        factor = max(1, int(bs) // new_bs)
        grad_accum = base_config.get("gradient_accumulation_steps", 1) or 1
        candidates.append({
            "patch": {
                "batch_size": new_bs,
                "gradient_accumulation_steps": int(grad_accum) * factor,
                "gradient_checkpointing": True,
            },
            "rationale": (
                "Combine the two bounded memory interventions only after the "
                "single-policy variants have been considered."
            ),
            "confidence": None,
            "policy_rule": "halve_batch_and_checkpoint",
        })
    return candidates


# ============================================================================
# Interim evaluator -- deterministic but not yet confirmation verification
# ============================================================================

def score_trial(metrics: dict, resource_summary: dict, contract: RecoveryContract) -> float:
    """Rank completed trials for review; this is not a recovery verdict.

    Confirmation verification will replace this heuristic as the release
    criterion in the later verifier step.
    """
    score = 1.0  # base credit for completing at all
    if contract.goal_metric and contract.goal_metric in metrics:
        v = metrics[contract.goal_metric]
        score += v if contract.goal_direction == "maximize" else -v
    if contract.throughput_metric and contract.throughput_metric in metrics:
        score += 0.1 * metrics[contract.throughput_metric]
    vram_peak_mib = (resource_summary or {}).get("vram_used_mib_peak")
    if vram_peak_mib and contract.max_vram_gb:
        headroom_gb = contract.max_vram_gb - (vram_peak_mib / 1024)
        score += 0.05 * headroom_gb
    return score


# ============================================================================
# Interim executor -- deterministic and inspectable, but still in-process
# ============================================================================

def _call_train_fn(train_fn: Callable, config: dict, max_steps: Optional[int]):
    """Probe trials need a short run. If train_fn doesn't accept max_steps,
    fall back to a full call -- probing just won't be cheap for that user."""
    try:
        params = inspect.signature(train_fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "max_steps" in params:
        return train_fn(config, max_steps=max_steps)
    return train_fn(config)


def _run_trial(project: str, config: dict, train_fn: Callable, max_steps: Optional[int],
               storage: Storage) -> tuple:
    """Run one interim in-process trial as a normal WatcherML run.

    Returns
    (run_id, outcome, metrics, resource_summary). outcome is "success" or the
    deterministic failure rule name (e.g. "cuda_out_of_memory"). This must be
    replaced by the subprocess trial runner before the v1 release.
    """
    run = Run(project=project, config=config, storage=storage)
    run.start()
    metrics: dict = {}
    outcome = "success"
    try:
        metrics = _call_train_fn(train_fn, config, max_steps) or {}
        for name, value in metrics.items():
            run.log_metric(name, value)
        run._finish_success()
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        run._finish_failure(exc_type, exc_value, exc_tb)
        failure_row = storage.get_failure(run.run_id)
        diagnosis = json.loads(failure_row["diagnosis_json"]) if failure_row else {}
        outcome = diagnosis.get("rule", "failed")
    row = storage.get_run(run.run_id)
    resource_summary = json.loads(row["resource_json"] or "{}") if row else {}
    return run.run_id, outcome, metrics, resource_summary


# ============================================================================
# Orchestrator
# ============================================================================

def recover_from_oom(
    project: str,
    failed_run_id: str,
    train_fn: Callable,
    contract: Optional[RecoveryContract] = None,
    storage: Optional[Storage] = None,
) -> dict:
    """Run one deterministic interim campaign against a recorded CUDA OOM.

    train_fn(config) -> metrics_dict, or raise on failure (e.g. CUDA OOM).
    For cheap probing, also accept train_fn(config, max_steps=N) -- if your
    signature doesn't take max_steps, probing silently falls back to full
    trials (still safe, just not cheap).

    A completed full trial is provisional. The later confirmation verifier is
    responsible for declaring a verified recovery.
    """
    storage = storage or Storage()
    contract = contract or RecoveryContract()

    if contract.max_trials < 1:
        raise ValueError("contract.max_trials must be at least 1")
    if contract.max_candidates < 1:
        raise ValueError("contract.max_candidates must be at least 1")
    if contract.probe_steps < 1:
        raise ValueError("contract.probe_steps must be at least 1")
    if contract.goal_direction not in {"maximize", "minimize"}:
        raise ValueError("contract.goal_direction must be 'maximize' or 'minimize'")

    observation = observe(storage, failed_run_id)
    if observation["failure_class"] != "cuda_out_of_memory":
        raise ValueError(
            f"Run '{failed_run_id}' has failure class "
            f"'{observation['failure_class']}', not 'cuda_out_of_memory'."
        )

    campaign_id = f"recovery-{uuid.uuid4().hex[:8]}"
    started_at = time.time()
    storage.create_recovery_campaign(campaign_id, project, failed_run_id, asdict(contract), started_at)
    trial_budget = min(contract.max_trials, HARD_TRIAL_CAP)

    print(f"WatcherML deterministic OOM campaign: {campaign_id} "
          f"(budget: {trial_budget} trials, {contract.probe_steps}-step probes)")
    print("Policy: deterministic v1 OOM intervention ladder\n")

    # -- 1. Deterministic classification ----------------------------------
    hypotheses = deterministic_hypotheses(observation)
    print("Evidence-backed diagnosis:")
    for h in hypotheses:
        print(f"  - {h.get('cause')}: {h.get('explanation', '')}")

    # -- 2. Deterministic intervention policy -----------------------------
    base_config = observation["config"]
    candidates = []
    rejected_count = 0
    for raw in deterministic_candidates(base_config):
        cleaned, rejected = validate_patch(raw.get("patch", {}))
        rejected_count += len(rejected)
        if cleaned:
            candidates.append({"patch": cleaned, "rationale": raw.get("rationale", ""),
                               "confidence": None, "hypotheses": hypotheses,
                               "policy_rule": raw.get("policy_rule")})
    candidates = candidates[:contract.max_candidates]

    print(f"\nBounded interventions ({rejected_count} key(s) rejected by policy):")
    for c in candidates:
        print(f"  - [{c['policy_rule']}] {c['patch']} -- {c['rationale']}")

    # -- 4. Executor: probe trials (cheap elimination) ----------------------
    print(f"\nRunning probe trials ({contract.probe_steps} steps each)...")
    trials_run = 0
    total_seconds = 0.0
    wall_budget_seconds = (
        contract.max_wall_minutes * 60 if contract.max_wall_minutes else None
    )
    budget_exhausted = False
    survivors = []
    for c in candidates:
        if trials_run >= trial_budget or (
            wall_budget_seconds and total_seconds >= wall_budget_seconds
        ):
            budget_exhausted = True
            break
        candidate_config = {**base_config, **c["patch"]}
        run_id, outcome, metrics, resource_summary = _run_trial(
            project, candidate_config, train_fn, contract.probe_steps, storage)
        trials_run += 1
        total_seconds += (storage.get_run(run_id)["duration_seconds"] or 0)
        score = score_trial(metrics, resource_summary, contract) if outcome == "success" else None
        storage.save_recovery_trial(campaign_id, run_id, "probe", c["hypotheses"], c["patch"],
                                     c["rationale"], c.get("confidence"), outcome, score,
                                     False, time.time())
        print(f"  probe {run_id}: {'survived' if outcome == 'success' else f'eliminated ({outcome})'}")
        if outcome == "success":
            survivors.append({**c, "config": candidate_config})

    # -- 5. Executor: full trials for survivors -----------------------------
    if survivors:
        print(f"\nRunning full trials for {len(survivors)} surviving candidate(s)...")
    results = []
    objective_met = False
    for s in survivors:
        if trials_run >= trial_budget or (
            wall_budget_seconds and total_seconds >= wall_budget_seconds
        ):
            budget_exhausted = True
            print("  budget exhausted before all survivors could run a full trial.")
            break
        run_id, outcome, metrics, resource_summary = _run_trial(
            project, s["config"], train_fn, None, storage)
        trials_run += 1
        total_seconds += (storage.get_run(run_id)["duration_seconds"] or 0)
        score = score_trial(metrics, resource_summary, contract) if outcome == "success" else None
        storage.save_recovery_trial(campaign_id, run_id, "full", s["hypotheses"], s["patch"],
                                     s["rationale"], s.get("confidence"), outcome, score,
                                     False, time.time())
        print(f"  full {run_id}: {'success' if outcome == 'success' else f'failed ({outcome})'}"
              + (f", score={score:.3f}" if score is not None else ""))
        if outcome == "success":
            results.append({"run_id": run_id, "patch": s["patch"], "rationale": s["rationale"],
                             "score": score, "metrics": metrics, "resource": resource_summary})
            # Real target enforcement: stop as soon as the acceptance threshold
            # is genuinely met, rather than always burning the full trial budget.
            if contract.target is not None and contract.goal_metric in metrics:
                value = metrics[contract.goal_metric]
                met = value >= contract.target if contract.goal_direction == "maximize" else value <= contract.target
                if met:
                    objective_met = True
                    break

    # -- 6. Select + report --------------------------------------------------
    best = max(results, key=lambda r: r["score"]) if results else None
    ended_at = time.time()
    stopped_reason = (
        "metric target met (provisional)" if objective_met else
        "no applicable deterministic intervention" if not candidates else
        "no candidate survived probing" if not survivors else
        "no surviving candidate completed a full trial" if not results else
        "wall-time or trial budget exhausted" if budget_exhausted else
        "all candidates evaluated (confirmation pending)"
    )

    # baseline_score: the goal metric's value on the nearest successful run,
    # if one was found and the metric was actually logged there. Never
    # fabricated -- None if we don't have a real number for it.
    baseline_score = None
    nearest_id = observation.get("nearest_successful_run")
    if nearest_id and contract.goal_metric:
        nearest_metrics = storage.final_metrics(nearest_id)
        baseline_score = nearest_metrics.get(contract.goal_metric)

    peak_vram_gb = None
    if best and best["resource"].get("vram_used_mib_peak"):
        peak_vram_gb = round(best["resource"]["vram_used_mib_peak"] / 1024, 2)

    report = {
        "campaign_id": campaign_id,
        "baseline_run_id": failed_run_id,
        "baseline_peak_vram_gb": observation.get("peak_vram_gb"),
        "baseline_score": baseline_score,
        "trials_run": trials_run,
        "trial_budget": trial_budget,
        "trial_seconds_used": round(total_seconds, 1),
        "wall_minutes_budget": contract.max_wall_minutes,
        "candidates_proposed": len(candidates),
        "rejected_patch_keys": rejected_count,
        "survivors": len(survivors),
        "stopped_reason": stopped_reason,
        "objective_met": False,
        "metric_target_met": objective_met,
        "target": contract.target,
        "permissions": dict(KEY_PERMISSIONS),
        "verification_status": "pending_confirmation" if best else "not_recovered",
        "best_run_id": None,
        "best_patch": None,
        "best_score": None,
        "best_metrics": None,
        "provisional_best_run_id": best["run_id"] if best else None,
        "provisional_best_patch": best["patch"] if best else None,
        "provisional_best_score": best["score"] if best else None,
        "provisional_best_metrics": best["metrics"] if best else None,
        "peak_vram_gb": peak_vram_gb,
    }
    storage.finish_recovery_campaign(campaign_id, ended_at, stopped_reason,
                                      None, report)

    _print_report(report)
    return report


def _print_report(report: dict):
    print(f"\nCampaign stopped: {report['stopped_reason']}\n")
    print("Baseline:")
    print(f"  run:            {report['baseline_run_id']}")
    print(f"  peak_vram_gb:   {report.get('baseline_peak_vram_gb')}")
    if report.get("baseline_score") is not None:
        print(f"  baseline_score: {report['baseline_score']}")
    print("  status:         cuda_out_of_memory\n")
    if report["provisional_best_run_id"]:
        print("Best completed trial (confirmation pending):")
        print(f"  run:            {report['provisional_best_run_id']}")
        print(f"  patch:          {report['provisional_best_patch']}")
        print(f"  ranking_score:  {report['provisional_best_score']:.3f}")
        if report.get("peak_vram_gb") is not None:
            print(f"  peak_vram_gb:   {report['peak_vram_gb']}")
        if report.get("target") is not None:
            print(
                f"  metric_target:  {report['target']}  "
                f"({'MET' if report['metric_target_met'] else 'not met'})"
            )
        if report.get("provisional_best_metrics"):
            for k, v in report["provisional_best_metrics"].items():
                print(f"  {k}: {v}")
        print(
            "\n  This is not a verified recovery yet. The subprocess confirmation "
            "verifier must pass before promotion."
        )
        print(
            "  Inspect it yourself: watcher inspect "
            f"{report['provisional_best_run_id']}"
        )
    else:
        print("No intervention completed a full trial within budget.")
    print(f"\nTrials executed: {report['trials_run']} / {report['trial_budget']}"
          + (f"   Trial wall time: {report['trial_seconds_used']:.0f}s / "
             f"{report['wall_minutes_budget']*60:.0f}s budget"
             if report.get("wall_minutes_budget") else ""))
    print(f"Candidates proposed: {report['candidates_proposed']} "
          f"({report['rejected_patch_keys']} proposed key(s) rejected by the policy engine)")
    print(
        "\nReminder: the provisional ranking is a heuristic combination of completion, "
        "your goal metric, throughput, and VRAM headroom. It is evidence for the "
        "next confirmation step, not a recovery verdict."
    )