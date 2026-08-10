"""OOM Recovery Agent -- the first slice of WatcherML's autonomous experiment
recovery loop, scoped deliberately narrow per the product spec:

    Observe -> Hypothesize -> Intervene -> Evaluate -> Remember

This module is NOT a general autonomous optimizer. It does exactly one thing:
given a run that failed with a CUDA OOM, it proposes a small number of
candidate fixes restricted to six memory-relevant config keys, tests them
cheaply (short probe trials) before committing to full trials, and reports
which fix was actually verified to work -- never which one an LLM merely
suggested.

Role separation (this is deliberate, not incidental):
  - Observer       (`observe`)                  -- facts only, no LLM
  - Diagnostician  (`_get_llm_hypotheses`)      -- LiteLLM, ranks causes
  - Planner        (`_get_llm_patches`)         -- LiteLLM, proposes patches
  - Policy engine  (`validate_patch`)            -- deterministic, the only
                                                     thing standing between an
                                                     LLM's opinion and anything
                                                     actually running
  - Executor       (`_run_trial`)                -- deterministic, runs a real
                                                     WatcherML Run per trial
  - Evaluator      (`score_trial`)               -- deterministic, decides
                                                     what "better" means
  - Memory         (`storage.*_recovery_*`)      -- every hypothesis, patch,
                                                     and outcome persisted,
                                                     independently inspectable

Both LLM roles have deterministic fallbacks. The agent still runs end to end
with no LLM configured -- it just proposes more generic candidates.

Safety notes, deliberately narrow for this MVP:
  - The policy engine allow-lists exactly six keys (see ALLOWED_KEYS) and
    rejects everything else the LLM proposes, silently and countably.
  - This agent only ever changes config -- never code, dependencies, or
    datasets -- so no git worktree isolation is needed for this MVP (there is
    no code to isolate). That changes the moment code-patch trials are added.
  - Trial count is hard-capped regardless of what's requested (HARD_TRIAL_CAP).
  - The metrics your train_fn returns should be VALIDATION metrics. This
    agent compares candidates against each other using whatever you log --
    if that's test-set performance, you will be selecting a "best" config by
    repeatedly optimizing against your test set, which silently invalidates
    it. Evaluate the winner against a test set once, after the campaign ends,
    outside this loop.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import litellm  # <-- NEW: unified LLM interface

from .capsule import compare_to_last_success
from .run import Run
from .storage import Storage

HARD_TRIAL_CAP = 10  # enforced regardless of what a contract requests

ALLOWED_KEYS = {
    "batch_size", "gradient_accumulation_steps", "precision",
    "sequence_length", "gradient_checkpointing", "num_workers",
}

_KEY_SPECS = {
    "batch_size": {"type": int, "min": 1},
    "gradient_accumulation_steps": {"type": int, "min": 1},
    "precision": {"type": str, "choices": {"fp32", "fp16", "bf16"}},
    "sequence_length": {"type": int, "min": 1},
    "gradient_checkpointing": {"type": bool},
    "num_workers": {"type": int, "min": 0},
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
    max_gpu_hours: Optional[float] = None       # real budget cap -- enforced against summed trial wall-clock time
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
# Policy engine -- the one thing standing between an LLM's proposal and
# anything actually executing. Nothing reaches a trial without passing here.
# ============================================================================

def validate_patch(patch: dict) -> tuple:
    """Filter an LLM-proposed patch down to only allowed keys with valid
    values. Returns (cleaned_patch, rejected_keys). Never raises -- a
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
# Observer -- facts only, no LLM involved
# ============================================================================

def observe(storage: Storage, run_id: str) -> dict:
    """Build a factual observation report from an already-computed failure
    capsule. Raises ValueError if the run didn't fail (nothing to observe)."""
    row = storage.get_run(run_id)
    if row is None:
        raise ValueError(f"Run '{run_id}' not found.")
    failure = storage.get_failure(run_id)
    if failure is None:
        raise ValueError(f"Run '{run_id}' did not fail -- nothing to recover.")

    diagnosis = json.loads(failure["diagnosis_json"] or "{}")
    evidence = json.loads(failure["evidence_json"] or "{}")
    resource = evidence.get("resource_state_at_failure") or {}
    vram_peak_mib = resource.get("vram_used_mib_peak")
    recent_metrics = evidence.get("recent_metrics") or []
    comparison = compare_to_last_success(storage, row["project"], run_id)

    return {
        "run_id": run_id,
        "project": row["project"],
        "status": "failed",
        "failure_class": diagnosis.get("rule"),
        "failure_step": recent_metrics[-1]["step"] if recent_metrics else None,
        "peak_vram_gb": round(vram_peak_mib / 1024, 2) if vram_peak_mib else None,
        "config": evidence.get("config", {}),
        "gpu": evidence.get("gpu", {}),
        "nearest_successful_run": comparison["run_id"] if comparison else None,
        "nearest_successful_config": comparison["config"] if comparison else None,
    }


# ============================================================================
# Deterministic fallbacks for the Diagnostician and Planner roles, used when
# LiteLLM is unavailable or returns nothing usable. The agent must still work.
# ============================================================================

def _fallback_hypotheses(observation: dict) -> list:
    return [{
        "cause": "activation_memory_or_batch_size",
        "explanation": (
            "Generic OOM pattern: batch size and/or activation memory likely "
            "exceeded available VRAM (deterministic fallback -- LiteLLM unavailable)."
        ),
        "confidence": 0.5,
    }]


def _fallback_candidates(base_config: dict) -> list:
    candidates = []
    bs = base_config.get("batch_size")
    if isinstance(bs, (int, float)) and bs > 1:
        new_bs = max(1, int(bs) // 2)
        factor = max(1, int(bs) // new_bs)
        grad_accum = base_config.get("gradient_accumulation_steps", 1) or 1
        candidates.append({
            "patch": {"batch_size": new_bs, "gradient_accumulation_steps": int(grad_accum) * factor},
            "rationale": "Halve batch size, compensate with gradient accumulation (deterministic fallback).",
            "confidence": None,
        })
    if base_config.get("precision") != "bf16":
        candidates.append({
            "patch": {"precision": "bf16"},
            "rationale": "Switch to bf16 to reduce activation memory footprint (deterministic fallback).",
            "confidence": None,
        })
    if not base_config.get("gradient_checkpointing"):
        candidates.append({
            "patch": {"gradient_checkpointing": True},
            "rationale": "Enable gradient checkpointing to trade compute for memory (deterministic fallback).",
            "confidence": None,
        })
    return candidates


# ============================================================================
# NEW: LiteLLM-based Diagnostician & Planner
# ============================================================================

def _llm_query(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> Optional[str]:
    """Send a prompt to LiteLLM and return the raw text response.
    Returns None if anything fails."""
    try:
        # LiteLLM uses standard OpenAI-compatible env vars by default:
        # OPENAI_API_KEY, OPENAI_API_BASE, etc.
        # Pass api_base explicitly if provided.
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if api_base:
            kwargs["api_base"] = api_base

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"[WatcherML] LiteLLM query failed: {e}")
        return None


def _get_llm_hypotheses(
    observation: dict,
    model: str,
    api_base: Optional[str] = None,
) -> Optional[list]:
    """Ask LiteLLM to rank possible causes of the OOM failure.
    Expects a JSON array: [{"cause": "...", "explanation": "...", "confidence": 0.0}]."""
    system = (
        "You are an expert ML debugging assistant. Your output must be **valid JSON only**."
        " No markdown, no prose outside the JSON. "
        "Return a JSON array of objects, each with keys: 'cause' (string), "
        "'explanation' (string), and 'confidence' (float between 0 and 1)."
    )
    user = (
        f"Failure: CUDA OOM. Trace: {observation.get('failure_step') or 'No step info'}.\n"
        f"GPU: {observation.get('gpu', {})}\n"
        f"Config at failure: {observation.get('config', {})}\n"
        f"Peak VRAM: {observation.get('peak_vram_gb')} GB\n"
        f"Nearest successful run config (if any): {observation.get('nearest_successful_config', 'None')}\n"
        "List the top 3 likely causes with confidence scores."
    )
    raw = _llm_query(system, user, model, api_base)
    if not raw:
        return None
    try:
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        return None


def _get_llm_patches(
    observation: dict,
    hypotheses: list,
    base_config: dict,
    model: str,
    api_base: Optional[str] = None,
) -> Optional[list]:
    """Ask LiteLLM to propose config patches based on the top hypotheses.
    Expects a JSON array: [{"patch": {"batch_size": 16}, "rationale": "...", "confidence": 0.0}]."""
    system = (
        "You are an expert ML debugging assistant. Your output must be **valid JSON only**."
        " No markdown, no prose outside the JSON. "
        "Return a JSON array of objects, each with keys: 'patch' (object with key:value pairs), "
        "'rationale' (string), and 'confidence' (float between 0 and 1). "
        f"You may only propose changes to these keys: {list(ALLOWED_KEYS)}."
    )
    user = (
        f"Failure: CUDA OOM. Config: {base_config}\n"
        f"Hypotheses: {hypotheses}\n"
        "Propose up to 3 distinct config patches that could fix the OOM. "
        "Prioritize safe, conservative changes. If a hypothesis suggests changing a key "
        "not in the allowed list, propose an alternative allowed key that would have a similar effect."
    )
    raw = _llm_query(system, user, model, api_base)
    if not raw:
        return None
    try:
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        return None


# ============================================================================
# Evaluator -- deterministic. The LLM never decides what "better" means.
# ============================================================================

def score_trial(metrics: dict, resource_summary: dict, contract: RecoveryContract) -> float:
    """Heuristic combination of success + goal metric + throughput + VRAM
    headroom. This is a documented heuristic, not a scientifically precise
    utility function -- tune the weights for your situation if you use this
    seriously. Only called for trials that didn't fail."""
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
# Executor -- deterministic. Runs a real, independently-inspectable Run per trial.
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
    """Runs one trial as a normal WatcherML Run. Returns
    (run_id, outcome, metrics, resource_summary). outcome is "success" or the
    deterministic failure rule name (e.g. "cuda_out_of_memory")."""
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
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    storage: Optional[Storage] = None,
) -> dict:
    """Run one OOM recovery campaign against a specific failed run.

    train_fn(config) -> metrics_dict, or raise on failure (e.g. CUDA OOM).
    For cheap probing, also accept train_fn(config, max_steps=N) -- if your
    signature doesn't take max_steps, probing silently falls back to full
    trials (still safe, just not cheap).

    Args:
        model: LiteLLM model string (e.g. "openai/gpt-4o-mini", "ollama/llama3",
               "groq/llama-3.1-70b-versatile", "claude-3-haiku-20240307").
               Defaults to env var WATCHER_LLM_MODEL or "openai/gpt-4o-mini".
        api_base: Optional custom API endpoint. If not set, LiteLLM uses
                  the default for the provider (e.g. OPENAI_API_BASE env var).
    """
    storage = storage or Storage()
    contract = contract or RecoveryContract()

    # Determine model and API base from env or defaults
    if model is None:
        model = os.getenv("WATCHER_LLM_MODEL", "openai/gpt-4o-mini")
    if api_base is None:
        api_base = os.getenv("OPENAI_API_BASE") or os.getenv("WATCHER_LLM_API_BASE")

    campaign_id = f"recovery-{uuid.uuid4().hex[:8]}"
    started_at = time.time()
    storage.create_recovery_campaign(campaign_id, project, failed_run_id, asdict(contract), started_at)
    trial_budget = min(contract.max_trials, HARD_TRIAL_CAP)

    print(f"WatcherML OOM recovery agent: campaign {campaign_id} "
          f"(budget: {trial_budget} trials, {contract.probe_steps}-step probes)")
    print(f"LLM backend: {model}" + (f" (base: {api_base})" if api_base else "") + "\n")

    # -- 1. Observer ------------------------------------------------------
    observation = observe(storage, failed_run_id)
    if observation["failure_class"] != "cuda_out_of_memory":
        print(f"Warning: '{failed_run_id}' was diagnosed as "
              f"'{observation['failure_class']}', not cuda_out_of_memory. This agent "
              "is scoped to OOM recovery only -- proceeding, but hypotheses may not fit.\n")

    # -- 2. Diagnostician (LiteLLM) ------------------------------------------
    hypotheses = _get_llm_hypotheses(observation, model, api_base)
    used_llm_diagnosis = hypotheses is not None
    if not hypotheses:
        hypotheses = _fallback_hypotheses(observation)
        print(f"Hypotheses (deterministic fallback — LiteLLM unavailable):")
    else:
        print(f"Hypotheses (LiteLLM — {model}):")
    for h in hypotheses:
        print(f"  - {h.get('cause')} (confidence {h.get('confidence')}): {h.get('explanation', '')}")

    # -- 3. Planner + policy engine (LiteLLM) ------------------------------
    base_config = observation["config"]
    raw_candidates = _get_llm_patches(observation, hypotheses, base_config, model, api_base)
    used_llm_planner = raw_candidates is not None
    candidates = []
    rejected_count = 0
    for raw in (raw_candidates or []):
        cleaned, rejected = validate_patch(raw.get("patch", {}))
        rejected_count += len(rejected)
        if cleaned:
            candidates.append({"patch": cleaned, "rationale": raw.get("rationale", ""),
                                "confidence": raw.get("confidence"), "hypotheses": hypotheses})
    if not candidates:
        used_llm_planner = False
        for raw in _fallback_candidates(base_config):
            cleaned, _ = validate_patch(raw["patch"])
            if cleaned:
                candidates.append({"patch": cleaned, "rationale": raw["rationale"],
                                    "confidence": raw["confidence"], "hypotheses": hypotheses})
    candidates = candidates[:contract.max_candidates]

    print(f"\nCandidates ({'LiteLLM, policy-validated' if used_llm_planner else 'deterministic fallback'}, "
          f"{rejected_count} proposed key(s) rejected by policy engine):")
    for c in candidates:
        print(f"  - {c['patch']}  -- {c['rationale']}")

    # -- 4. Executor: probe trials (cheap elimination) ----------------------
    print(f"\nRunning probe trials ({contract.probe_steps} steps each)...")
    trials_run = 0
    total_seconds = 0.0
    gpu_budget_seconds = contract.max_gpu_hours * 3600 if contract.max_gpu_hours else None
    budget_exhausted = False
    survivors = []
    for c in candidates:
        if trials_run >= trial_budget or (gpu_budget_seconds and total_seconds >= gpu_budget_seconds):
            budget_exhausted = trials_run >= trial_budget or bool(gpu_budget_seconds)
            break
        candidate_config = {**base_config, **c["patch"]}
        run_id, outcome, metrics, resource_summary = _run_trial(
            project, candidate_config, train_fn, contract.probe_steps, storage)
        trials_run += 1
        total_seconds += (storage.get_run(run_id)["duration_seconds"] or 0)
        score = score_trial(metrics, resource_summary, contract) if outcome == "success" else None
        storage.save_recovery_trial(campaign_id, run_id, "probe", c["hypotheses"], c["patch"],
                                     c["rationale"], c.get("confidence"), outcome, score,
                                     outcome == "success", time.time())
        print(f"  probe {run_id}: {'survived' if outcome == 'success' else f'eliminated ({outcome})'}")
        if outcome == "success":
            survivors.append({**c, "config": candidate_config})

    # -- 5. Executor: full trials for survivors -----------------------------
    if survivors:
        print(f"\nRunning full trials for {len(survivors)} surviving candidate(s)...")
    results = []
    objective_met = False
    for s in survivors:
        if trials_run >= trial_budget or (gpu_budget_seconds and total_seconds >= gpu_budget_seconds):
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
                                     outcome == "success", time.time())
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
        "objective target met" if objective_met else
        "no candidate survived probing" if not survivors else
        "no surviving candidate completed a full trial" if not results else
        "gpu-hour or trial budget exhausted" if budget_exhausted else
        "all candidates evaluated"
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
        "gpu_seconds_used": round(total_seconds, 1),
        "gpu_hours_budget": contract.max_gpu_hours,
        "candidates_proposed": len(candidates),
        "rejected_patch_keys": rejected_count,
        "survivors": len(survivors),
        "stopped_reason": stopped_reason,
        "objective_met": objective_met,
        "target": contract.target,
        "permissions": dict(KEY_PERMISSIONS),
        "best_run_id": best["run_id"] if best else None,
        "best_patch": best["patch"] if best else None,
        "best_score": best["score"] if best else None,
        "best_metrics": best["metrics"] if best else None,
        "peak_vram_gb": peak_vram_gb,
    }
    storage.finish_recovery_campaign(campaign_id, ended_at, stopped_reason,
                                      best["run_id"] if best else None, report)

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
    if report["best_run_id"]:
        print("Best verified trial:")
        print(f"  run:            {report['best_run_id']}")
        print(f"  patch:          {report['best_patch']}")
        print(f"  score:          {report['best_score']:.3f}")
        if report.get("peak_vram_gb") is not None:
            print(f"  peak_vram_gb:   {report['peak_vram_gb']}")
        if report.get("target") is not None:
            print(f"  target:         {report['target']}  ({'MET' if report['objective_met'] else 'not met'})")
        if report.get("best_metrics"):
            for k, v in report["best_metrics"].items():
                print(f"  {k}: {v}")
        print(f"\n  Inspect it yourself: watcher inspect {report['best_run_id']}")
    else:
        print("No candidate produced a verified fix within budget.")
    print(f"\nTrials executed: {report['trials_run']} / {report['trial_budget']}"
          + (f"   GPU time used: {report['gpu_seconds_used']:.0f}s / "
             f"{report['gpu_hours_budget']*3600:.0f}s budget" if report.get("gpu_hours_budget") else ""))
    print(f"Candidates proposed: {report['candidates_proposed']} "
          f"({report['rejected_patch_keys']} proposed key(s) rejected by the policy engine)")
    print(
        "\nReminder: 'best' here is a heuristic combination of success, your goal metric, "
        "throughput, and VRAM headroom, scored on whatever metrics your train_fn returned. "
        "If those were validation metrics (as they should be), this pick is a lead worth "
        "reviewing -- not a decision. Evaluate it against your test set once, separately, "
        "outside this loop."
    )