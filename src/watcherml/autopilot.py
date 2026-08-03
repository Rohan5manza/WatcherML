"""Autopilot: an OPT-IN, bounded loop that runs your training function repeatedly,
using either the Ollama advisor or a deterministic fallback heuristic to propose the
next config after each iteration.

This is explicitly labeled experimental. Your own product spec is right to flag that
a handful of runs is not much evidence for an LLM (or a human) to draw conclusions
from -- so this does not silently retrain a model and ship it. It:

  - runs a bounded number of iterations (default 5, hard cap enforced)
  - logs every iteration as a completely normal, independently inspectable WatcherML run
  - never modifies anything outside WatcherML's own storage
  - stops and tells you why, rather than looping forever chasing marginal gains
  - falls back to a simple deterministic heuristic if Ollama isn't running, so the
    loop still terminates sensibly without an LLM

Usage:
    from watcherml.autopilot import autopilot

    def train(config):
        # your existing training code, parameterized by `config`
        ...
        return {"val_accuracy": acc}   # or raise on failure, e.g. CUDA OOM

    result = autopilot(
        project="tomato-disease",
        train_fn=train,
        base_config={"model": "resnet50", "lr": 1e-3, "batch_size": 32},
        goal_metric="val_accuracy",
        goal_direction="maximize",
        max_iterations=5,
    )
"""
from __future__ import annotations

import sys
import traceback as tb_module
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import advisor
from .diff import compare_runs
from .run import Run
from .storage import Storage

HARD_ITERATION_CAP = 10  # even if a caller passes a larger max_iterations


@dataclass
class AutopilotResult:
    run_ids: list = field(default_factory=list)
    best_run_id: Optional[str] = None
    best_metric_value: Optional[float] = None
    stopped_reason: str = ""


def _cuda_oom_fallback(config: dict) -> dict:
    """Deterministic fallback used when Ollama is unavailable: halve batch_size on OOM."""
    new_config = dict(config)
    bs = new_config.get("batch_size")
    if isinstance(bs, (int, float)) and bs > 1:
        new_config["batch_size"] = max(1, int(bs) // 2)
    return new_config


def _better(direction: str, a: Optional[float], b: Optional[float]) -> bool:
    """Is b better than a?"""
    if b is None:
        return False
    if a is None:
        return True
    return b > a if direction == "maximize" else b < a


def autopilot(
    project: str,
    train_fn: Callable[[dict], dict],
    base_config: dict,
    goal_metric: str,
    goal_direction: str = "maximize",
    max_iterations: int = 5,
    model: str = advisor.DEFAULT_MODEL,
    host: str = advisor.DEFAULT_HOST,
    storage: Optional[Storage] = None,
) -> AutopilotResult:
    if goal_direction not in ("maximize", "minimize"):
        raise ValueError("goal_direction must be 'maximize' or 'minimize'")

    max_iterations = min(max_iterations, HARD_ITERATION_CAP)
    storage = storage or Storage()
    use_llm = advisor.is_available(host=host)

    print(f"WatcherML autopilot: up to {max_iterations} iterations "
          f"({'Ollama-guided' if use_llm else 'deterministic fallback, Ollama not detected'}). "
          f"Every iteration is logged as a normal run you can inspect independently.\n")

    result = AutopilotResult()
    config = dict(base_config)
    history = []  # what we hand to the LLM / print to the user

    for i in range(1, max_iterations + 1):
        print(f"--- autopilot iteration {i}/{max_iterations} ---")
        run = Run(project=project, config=config, storage=storage)
        run.start()
        result.run_ids.append(run.run_id)

        entry = {"run_id": run.run_id, "config": config, "status": None,
                  "metrics": None, "failure_rule": None}

        try:
            metrics = train_fn(config) or {}
            for name, value in metrics.items():
                run.log_metric(name, value)
            run._finish_success()
            entry["status"] = "success"
            entry["metrics"] = metrics

            value = metrics.get(goal_metric)
            if _better(goal_direction, result.best_metric_value, value):
                result.best_metric_value = value
                result.best_run_id = run.run_id

        except Exception as exc:
            exc_type, exc_value, exc_tb = sys.exc_info()
            run._finish_failure(exc_type, exc_value, exc_tb)
            entry["status"] = "failed"
            failure_row = storage.get_failure(run.run_id)
            if failure_row is not None:
                import json as _json
                entry["failure_rule"] = _json.loads(failure_row["diagnosis_json"]).get("rule")

        history.append(entry)

        if i == max_iterations:
            result.stopped_reason = "reached max_iterations"
            break

        # -- decide the next config -----------------------------------------
        suggestion = None
        if use_llm:
            suggestion = advisor.suggest_next_config(
                run_history=history, goal_metric=goal_metric, goal_direction=goal_direction,
                model=model, host=host,
            )

        if suggestion and suggestion.get("config"):
            next_config = {**config, **suggestion["config"]}
            rationale = suggestion.get("rationale", "")
            print(f"  next config (AI-suggested): {suggestion['config']}  -- {rationale}")
        elif entry["failure_rule"] == "cuda_out_of_memory":
            next_config = _cuda_oom_fallback(config)
            print(f"  next config (deterministic fallback, halving batch_size): {next_config}")
        elif entry["status"] == "failed":
            result.stopped_reason = (
                f"iteration {i} failed ({entry['failure_rule'] or 'unclassified'}) "
                "and no fallback rule applies -- stopping rather than guessing."
            )
            print(f"  {result.stopped_reason}")
            break
        else:
            result.stopped_reason = (
                "no further suggestion available (Ollama not running and no "
                "deterministic fallback applies) -- stopping after a successful run "
                "rather than repeating the same config."
            )
            print(f"  {result.stopped_reason}")
            break

        if next_config == config:
            result.stopped_reason = "suggested config was identical to the current one -- stopping."
            print(f"  {result.stopped_reason}")
            break

        config = next_config
        print()

    if not result.stopped_reason:
        result.stopped_reason = "reached max_iterations"

    print(f"\nAutopilot stopped: {result.stopped_reason}")
    if result.best_run_id:
        print(f"Best run so far: {result.best_run_id}  ({goal_metric} = {result.best_metric_value})")
        print(f"Review it yourself: watcher inspect {result.best_run_id}")
        if len(result.run_ids) > 1:
            print(f"Compare the first and best run: watcher compare {result.run_ids[0]} {result.best_run_id}")
    print(
        "\nReminder: this ran on very few data points. Treat the result as a lead worth "
        "reviewing, not a decision -- inspect the best run yourself before trusting it."
    )
    return result
