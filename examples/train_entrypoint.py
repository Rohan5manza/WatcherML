"""Minimal importable training entrypoint for scripts, Jupyter, and Colab.

In Colab, create this file with ``%%writefile train_entrypoint.py`` before
starting a recovery campaign. Reconstruct models/data inside ``train``; do not
depend on notebook globals, closures, or an already-initialized CUDA process.
"""
from __future__ import annotations


def train(config: dict, max_steps: int | None = None) -> dict[str, float]:
    """Replace the body with real model/data construction and training."""
    configured_steps = int(config.get("training_steps", 100))
    steps_to_run = min(configured_steps, max_steps) if max_steps else configured_steps

    # model = build_model(config)
    # dataset = load_dataset(config)
    # metrics = run_training(model, dataset, config, steps_to_run)

    # Deterministic placeholder so the contract can be validated immediately.
    metrics = {
        "steps_completed": float(steps_to_run),
        "validation_loss": 0.0,
    }
    return metrics
