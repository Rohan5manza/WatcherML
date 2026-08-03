"""Deterministic failure diagnosis rules.

These are the source of truth for failure capsules. An optional LLM layer may
later add a natural-language explanation, but it must never be required for
the capsule to be useful — these rules always run first and always produce
a result on their own.
"""
from __future__ import annotations

import re
from typing import Optional


def _rule_cuda_oom(exc_type, message, tb_str):
    if "out of memory" in message.lower() and "cuda" in message.lower():
        return {
            "rule": "cuda_out_of_memory",
            "evidence_categories": ["config", "resource_state_at_failure", "gpu"],
            "summary": "CUDA ran out of GPU memory during this run.",
            "likely_cause": "Batch size, model size, or sequence length exceeded available VRAM.",
            "suggested_actions": [
                "Reduce batch_size or gradient-accumulate instead.",
                "Reduce sequence length / image resolution.",
                "Check for tensors or activations not being freed between steps.",
                "Confirm no other process is holding GPU memory (nvidia-smi).",
            ],
        }
    return None


def _rule_nan_loss(exc_type, message, tb_str):
    if re.search(r"\bnan\b", message, re.IGNORECASE) or "exploding" in message.lower():
        return {
            "rule": "nan_or_exploding_loss",
            "evidence_categories": ["recent_metrics", "config"],
            "summary": "Training loss became NaN or diverged.",
            "likely_cause": "Learning rate too high, unstable numerics, or a bad batch (e.g. div-by-zero, log(0)).",
            "suggested_actions": [
                "Lower the learning rate or add warmup.",
                "Add gradient clipping.",
                "Check input normalization and label ranges.",
                "Verify loss function for edge cases (log of 0, division by 0).",
            ],
        }
    return None


def _rule_shape_mismatch(exc_type, message, tb_str):
    if re.search(r"size mismatch|shape.*(mismatch|expected)|dimension.*(mismatch|out of range)",
                 message, re.IGNORECASE):
        return {
            "rule": "tensor_shape_mismatch",
            "evidence_categories": ["config", "git"],
            "summary": "A tensor shape did not match what a layer or operation expected.",
            "likely_cause": "A model, batch, or config change altered an input/output shape somewhere upstream.",
            "suggested_actions": [
                "Compare the exact shapes in the traceback against the previous successful run.",
                "Check recent changes to batch construction, model architecture, or config.",
            ],
        }
    return None


def _rule_missing_file(exc_type, message, tb_str):
    if exc_type in ("FileNotFoundError",) or "no such file or directory" in message.lower():
        return {
            "rule": "missing_file_or_dataset_path",
            "evidence_categories": ["config", "env"],
            "summary": "A required file or dataset path could not be found.",
            "likely_cause": "Dataset path is wrong, dataset wasn't downloaded/mounted, or a relative path assumption broke.",
            "suggested_actions": [
                "Confirm the path exists on this machine.",
                "Check whether this run's dataset_fingerprint matches an expected dataset location.",
            ],
        }
    return None


def _rule_dataloader_worker(exc_type, message, tb_str):
    if "dataloader worker" in message.lower() or "workers exited unexpectedly" in message.lower():
        return {
            "rule": "dataloader_worker_failure",
            "evidence_categories": ["config", "env"],
            "summary": "A DataLoader worker process crashed.",
            "likely_cause": "An exception inside a dataset __getitem__, insufficient shared memory, or worker OOM.",
            "suggested_actions": [
                "Try num_workers=0 to surface the real underlying exception.",
                "Check /dev/shm size if running in a container.",
                "Inspect the dataset item that was being loaded around the failure.",
            ],
        }
    return None


def _rule_device_mismatch(exc_type, message, tb_str):
    if "expected all tensors to be on the same device" in message.lower() or \
       re.search(r"cuda:\d+.*cpu|cpu.*cuda:\d+", message.lower()):
        return {
            "rule": "device_mismatch",
            "evidence_categories": ["gpu", "config"],
            "summary": "Tensors on different devices (e.g. CPU vs GPU) were used together.",
            "likely_cause": "A tensor, model, or module wasn't moved to the same device as the rest of the computation.",
            "suggested_actions": [
                "Audit .to(device) calls for the model, inputs, and any newly added tensors/losses.",
            ],
        }
    return None


def _rule_dependency_or_cuda_compat(exc_type, message, tb_str):
    if exc_type in ("ImportError", "ModuleNotFoundError") or \
       "cuda driver version is insufficient" in message.lower() or \
       "compiled with" in message.lower():
        return {
            "rule": "dependency_or_cuda_compatibility",
            "evidence_categories": ["env", "gpu"],
            "summary": "A dependency or CUDA/driver version mismatch prevented the run from starting or continuing.",
            "likely_cause": "Package installed for a different CUDA/driver/Python version, or missing dependency.",
            "suggested_actions": [
                "Compare this run's environment snapshot against the last known-good run.",
                "Verify driver/CUDA toolkit version compatibility with the installed framework build.",
            ],
        }
    return None


RULES = [
    _rule_cuda_oom,
    _rule_nan_loss,
    _rule_shape_mismatch,
    _rule_missing_file,
    _rule_dataloader_worker,
    _rule_device_mismatch,
    _rule_dependency_or_cuda_compat,
]


def diagnose(exc_type: str, message: str, tb_str: str) -> dict:
    """Run every deterministic rule and return the first match, or an 'unclassified' result."""
    for rule in RULES:
        result = rule(exc_type, message or "", tb_str or "")
        if result:
            return result
    return {
        "rule": "unclassified",
        "summary": "This failure did not match a known deterministic pattern.",
        "likely_cause": None,
        "evidence_categories": [],
        "suggested_actions": [
            "Review the full traceback and evidence below.",
        ],
    }