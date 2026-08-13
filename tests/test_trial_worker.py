"""Acceptance tests for the one-trial subprocess worker."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from watcherml.entrypoint import TrainingEntrypoint
from watcherml.storage import Storage
from watcherml.trial_protocol import (
    EXIT_CONTRACT_ERROR,
    EXIT_SUCCESS,
    EXIT_TRAINING_FAILED,
    TRIAL_RESULT_SCHEMA_VERSION,
    TrialProtocolError,
    TrialRequest,
    load_result,
    write_request,
)


def _write_module(project_root: Path, name: str, source: str) -> None:
    (project_root / f"{name}.py").write_text(source, encoding="utf-8")


def _run_worker(tmp_path: Path, request: TrialRequest):
    project_root = tmp_path / "project"
    storage_root = tmp_path / "watcher-data"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    project_root.mkdir(exist_ok=True)
    write_request(request_path, request)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "watcherml._trial_worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            "--project-root",
            str(project_root),
            "--storage-root",
            str(storage_root),
        ],
        cwd=project_root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result_path.exists(), completed.stderr
    return completed, load_result(result_path), project_root, storage_root, result_path


def test_successful_probe_runs_in_fresh_process_and_honors_working_directory(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "relative-value.txt").write_text("0.125", encoding="utf-8")
    _write_module(
        project_root,
        "train_success",
        """
import os

def main(config, max_steps=None):
    with open("relative-value.txt", encoding="utf-8") as handle:
        loss = float(handle.read())
    return {
        "loss": loss,
        "steps_completed": max_steps,
        "worker_pid_metric": os.getpid(),
    }
""",
    )
    request = TrialRequest(
        trial_id="trial-success",
        campaign_id="campaign-smoke",
        source_run_id="source-run",
        project="worker-smoke",
        phase="probe",
        entrypoint=TrainingEntrypoint("train_success:main"),
        config={"batch_size": 8},
        max_steps=3,
    )
    completed, result, _, storage_root, result_path = _run_worker(tmp_path, request)

    assert completed.returncode == EXIT_SUCCESS
    assert result.status == "success"
    assert result.worker_pid != os.getpid()
    assert result.metrics["loss"] == 0.125
    assert result.metrics["steps_completed"] == 3.0
    assert result.metrics["worker_pid_metric"] == float(result.worker_pid)
    assert result.to_dict()["schema"]["version"] == TRIAL_RESULT_SCHEMA_VERSION
    assert result.run_id is not None

    storage = Storage(root=str(storage_root))
    row = storage.get_run(result.run_id)
    assert row is not None
    assert row["exit_status"] == "success"
    assert storage.final_metrics(result.run_id)["loss"] == 0.125
    assert not list(result_path.parent.glob(f".{result_path.name}.*.tmp"))


def test_cuda_oom_becomes_training_failure_with_persisted_capsule(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_module(
        project_root,
        "train_oom",
        """
def main(config, max_steps=None):
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
""",
    )
    request = TrialRequest(
        trial_id="trial-oom",
        project="worker-smoke",
        phase="probe",
        entrypoint=TrainingEntrypoint("train_oom:main"),
        config={"batch_size": 32, "gradient_accumulation_steps": 1},
        max_steps=3,
    )
    completed, result, _, storage_root, _ = _run_worker(tmp_path, request)

    assert completed.returncode == EXIT_TRAINING_FAILED
    assert result.status == "training_failed"
    assert result.failure_class == "cuda_out_of_memory"
    assert result.capsule_schema_version == "1.0"
    assert result.error["type"] == "RuntimeError"
    assert result.run_id is not None

    storage = Storage(root=str(storage_root))
    row = storage.get_run(result.run_id)
    assert row["exit_status"] == "failed"
    capsule = storage.get_failure_capsule(result.run_id)
    assert capsule["failure_class"] == "cuda_out_of_memory"


def test_probe_rejects_unbounded_entrypoint_before_creating_run(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_module(
        project_root,
        "train_unbounded",
        "def main(config):\n    return {'loss': 1.0}\n",
    )
    request = TrialRequest(
        trial_id="trial-invalid",
        project="worker-smoke",
        phase="probe",
        entrypoint=TrainingEntrypoint("train_unbounded:main"),
        config={},
        max_steps=3,
    )
    completed, result, _, storage_root, _ = _run_worker(tmp_path, request)

    assert completed.returncode == EXIT_CONTRACT_ERROR
    assert result.status == "contract_error"
    assert result.run_id is None
    assert "will not silently run full training" in result.error["message"]
    storage = Storage(root=str(storage_root))
    assert storage.list_runs(project="worker-smoke") == []


def test_invalid_phase_and_probe_without_limit_are_rejected():
    base = dict(
        trial_id="trial-invalid",
        project="worker-smoke",
        entrypoint=TrainingEntrypoint("train:main"),
        config={},
    )
    with pytest.raises(TrialProtocolError, match="phase"):
        TrialRequest(phase="tune", **base)
    with pytest.raises(TrialProtocolError, match="require max_steps"):
        TrialRequest(phase="probe", **base)


def test_malformed_request_still_produces_contract_error_result(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    request_path = tmp_path / "bad-request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps({"schema": {"name": "wrong"}}), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "watcherml._trial_worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            "--project-root",
            str(project_root),
            "--storage-root",
            str(tmp_path / "watcher-data"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    result = load_result(result_path)
    assert completed.returncode == EXIT_CONTRACT_ERROR
    assert result.status == "contract_error"
    assert result.trial_id is None
    assert "schema.name" in result.error["message"]
