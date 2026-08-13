"""Acceptance tests for WatcherML's parent-side isolated trial runner."""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from watcherml.entrypoint import TrainingEntrypoint
from watcherml.storage import Storage
from watcherml.trial_protocol import TrialRequest
from watcherml.trial_runner import (
    TRIAL_EXECUTION_SCHEMA_VERSION,
    TrialRunnerError,
    run_trial,
)


def _write_module(project_root: Path, name: str, source: str) -> None:
    (project_root / "{}.py".format(name)).write_text(source, encoding="utf-8")


def _request(
    *,
    trial_id: str,
    target: str,
    config=None,
    max_steps: int = 3,
    environment_patch=None,
) -> TrialRequest:
    return TrialRequest(
        trial_id=trial_id,
        run_id="{}-run".format(trial_id),
        campaign_id="campaign-runner-tests",
        source_run_id="source-oom-run",
        project="runner-tests",
        phase="probe",
        entrypoint=TrainingEntrypoint(target),
        config=config or {},
        max_steps=max_steps,
        environment_patch=environment_patch or {},
    )


def test_success_uses_parent_run_id_captures_logs_and_writes_manifest(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_module(
        project_root,
        "train_success",
        """
import os
import sys

ENVIRONMENT_WAS_PRESENT_AT_IMPORT = (
    os.environ.get("WATCHERML_RUNNER_TEST") == "enabled"
)

def main(config, max_steps=None):
    print("watcherml-stdout-marker")
    print("watcherml-stderr-marker", file=sys.stderr)
    return {
        "loss": 0.125,
        "steps_completed": max_steps,
        "environment_applied": 1.0 if ENVIRONMENT_WAS_PRESENT_AT_IMPORT else 0.0,
    }
""",
    )
    monkeypatch.delenv("WATCHERML_RUNNER_TEST", raising=False)
    request = _request(
        trial_id="runner-success",
        target="train_success:main",
        config={"batch_size": 8},
        environment_patch={"WATCHERML_RUNNER_TEST": "enabled"},
    )

    execution = run_trial(
        request,
        project_root=project_root,
        storage_root=tmp_path / "watcher-data",
        timeout_seconds=20,
    )

    assert execution.status == "success"
    assert execution.succeeded is True
    assert execution.timed_out is False
    assert execution.result is not None
    assert execution.result.run_id == request.run_id
    assert execution.result.metrics["environment_applied"] == 1.0
    assert execution.result.metrics["steps_completed"] == 3.0
    assert "WATCHERML_RUNNER_TEST" not in os.environ

    assert "watcherml-stdout-marker" in Path(execution.stdout_path).read_text(
        encoding="utf-8"
    )
    assert "watcherml-stderr-marker" in Path(execution.stderr_path).read_text(
        encoding="utf-8"
    )

    manifest = json.loads(
        Path(execution.execution_path).read_text(encoding="utf-8")
    )
    assert manifest["schema"]["version"] == TRIAL_EXECUTION_SCHEMA_VERSION
    assert manifest["status"] == "success"
    assert manifest["run_id"] == request.run_id
    assert manifest["worker_result"]["status"] == "success"

    storage = Storage(root=str(tmp_path / "watcher-data"))
    try:
        row = storage.get_run(request.run_id)
        assert row is not None
        assert row["exit_status"] == "success"
        assert storage.final_metrics(request.run_id)["loss"] == 0.125
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            close()


def test_cuda_oom_is_a_training_failure_with_a_linked_capsule(tmp_path):
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
    request = _request(
        trial_id="runner-oom",
        target="train_oom:main",
        config={"batch_size": 32, "gradient_accumulation_steps": 1},
    )

    execution = run_trial(
        request,
        project_root=project_root,
        storage_root=tmp_path / "watcher-data",
        timeout_seconds=20,
    )

    assert execution.status == "training_failed"
    assert execution.succeeded is False
    assert execution.result is not None
    assert execution.result.run_id == request.run_id
    assert execution.result.failure_class == "cuda_out_of_memory"
    assert execution.result.capsule_schema_version == "1.0"

    storage = Storage(root=str(tmp_path / "watcher-data"))
    try:
        capsule = storage.get_failure_capsule(request.run_id)
        assert capsule is not None
        assert capsule["failure_class"] == "cuda_out_of_memory"
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            close()


def test_unbounded_probe_is_rejected_before_a_run_is_created(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_module(
        project_root,
        "train_unbounded",
        "def main(config):\n    return {'loss': 1.0}\n",
    )
    request = _request(
        trial_id="runner-contract-error",
        target="train_unbounded:main",
    )

    execution = run_trial(
        request,
        project_root=project_root,
        storage_root=tmp_path / "watcher-data",
        timeout_seconds=20,
    )

    assert execution.status == "contract_error"
    assert execution.result is not None
    assert execution.result.run_id is None
    assert "will not silently run full training" in execution.result.error["message"]
    storage = Storage(root=str(tmp_path / "watcher-data"))
    try:
        assert storage.get_run(request.run_id) is None
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            close()


def test_timeout_terminates_worker_and_marks_running_row(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_module(
        project_root,
        "train_slow",
        """
import time

def main(config, max_steps=None):
    time.sleep(30)
    return {"steps_completed": max_steps}
""",
    )
    request = _request(
        trial_id="runner-timeout",
        target="train_slow:main",
    )

    execution = run_trial(
        request,
        project_root=project_root,
        storage_root=tmp_path / "watcher-data",
        timeout_seconds=3,
        termination_grace_seconds=0.25,
    )

    assert execution.status == "timeout"
    assert execution.timed_out is True
    assert execution.succeeded is False
    assert execution.result is None
    assert execution.termination in {
        "sigterm",
        "sigkill",
        "terminate",
        "kill",
        "already_exited",
    }
    manifest = json.loads(
        Path(execution.execution_path).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "timeout"
    assert manifest["timed_out"] is True

    storage = Storage(root=str(tmp_path / "watcher-data"))
    try:
        row = storage.get_run(request.run_id)
        assert row is not None
        assert row["exit_status"] == "timeout"
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_timeout_signals_descendant_processes_in_the_worker_group(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    marker_path = tmp_path / "descendant-received-sigterm.txt"
    _write_module(
        project_root,
        "train_with_child",
        """
import subprocess
import sys
import time

def main(config, max_steps=None):
    marker = config["marker_path"]
    child_code = '''
import signal
import sys
import time

marker = sys.argv[1]

def handle_sigterm(signum, frame):
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("received")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
time.sleep(30)
'''
    subprocess.Popen([sys.executable, "-c", child_code, marker])
    time.sleep(30)
    return {"steps_completed": max_steps}
""",
    )
    request = _request(
        trial_id="runner-process-group",
        target="train_with_child:main",
        config={"marker_path": str(marker_path)},
    )

    execution = run_trial(
        request,
        project_root=project_root,
        storage_root=tmp_path / "watcher-data",
        timeout_seconds=3,
        termination_grace_seconds=1,
    )

    assert execution.status == "timeout"
    deadline = time.monotonic() + 2
    while not marker_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker_path.read_text(encoding="utf-8") == "received"


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX executable script")
def test_missing_worker_result_becomes_protocol_error(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    request = _request(
        trial_id="runner-missing-result",
        target="unused:main",
    )

    execution = run_trial(
        request,
        project_root=project_root,
        storage_root=tmp_path / "watcher-data",
        python_executable=fake_python,
        timeout_seconds=10,
    )

    assert execution.status == "protocol_error"
    assert execution.child_exit_code == 0
    assert execution.result is None
    assert "could not read protocol file" in execution.error["message"]


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX execute permissions")
def test_process_launch_failure_is_recorded(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    non_executable = tmp_path / "not-executable"
    non_executable.write_text("not an executable", encoding="utf-8")
    non_executable.chmod(0o600)
    request = _request(
        trial_id="runner-launch-error",
        target="unused:main",
    )

    execution = run_trial(
        request,
        project_root=project_root,
        storage_root=tmp_path / "watcher-data",
        python_executable=non_executable,
        timeout_seconds=10,
    )

    assert execution.status == "launch_error"
    assert execution.child_pid is None
    assert execution.result is None
    assert execution.error["type"] in {"PermissionError", "OSError"}
    manifest = json.loads(
        Path(execution.execution_path).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "launch_error"


def test_existing_trial_directory_is_never_overwritten(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_module(
        project_root,
        "train_once",
        "def main(config, max_steps=None):\n    return {'loss': 0.5}\n",
    )
    request = _request(
        trial_id="runner-immutable",
        target="train_once:main",
    )
    storage_root = tmp_path / "watcher-data"

    first = run_trial(
        request,
        project_root=project_root,
        storage_root=storage_root,
        timeout_seconds=20,
    )
    original_request = Path(first.request_path).read_bytes()
    original_manifest = Path(first.execution_path).read_bytes()

    with pytest.raises(TrialRunnerError, match="will not overwrite evidence"):
        run_trial(
            request,
            project_root=project_root,
            storage_root=storage_root,
            timeout_seconds=20,
        )

    assert Path(first.request_path).read_bytes() == original_request
    assert Path(first.execution_path).read_bytes() == original_manifest


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("nan"), float("inf"), True],
)
def test_invalid_timeout_is_rejected_before_launch(tmp_path, timeout):
    project_root = tmp_path / "project"
    project_root.mkdir()
    request = _request(
        trial_id="runner-invalid-timeout",
        target="unused:main",
    )

    with pytest.raises(TrialRunnerError, match="positive finite"):
        run_trial(
            request,
            project_root=project_root,
            storage_root=tmp_path / "watcher-data",
            timeout_seconds=timeout,
        )

    assert not (tmp_path / "watcher-data" / "trials").exists()