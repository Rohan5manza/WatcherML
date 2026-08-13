"""Parent-side isolated trial runner for WatcherML.

The runner launches exactly one ``watcherml._trial_worker`` process, captures
its logs, enforces a hard deadline, terminates its process group when needed,
validates its versioned result, and writes an immutable execution manifest.

It deliberately contains no OOM diagnosis, intervention selection, candidate
ranking, or confirmation policy. Those layers consume this runner.
"""
from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .storage import Storage
from .trial_protocol import (
    TRIAL_STATUSES,
    TrialProtocolError,
    TrialRequest,
    TrialResult,
    atomic_write_json,
    load_result,
    write_request,
)


TRIAL_EXECUTION_SCHEMA_NAME = "watcherml.trial-execution"
TRIAL_EXECUTION_SCHEMA_VERSION = "1.0"

PARENT_EXECUTION_STATUSES = frozenset(
    set(TRIAL_STATUSES) | {"timeout", "launch_error", "protocol_error"}
)

DEFAULT_TIMEOUT_SECONDS = 60.0 * 60.0
DEFAULT_TERMINATION_GRACE_SECONDS = 5.0


class TrialRunnerError(RuntimeError):
    """Raised when the parent cannot safely prepare or persist a trial."""


@dataclass(frozen=True)
class TrialExecution:
    """Parent-observed outcome of one worker-process attempt."""

    trial_id: str
    run_id: str
    project: str
    phase: str
    status: str
    child_exit_code: Optional[int]
    child_pid: Optional[int]
    timed_out: bool
    termination: Optional[str]
    started_at: float
    ended_at: float
    duration_seconds: float
    trial_directory: str
    request_path: str
    result_path: str
    stdout_path: str
    stderr_path: str
    execution_path: str
    result: Optional[TrialResult]
    error: Optional[dict]

    def __post_init__(self) -> None:
        if self.status not in PARENT_EXECUTION_STATUSES:
            raise TrialRunnerError(
                "invalid parent execution status: {!r}".format(self.status)
            )
        if self.ended_at < self.started_at or self.duration_seconds < 0:
            raise TrialRunnerError("execution timestamps are inconsistent")
        if self.status == "timeout" and not self.timed_out:
            raise TrialRunnerError("timeout status requires timed_out=True")
        if self.timed_out and self.status != "timeout":
            raise TrialRunnerError("timed_out=True requires timeout status")

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": TRIAL_EXECUTION_SCHEMA_NAME,
                "version": TRIAL_EXECUTION_SCHEMA_VERSION,
            },
            "trial_id": self.trial_id,
            "run_id": self.run_id,
            "project": self.project,
            "phase": self.phase,
            "status": self.status,
            "child_exit_code": self.child_exit_code,
            "child_pid": self.child_pid,
            "timed_out": self.timed_out,
            "termination": self.termination,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "trial_directory": self.trial_directory,
            "artifacts": {
                "request": self.request_path,
                "result": self.result_path,
                "stdout": self.stdout_path,
                "stderr": self.stderr_path,
                "execution": self.execution_path,
            },
            "worker_result": self.result.to_dict() if self.result else None,
            "error": self.error,
        }


def run_trial(
    request: TrialRequest,
    *,
    project_root: Union[str, Path],
    storage_root: Union[str, Path],
    trials_root: Optional[Union[str, Path]] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    python_executable: Optional[Union[str, Path]] = None,
) -> TrialExecution:
    """Launch and supervise one isolated WatcherML worker.

    A unique ``trial_id`` is required. Existing trial directories are never
    overwritten because they are part of the recovery audit trail.

    The timeout covers the child process lifetime. On POSIX systems the worker
    starts a new session, allowing WatcherML to terminate its full process
    group, including DataLoader workers and other descendants.
    """
    if not isinstance(request, TrialRequest):
        raise TrialRunnerError("request must be a TrialRequest")
    timeout = _positive_finite(timeout_seconds, "timeout_seconds")
    grace = _nonnegative_finite(
        termination_grace_seconds,
        "termination_grace_seconds",
    )

    project_directory = _existing_directory(project_root, "project_root")
    storage_directory = Path(storage_root).expanduser().resolve()
    storage_directory.mkdir(parents=True, exist_ok=True)
    if not storage_directory.is_dir():
        raise TrialRunnerError(
            "storage_root is not a directory: {}".format(storage_directory)
        )

    trial_parent = (
        Path(trials_root).expanduser().resolve()
        if trials_root is not None
        else storage_directory / "trials"
    )
    trial_parent.mkdir(parents=True, exist_ok=True)
    trial_directory = trial_parent / request.trial_id
    try:
        trial_directory.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise TrialRunnerError(
            "trial directory already exists for {!r}; trial_id values must be "
            "unique and WatcherML will not overwrite evidence".format(
                request.trial_id
            )
        ) from exc

    request_path = trial_directory / "request.json"
    result_path = trial_directory / "result.json"
    stdout_path = trial_directory / "stdout.log"
    stderr_path = trial_directory / "stderr.log"
    execution_path = trial_directory / "execution.json"

    try:
        write_request(request_path, request)
    except Exception as exc:
        raise TrialRunnerError(
            "could not persist trial request: {}".format(exc)
        ) from exc

    executable = _resolve_python_executable(python_executable)
    command = [
        executable,
        "-m",
        "watcherml._trial_worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        "--project-root",
        str(project_directory),
        "--storage-root",
        str(storage_directory),
    ]

    # Applying the patch here is essential: allocator/runtime variables must
    # exist before the child interpreter imports a framework such as PyTorch.
    child_environment = os.environ.copy()
    child_environment.update(request.environment_patch)
    child_environment.setdefault("PYTHONUNBUFFERED", "1")

    started_at = time.time()
    monotonic_started = time.monotonic()
    process = None

    popen_options = _process_group_options()
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open(
            "wb"
        ) as stderr_handle:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(project_directory),
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    **popen_options
                )
            except Exception as exc:
                ended_at = time.time()
                execution = _execution(
                    request,
                    status="launch_error",
                    process=None,
                    timed_out=False,
                    termination=None,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_seconds=time.monotonic() - monotonic_started,
                    trial_directory=trial_directory,
                    request_path=request_path,
                    result_path=result_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    execution_path=execution_path,
                    result=None,
                    error=exc,
                )
                _persist_execution(execution)
                return execution

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                termination = _terminate_process_group(process, grace)
                ended_at = time.time()
                duration = time.monotonic() - monotonic_started
                _mark_timed_out_run(
                    storage_directory,
                    request.run_id,
                    ended_at=ended_at,
                    duration_seconds=duration,
                )
                execution = _execution(
                    request,
                    status="timeout",
                    process=process,
                    timed_out=True,
                    termination=termination,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_seconds=duration,
                    trial_directory=trial_directory,
                    request_path=request_path,
                    result_path=result_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    execution_path=execution_path,
                    result=None,
                    error=TimeoutError(
                        "trial exceeded {:.3f} seconds".format(timeout)
                    ),
                )
                _persist_execution(execution)
                return execution
    finally:
        # Popen.wait normally reaps the process. This protects unusual parent
        # exceptions without leaving a worker running in the background.
        if process is not None and process.poll() is None:
            _terminate_process_group(process, grace)

    ended_at = time.time()
    duration = time.monotonic() - monotonic_started

    try:
        result = load_result(result_path)
    except (TrialProtocolError, OSError, ValueError) as exc:
        execution = _execution(
            request,
            status="protocol_error",
            process=process,
            timed_out=False,
            termination=None,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            trial_directory=trial_directory,
            request_path=request_path,
            result_path=result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            execution_path=execution_path,
            result=None,
            error=exc,
        )
        _persist_execution(execution)
        return execution

    protocol_errors = _validate_result_against_request(
        request,
        result,
        process.returncode,
    )
    if protocol_errors:
        execution = _execution(
            request,
            status="protocol_error",
            process=process,
            timed_out=False,
            termination=None,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            trial_directory=trial_directory,
            request_path=request_path,
            result_path=result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            execution_path=execution_path,
            result=result,
            error=TrialProtocolError("; ".join(protocol_errors)),
        )
        _persist_execution(execution)
        return execution

    execution = _execution(
        request,
        status=result.status,
        process=process,
        timed_out=False,
        termination=None,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration,
        trial_directory=trial_directory,
        request_path=request_path,
        result_path=result_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        execution_path=execution_path,
        result=result,
        error=None,
    )
    _persist_execution(execution)
    return execution


def _execution(
    request: TrialRequest,
    *,
    status: str,
    process,
    timed_out: bool,
    termination: Optional[str],
    started_at: float,
    ended_at: float,
    duration_seconds: float,
    trial_directory: Path,
    request_path: Path,
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    execution_path: Path,
    result: Optional[TrialResult],
    error: Optional[Exception],
) -> TrialExecution:
    return TrialExecution(
        trial_id=request.trial_id,
        run_id=request.run_id,
        project=request.project,
        phase=request.phase,
        status=status,
        child_exit_code=process.returncode if process is not None else None,
        child_pid=process.pid if process is not None else None,
        timed_out=timed_out,
        termination=termination,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=max(0.0, float(duration_seconds)),
        trial_directory=str(trial_directory),
        request_path=str(request_path),
        result_path=str(result_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        execution_path=str(execution_path),
        result=result,
        error=(
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
    )


def _persist_execution(execution: TrialExecution) -> None:
    try:
        atomic_write_json(execution.execution_path, execution.to_dict())
    except Exception as exc:
        raise TrialRunnerError(
            "could not persist trial execution manifest: {}".format(exc)
        ) from exc


def _validate_result_against_request(
    request: TrialRequest,
    result: TrialResult,
    child_exit_code: Optional[int],
) -> List[str]:
    errors: List[str] = []
    if result.trial_id != request.trial_id:
        errors.append("result trial_id does not match request")
    if result.project != request.project:
        errors.append("result project does not match request")
    if result.phase != request.phase:
        errors.append("result phase does not match request")
    if result.campaign_id != request.campaign_id:
        errors.append("result campaign_id does not match request")
    if result.source_run_id != request.source_run_id:
        errors.append("result source_run_id does not match request")
    if result.run_id is not None and result.run_id != request.run_id:
        errors.append("result run_id does not match the parent-selected run_id")
    if child_exit_code != result.worker_exit_code:
        errors.append("child exit code does not match worker result")
    if result.status in {"success", "training_failed"} and result.run_id is None:
        errors.append("training outcome is missing its persisted run_id")
    return errors


def _process_group_options() -> dict:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI later
        creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creation_flag}
    return {}


def _terminate_process_group(process, grace_seconds: float) -> str:
    if os.name == "posix":
        process_group_id = process.pid
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            if process.poll() is None:
                process.wait()
            return "already_exited"

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()  # reap the group leader if it has already exited
            if not _posix_process_group_exists(process_group_id):
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        termination = "sigterm"
        process.poll()
        if _posix_process_group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            termination = "sigkill"

        if process.poll() is None:
            process.wait()
        return termination

    if process.poll() is not None:
        return "already_exited"

    # Windows fallback. A future Windows executor can use Job Objects for
    # stronger descendant cleanup; macOS/Linux are the v1 acceptance targets.
    process.terminate()  # pragma: no cover - Windows path
    try:  # pragma: no cover - Windows path
        process.wait(timeout=grace_seconds)
        return "terminate"
    except subprocess.TimeoutExpired:  # pragma: no cover - Windows path
        process.kill()
        process.wait()
        return "kill"


def _posix_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The group exists even if the current user cannot signal it.
        return True


def _mark_timed_out_run(
    storage_root: Path,
    run_id: str,
    *,
    ended_at: float,
    duration_seconds: float,
) -> None:
    """Mark a child-created running row as timed out, if it exists."""
    try:
        storage = Storage(root=str(storage_root))
        row = storage.get_run(run_id)
        if row is not None and row["exit_status"] == "running":
            storage.upsert_run(
                run_id,
                ended_at=ended_at,
                duration_seconds=max(0.0, duration_seconds),
                exit_status="timeout",
            )
    except Exception:
        # execution.json remains the parent source of truth. Failure to update
        # a secondary run row must not hide the timeout outcome.
        pass


def _existing_directory(value: Union[str, Path], field_name: str) -> Path:
    directory = Path(value).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        raise TrialRunnerError(
            "{} is not an existing directory: {}".format(field_name, directory)
        )
    return directory


def _resolve_python_executable(
    value: Optional[Union[str, Path]],
) -> str:
    candidate = str(value) if value is not None else sys.executable
    resolved = shutil.which(candidate)
    if resolved is None:
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            resolved = str(path.resolve())
    if resolved is None:
        raise TrialRunnerError(
            "python executable was not found: {}".format(candidate)
        )
    return resolved


def _positive_finite(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise TrialRunnerError("{} must be a positive finite number".format(field_name))
    return float(value)


def _nonnegative_finite(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise TrialRunnerError(
            "{} must be a non-negative finite number".format(field_name)
        )
    return float(value)