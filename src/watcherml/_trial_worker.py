"""Execute exactly one WatcherML trial inside a child Python process.

This internal module is launched by ``trial_runner.py``. It consumes one
versioned request, creates at most one normal WatcherML run, and atomically
writes one machine-readable result.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .entrypoint import (
    EntrypointError,
    EntrypointResultError,
    invoke_entrypoint,
    validate_entrypoint,
)
from .run import Run
from .storage import Storage
from .trial_protocol import (
    EXIT_CONTRACT_ERROR,
    EXIT_SUCCESS,
    EXIT_TRAINING_FAILED,
    EXIT_WORKER_ERROR,
    TrialProtocolError,
    TrialRequest,
    TrialResult,
    load_request,
    write_result,
)


def execute_trial(
    request: TrialRequest,
    *,
    project_root: str,
    storage_root: str,
) -> TrialResult:
    """Validate and execute one request in the current worker process."""
    started_at = time.time()
    run: Optional[Run] = None
    metrics: Dict[str, float] = {}

    project_directory = Path(project_root).resolve()
    storage_directory = Path(storage_root).resolve()

    # The parent runner also applies this patch before Python starts. Applying
    # it here makes direct worker invocation and unit tests follow the same
    # request semantics. Never print environment values: requests may contain
    # operational details even though credential-like keys are rejected.
    with _temporary_environment(request.environment_patch):
        try:
            # Importing the target module happens during validation, after the
            # runtime environment patch has been installed.
            validate_entrypoint(
                request.entrypoint,
                project_root=str(project_directory),
                require_max_steps=request.phase == "probe",
            )
        except EntrypointError as exc:
            return _result(
                request,
                status="contract_error",
                exit_code=EXIT_CONTRACT_ERROR,
                started_at=started_at,
                error=exc,
            )

        # Failures creating storage or starting a Run are worker/infrastructure
        # failures. They intentionally escape to main(), which records
        # worker_error instead of blaming the user's training function.
        storage = Storage(root=str(storage_directory))
        run = Run(
            project=request.project,
            config=request.config,
            run_id=request.run_id,
            storage=storage,
        )

        with _temporary_current_directory(project_directory):
            run.start()
            try:
                with run:
                    metrics = invoke_entrypoint(
                        request.entrypoint,
                        request.config,
                        project_root=str(project_directory),
                        max_steps=request.max_steps,
                    )
                    run.log(metrics, step=request.max_steps)
            except EntrypointResultError as exc:
                return _result(
                    request,
                    status="contract_error",
                    exit_code=EXIT_CONTRACT_ERROR,
                    started_at=started_at,
                    run=run,
                    storage=storage,
                    error=exc,
                )
            except EntrypointError as exc:
                return _result(
                    request,
                    status="contract_error",
                    exit_code=EXIT_CONTRACT_ERROR,
                    started_at=started_at,
                    run=run,
                    storage=storage,
                    error=exc,
                )
            except Exception as exc:
                # Run.__exit__ persisted the failed run and deterministic
                # capsule before the exception reached this branch.
                return _result(
                    request,
                    status="training_failed",
                    exit_code=EXIT_TRAINING_FAILED,
                    started_at=started_at,
                    run=run,
                    storage=storage,
                    error=exc,
                )

    return _result(
        request,
        status="success",
        exit_code=EXIT_SUCCESS,
        started_at=started_at,
        run=run,
        metrics=metrics,
    )


def _result(
    request: TrialRequest,
    *,
    status: str,
    exit_code: int,
    started_at: float,
    run: Optional[Run] = None,
    storage: Optional[Storage] = None,
    metrics: Optional[Dict[str, float]] = None,
    error: Optional[Exception] = None,
) -> TrialResult:
    failure_class = None
    capsule_schema_version = None

    if run is not None and storage is not None:
        capsule = storage.get_failure_capsule(run.run_id)
        if capsule:
            failure_class = capsule.get("failure_class")
            capsule_schema_version = capsule.get("capsule_schema_version")

    ended_at = time.time()
    return TrialResult(
        trial_id=request.trial_id,
        campaign_id=request.campaign_id,
        source_run_id=request.source_run_id,
        project=request.project,
        phase=request.phase,
        status=status,
        worker_exit_code=exit_code,
        run_id=run.run_id if run is not None else None,
        metrics=metrics or {},
        failure_class=failure_class,
        capsule_schema_version=capsule_schema_version,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=ended_at - started_at,
        worker_pid=os.getpid(),
        error=(
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
    )


def _invalid_request_result(error: Exception, started_at: float) -> TrialResult:
    ended_at = time.time()
    return TrialResult(
        trial_id=None,
        campaign_id=None,
        source_run_id=None,
        project=None,
        phase=None,
        status="contract_error",
        worker_exit_code=EXIT_CONTRACT_ERROR,
        run_id=None,
        metrics={},
        failure_class=None,
        capsule_schema_version=None,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=ended_at - started_at,
        worker_pid=os.getpid(),
        error={"type": type(error).__name__, "message": str(error)},
    )


def _worker_error_result(
    request: TrialRequest,
    error: Exception,
    started_at: float,
) -> TrialResult:
    ended_at = time.time()
    return TrialResult(
        trial_id=request.trial_id,
        campaign_id=request.campaign_id,
        source_run_id=request.source_run_id,
        project=request.project,
        phase=request.phase,
        status="worker_error",
        worker_exit_code=EXIT_WORKER_ERROR,
        run_id=None,
        metrics={},
        failure_class=None,
        capsule_schema_version=None,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=ended_at - started_at,
        worker_pid=os.getpid(),
        error={"type": type(error).__name__, "message": str(error)},
    )


@contextmanager
def _temporary_current_directory(directory: Path) -> Iterator[None]:
    if not directory.exists() or not directory.is_dir():
        raise TrialProtocolError(
            "project root does not exist: {}".format(directory)
        )
    previous = Path.cwd()
    os.chdir(str(directory))
    try:
        yield
    finally:
        os.chdir(str(previous))


@contextmanager
def _temporary_environment(patch: Dict[str, str]) -> Iterator[None]:
    """Apply an already protocol-validated environment patch temporarily."""
    previous = {key: os.environ.get(key) for key in patch}
    try:
        for key, value in patch.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m watcherml._trial_worker",
        description="Internal WatcherML one-trial child worker",
    )
    parser.add_argument("--request", required=True, help="trial request JSON")
    parser.add_argument("--result", required=True, help="result JSON destination")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--storage-root", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.time()

    try:
        request = load_request(args.request)
    except (TrialProtocolError, EntrypointError) as exc:
        result = _invalid_request_result(exc, started_at)
        try:
            write_result(args.result, result)
        except Exception as write_exc:
            print(
                "WatcherML worker could not write result: {}".format(write_exc),
                file=sys.stderr,
            )
            return EXIT_WORKER_ERROR
        return result.worker_exit_code

    try:
        result = execute_trial(
            request,
            project_root=args.project_root,
            storage_root=args.storage_root,
        )
    except Exception as exc:
        # This branch is reserved for worker or infrastructure failures, not
        # exceptions raised normally by the validated training callable.
        result = _worker_error_result(request, exc, started_at)

    try:
        write_result(args.result, result)
    except Exception as exc:
        print(
            "WatcherML worker could not write result: {}".format(exc),
            file=sys.stderr,
        )
        return EXIT_WORKER_ERROR

    return result.worker_exit_code


if __name__ == "__main__":
    raise SystemExit(main())