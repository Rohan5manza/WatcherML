"""Versioned JSON protocol shared by WatcherML's trial parent and child.

The protocol transports an already-authorized trial. It does not decide which
interventions are safe; capability detection and the policy engine do that
before constructing a request.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from .entrypoint import EntrypointError, TrainingEntrypoint, validate_config


TRIAL_REQUEST_SCHEMA_NAME = "watcherml.trial-request"
TRIAL_REQUEST_SCHEMA_VERSION = "1.1"
TRIAL_RESULT_SCHEMA_NAME = "watcherml.trial-result"
TRIAL_RESULT_SCHEMA_VERSION = "1.0"

MAX_PROTOCOL_BYTES = 2_000_000
MAX_ENVIRONMENT_PATCH_BYTES = 65_536
MAX_ENVIRONMENT_PATCH_KEYS = 64

TRIAL_PHASES = frozenset({"probe", "full", "confirmation"})
TRIAL_STATUSES = frozenset(
    {"success", "training_failed", "contract_error", "worker_error"}
)

EXIT_SUCCESS = 0
EXIT_TRAINING_FAILED = 10
EXIT_CONTRACT_ERROR = 20
EXIT_WORKER_ERROR = 30

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Trial requests are persisted as evidence. Credentials must be inherited from
# the already-configured parent environment, never copied into request.json.
_SENSITIVE_ENVIRONMENT_PATTERN = re.compile(
    r"(?:^|_)(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY|"
    r"PRIVATE_KEY|CREDENTIALS?)(?:$|_)"
)

# These variables can redirect imports or inject native code before the worker
# starts. They are not legitimate recovery interventions.
_FORBIDDEN_ENVIRONMENT_KEYS = frozenset(
    {
        "PYTHONHOME",
        "PYTHONPATH",
        "PATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    }
)


class TrialProtocolError(ValueError):
    """Raised when a request/result violates the stable trial protocol."""


@dataclass(frozen=True)
class TrialRequest:
    """One complete, auditable instruction for one child process.

    ``run_id`` is chosen by the parent. This lets the parent mark a persisted
    run as timed out even if the child is killed before it can write result.json.
    ``environment_patch`` contains only non-secret runtime changes already
    approved by policy, such as an allocator setting.
    """

    trial_id: str
    project: str
    phase: str
    entrypoint: TrainingEntrypoint
    config: dict
    run_id: Optional[str] = None
    max_steps: Optional[int] = None
    campaign_id: Optional[str] = None
    source_run_id: Optional[str] = None
    environment_patch: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.trial_id, "trial_id")
        # Backward-friendly construction for callers that only supplied a
        # trial_id before protocol 1.1. New campaign code should still choose
        # and pass run_id explicitly.
        normalized_run_id = self.run_id or self.trial_id
        _validate_id(normalized_run_id, "run_id")
        object.__setattr__(self, "run_id", normalized_run_id)

        if not isinstance(self.project, str) or not self.project.strip():
            raise TrialProtocolError("project must be a non-empty string")
        normalized_project = self.project.strip()
        if len(normalized_project) > 128:
            raise TrialProtocolError("project must be at most 128 characters")
        object.__setattr__(self, "project", normalized_project)

        if self.phase not in TRIAL_PHASES:
            raise TrialProtocolError(
                "phase must be one of {}, got {!r}".format(
                    sorted(TRIAL_PHASES), self.phase
                )
            )
        if not isinstance(self.entrypoint, TrainingEntrypoint):
            raise TrialProtocolError("entrypoint must be a TrainingEntrypoint")

        try:
            normalized_config = validate_config(self.config)
        except EntrypointError as exc:
            raise TrialProtocolError(str(exc)) from exc
        object.__setattr__(self, "config", normalized_config)

        if self.max_steps is not None and (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise TrialProtocolError("max_steps must be a positive integer or null")
        if self.phase == "probe" and self.max_steps is None:
            raise TrialProtocolError("probe requests require max_steps")
        if self.phase != "probe" and self.max_steps is not None:
            raise TrialProtocolError("only probe requests may set max_steps")

        if self.campaign_id is not None:
            _validate_id(self.campaign_id, "campaign_id")
        if self.source_run_id is not None:
            _validate_id(self.source_run_id, "source_run_id")

        object.__setattr__(
            self,
            "environment_patch",
            validate_environment_patch(self.environment_patch),
        )

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": TRIAL_REQUEST_SCHEMA_NAME,
                "version": TRIAL_REQUEST_SCHEMA_VERSION,
            },
            "trial_id": self.trial_id,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "source_run_id": self.source_run_id,
            "project": self.project,
            "phase": self.phase,
            "entrypoint": self.entrypoint.to_dict(),
            "config": self.config,
            "max_steps": self.max_steps,
            "environment_patch": self.environment_patch,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrialRequest":
        _validate_schema(
            payload,
            name=TRIAL_REQUEST_SCHEMA_NAME,
            version=TRIAL_REQUEST_SCHEMA_VERSION,
        )
        try:
            entrypoint = TrainingEntrypoint.from_dict(payload.get("entrypoint"))
            return cls(
                trial_id=payload.get("trial_id"),
                run_id=payload.get("run_id"),
                campaign_id=payload.get("campaign_id"),
                source_run_id=payload.get("source_run_id"),
                project=payload.get("project"),
                phase=payload.get("phase"),
                entrypoint=entrypoint,
                config=payload.get("config"),
                max_steps=payload.get("max_steps"),
                environment_patch=payload.get("environment_patch") or {},
            )
        except EntrypointError as exc:
            raise TrialProtocolError(str(exc)) from exc


@dataclass(frozen=True)
class TrialResult:
    """Result written by a worker that reached normal result handling.

    Parent-only outcomes such as timeout, launch failure, or malformed/missing
    result are represented by ``TrialExecution`` in ``trial_runner.py``.
    """

    trial_id: Optional[str]
    project: Optional[str]
    phase: Optional[str]
    status: str
    worker_exit_code: int
    run_id: Optional[str]
    metrics: Dict[str, float]
    failure_class: Optional[str]
    capsule_schema_version: Optional[str]
    started_at: float
    ended_at: float
    duration_seconds: float
    worker_pid: int
    error: Optional[dict]
    campaign_id: Optional[str] = None
    source_run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in TRIAL_STATUSES:
            raise TrialProtocolError(
                "invalid trial result status: {!r}".format(self.status)
            )
        expected_code = {
            "success": EXIT_SUCCESS,
            "training_failed": EXIT_TRAINING_FAILED,
            "contract_error": EXIT_CONTRACT_ERROR,
            "worker_error": EXIT_WORKER_ERROR,
        }[self.status]
        if self.worker_exit_code != expected_code:
            raise TrialProtocolError(
                "status {!r} requires exit code {}".format(
                    self.status, expected_code
                )
            )

        if self.trial_id is not None:
            _validate_id(self.trial_id, "trial_id")
        if self.run_id is not None:
            _validate_id(self.run_id, "run_id")
        if self.phase is not None and self.phase not in TRIAL_PHASES:
            raise TrialProtocolError("result contains an invalid phase")
        if not isinstance(self.worker_pid, int) or self.worker_pid < 1:
            raise TrialProtocolError("worker_pid must be a positive integer")
        if not _is_finite_number(self.started_at):
            raise TrialProtocolError("started_at must be finite")
        if not _is_finite_number(self.ended_at):
            raise TrialProtocolError("ended_at must be finite")
        if not _is_finite_number(self.duration_seconds):
            raise TrialProtocolError("duration_seconds must be finite")
        if self.ended_at < self.started_at or self.duration_seconds < 0:
            raise TrialProtocolError("trial result timestamps are inconsistent")
        if not isinstance(self.metrics, dict):
            raise TrialProtocolError("metrics must be an object")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise TrialProtocolError("metric names must be non-empty strings")
            if not _is_finite_number(value):
                raise TrialProtocolError(
                    "metric {!r} must be a finite number".format(name)
                )
        if self.error is not None and not isinstance(self.error, dict):
            raise TrialProtocolError("error must be an object or null")

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": TRIAL_RESULT_SCHEMA_NAME,
                "version": TRIAL_RESULT_SCHEMA_VERSION,
            },
            "trial_id": self.trial_id,
            "campaign_id": self.campaign_id,
            "source_run_id": self.source_run_id,
            "project": self.project,
            "phase": self.phase,
            "status": self.status,
            "worker_exit_code": self.worker_exit_code,
            "run_id": self.run_id,
            "metrics": self.metrics,
            "failure_class": self.failure_class,
            "capsule_schema_version": self.capsule_schema_version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "worker_pid": self.worker_pid,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrialResult":
        _validate_schema(
            payload,
            name=TRIAL_RESULT_SCHEMA_NAME,
            version=TRIAL_RESULT_SCHEMA_VERSION,
        )
        return cls(
            trial_id=payload.get("trial_id"),
            campaign_id=payload.get("campaign_id"),
            source_run_id=payload.get("source_run_id"),
            project=payload.get("project"),
            phase=payload.get("phase"),
            status=payload.get("status"),
            worker_exit_code=payload.get("worker_exit_code"),
            run_id=payload.get("run_id"),
            metrics=payload.get("metrics") or {},
            failure_class=payload.get("failure_class"),
            capsule_schema_version=payload.get("capsule_schema_version"),
            started_at=payload.get("started_at"),
            ended_at=payload.get("ended_at"),
            duration_seconds=payload.get("duration_seconds"),
            worker_pid=payload.get("worker_pid"),
            error=payload.get("error"),
        )


def validate_environment_patch(patch: Optional[dict]) -> Dict[str, str]:
    """Validate non-secret environment changes that will be persisted.

    This is a serialization and injection-safety boundary, not authorization.
    The policy engine must separately decide whether each key/value is allowed.
    """
    if patch is None:
        return {}
    if not isinstance(patch, dict):
        raise TrialProtocolError("environment_patch must be an object")
    if len(patch) > MAX_ENVIRONMENT_PATCH_KEYS:
        raise TrialProtocolError(
            "environment_patch exceeds the {}-key limit".format(
                MAX_ENVIRONMENT_PATCH_KEYS
            )
        )

    normalized: Dict[str, str] = {}
    for key, value in patch.items():
        if not isinstance(key, str) or not _ENVIRONMENT_KEY_PATTERN.fullmatch(key):
            raise TrialProtocolError(
                "environment variable names must contain letters, digits, and underscores"
            )
        upper_key = key.upper()
        if upper_key in _FORBIDDEN_ENVIRONMENT_KEYS:
            raise TrialProtocolError(
                "environment variable {!r} is forbidden in trial requests".format(key)
            )
        if _SENSITIVE_ENVIRONMENT_PATTERN.search(upper_key):
            raise TrialProtocolError(
                "credentials must be inherited, not persisted in environment_patch"
            )
        if not isinstance(value, str):
            raise TrialProtocolError("environment variable values must be strings")
        if "\x00" in value:
            raise TrialProtocolError("environment variable values cannot contain NUL")
        normalized[key] = value

    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(encoded) > MAX_ENVIRONMENT_PATCH_BYTES:
        raise TrialProtocolError(
            "environment_patch exceeds the {}-byte limit".format(
                MAX_ENVIRONMENT_PATCH_BYTES
            )
        )
    return normalized


def load_request(path: Union[str, Path]) -> TrialRequest:
    return TrialRequest.from_dict(_load_json_object(path))


def load_result(path: Union[str, Path]) -> TrialResult:
    return TrialResult.from_dict(_load_json_object(path))


def write_request(path: Union[str, Path], request: TrialRequest) -> None:
    atomic_write_json(path, request.to_dict())


def write_result(path: Union[str, Path], result: TrialResult) -> None:
    atomic_write_json(path, result.to_dict())


def atomic_write_json(path: Union[str, Path], payload: dict) -> None:
    """Durably replace one protocol file without exposing partial JSON."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_PROTOCOL_BYTES:
        raise TrialProtocolError(
            "protocol document exceeds the {}-byte v1 limit".format(
                MAX_PROTOCOL_BYTES
            )
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(destination))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_json_object(path: Union[str, Path]) -> dict:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise TrialProtocolError(
            "could not read protocol file {}: {}".format(source, exc)
        ) from exc
    if size > MAX_PROTOCOL_BYTES:
        raise TrialProtocolError(
            "protocol document exceeds the {}-byte v1 limit".format(
                MAX_PROTOCOL_BYTES
            )
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrialProtocolError(
            "invalid protocol JSON in {}: {}".format(source, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise TrialProtocolError("protocol document must be a JSON object")
    return payload


def _validate_schema(payload: dict, *, name: str, version: str) -> None:
    if not isinstance(payload, dict):
        raise TrialProtocolError("protocol payload must be an object")
    schema = payload.get("schema") or {}
    if schema.get("name") != name:
        raise TrialProtocolError("schema.name must be {!r}".format(name))
    if schema.get("version") != version:
        raise TrialProtocolError("schema.version must be {!r}".format(version))


def _validate_id(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise TrialProtocolError(
            "{} must match {!r}".format(field_name, _ID_PATTERN.pattern)
        )


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )