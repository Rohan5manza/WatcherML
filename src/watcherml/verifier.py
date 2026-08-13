"""Deterministic recovery verification for WatcherML CUDA OOM campaigns.

The verifier is the only layer allowed to turn successful confirmation runs
into a ``verified`` recovery verdict.  It evaluates immutable contract rules
against persisted confirmation evidence.  It does not run training, select an
intervention, rank candidates, infer missing values, or consult an LLM.

Verification is deliberately stricter than "the process exited successfully":

* every required confirmation is a distinct isolated execution;
* every execution is bound to the same contract, campaign, and candidate;
* the workload reaches the predeclared minimum progress;
* every metric remains inside its predeclared regression boundary;
* optional peak-VRAM and workload-identity constraints match; and
* no confirmation contains an OOM or another recorded failure class.

Missing required observations produce ``insufficient_evidence``.  Observed
contract violations produce ``rejected``.  Only a complete set of passing
checks produces ``verified``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Dict, Iterable, Optional, Tuple

from .entrypoint import EntrypointError, validate_config
from .recovery_contract import (
    MetricGuard,
    RecoveryContract,
    WorkloadIdentity,
    contract_digest,
)


CONFIRMATION_EVIDENCE_SCHEMA_NAME = "watcherml.confirmation-evidence"
CONFIRMATION_EVIDENCE_SCHEMA_VERSION = "1.0"
VERIFICATION_REPORT_SCHEMA_NAME = "watcherml.recovery-verification"
VERIFICATION_REPORT_SCHEMA_VERSION = "1.0"

VERIFICATION_VERDICTS = frozenset(
    {"verified", "rejected", "insufficient_evidence"}
)
CHECK_OUTCOMES = frozenset({"pass", "fail", "missing"})
TRIAL_PHASES = frozenset({"probe", "full", "confirmation"})
TRIAL_STATUSES = frozenset(
    {
        "success",
        "training_failed",
        "contract_error",
        "worker_error",
        "timeout",
        "launch_error",
        "protocol_error",
    }
)

MAX_CONFIRMATION_EVIDENCE = 32
MAX_ERROR_TEXT_LENGTH = 4_000

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHECK_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class VerificationError(ValueError):
    """Raised when verification input or a serialized report is malformed."""


@dataclass(frozen=True)
class ConfirmationEvidence:
    """Normalized evidence for one independently executed confirmation run.

    The campaign layer builds this object from the immutable trial request,
    parent execution manifest, worker result, and recorded run evidence.  A
    missing optional observation remains ``None`` so the verifier can report
    ``insufficient_evidence`` instead of guessing.
    """

    campaign_id: str
    candidate_id: str
    trial_id: str
    run_id: str
    project: str
    source_run_id: str
    contract_digest: str
    candidate_config_digest: str
    trial_request_digest: str
    execution_manifest_digest: str
    phase: str
    status: str
    metrics: Dict[str, float]
    progress_steps: Optional[int]
    peak_vram_bytes: Optional[int]
    workload_identity: WorkloadIdentity
    worker_pid: Optional[int]
    failure_class: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_id",
            "candidate_id",
            "trial_id",
            "run_id",
            "source_run_id",
        ):
            _validate_id(getattr(self, field_name), field_name)

        if not isinstance(self.project, str) or not self.project.strip():
            raise VerificationError("project must be a non-empty string")
        normalized_project = self.project.strip()
        if len(normalized_project) > 128:
            raise VerificationError("project must be at most 128 characters")
        object.__setattr__(self, "project", normalized_project)

        for field_name in (
            "contract_digest",
            "candidate_config_digest",
            "trial_request_digest",
            "execution_manifest_digest",
        ):
            _validate_digest(getattr(self, field_name), field_name)

        if self.phase not in TRIAL_PHASES:
            raise VerificationError(
                "phase must be one of {}".format(sorted(TRIAL_PHASES))
            )
        if self.status not in TRIAL_STATUSES:
            raise VerificationError(
                "status must be one of {}".format(sorted(TRIAL_STATUSES))
            )

        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(_normalize_metrics(self.metrics)),
        )
        if self.progress_steps is not None:
            object.__setattr__(
                self,
                "progress_steps",
                _nonnegative_int(self.progress_steps, "progress_steps"),
            )
        if self.peak_vram_bytes is not None:
            object.__setattr__(
                self,
                "peak_vram_bytes",
                _nonnegative_int(self.peak_vram_bytes, "peak_vram_bytes"),
            )
        if not isinstance(self.workload_identity, WorkloadIdentity):
            raise VerificationError(
                "workload_identity must be a WorkloadIdentity"
            )
        if self.worker_pid is not None:
            object.__setattr__(
                self,
                "worker_pid",
                _positive_int(self.worker_pid, "worker_pid"),
            )
        if self.failure_class is not None:
            if (
                not isinstance(self.failure_class, str)
                or not self.failure_class.strip()
            ):
                raise VerificationError(
                    "failure_class must be a non-empty string or null"
                )
            normalized_failure = self.failure_class.strip()
            if len(normalized_failure) > MAX_ERROR_TEXT_LENGTH:
                raise VerificationError("failure_class is too long")
            object.__setattr__(self, "failure_class", normalized_failure)

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": CONFIRMATION_EVIDENCE_SCHEMA_NAME,
                "version": CONFIRMATION_EVIDENCE_SCHEMA_VERSION,
            },
            "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "run_id": self.run_id,
            "project": self.project,
            "source_run_id": self.source_run_id,
            "contract_digest": self.contract_digest,
            "candidate_config_digest": self.candidate_config_digest,
            "trial_request_digest": self.trial_request_digest,
            "execution_manifest_digest": self.execution_manifest_digest,
            "phase": self.phase,
            "status": self.status,
            "metrics": dict(self.metrics),
            "progress_steps": self.progress_steps,
            "peak_vram_bytes": self.peak_vram_bytes,
            "workload_identity": self.workload_identity.to_dict(),
            "worker_pid": self.worker_pid,
            "failure_class": self.failure_class,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "ConfirmationEvidence":
        _validate_schema(
            payload,
            name=CONFIRMATION_EVIDENCE_SCHEMA_NAME,
            version=CONFIRMATION_EVIDENCE_SCHEMA_VERSION,
            artifact="confirmation evidence",
        )
        allowed = {
            "schema",
            "campaign_id",
            "candidate_id",
            "trial_id",
            "run_id",
            "project",
            "source_run_id",
            "contract_digest",
            "candidate_config_digest",
            "trial_request_digest",
            "execution_manifest_digest",
            "phase",
            "status",
            "metrics",
            "progress_steps",
            "peak_vram_bytes",
            "workload_identity",
            "worker_pid",
            "failure_class",
        }
        _reject_unknown_fields(payload, allowed, "confirmation evidence")
        try:
            return cls(
                campaign_id=payload["campaign_id"],
                candidate_id=payload["candidate_id"],
                trial_id=payload["trial_id"],
                run_id=payload["run_id"],
                project=payload["project"],
                source_run_id=payload["source_run_id"],
                contract_digest=payload["contract_digest"],
                candidate_config_digest=payload["candidate_config_digest"],
                trial_request_digest=payload["trial_request_digest"],
                execution_manifest_digest=payload[
                    "execution_manifest_digest"
                ],
                phase=payload["phase"],
                status=payload["status"],
                metrics=payload["metrics"],
                progress_steps=payload["progress_steps"],
                peak_vram_bytes=payload["peak_vram_bytes"],
                workload_identity=WorkloadIdentity.from_dict(
                    payload["workload_identity"]
                ),
                worker_pid=payload["worker_pid"],
                failure_class=payload["failure_class"],
            )
        except KeyError as exc:
            raise VerificationError(
                "confirmation evidence is missing a required field"
            ) from exc
        except ValueError as exc:
            if isinstance(exc, VerificationError):
                raise
            raise VerificationError(str(exc)) from exc

    @classmethod
    def from_json(cls, encoded: str) -> "ConfirmationEvidence":
        return cls.from_dict(
            _load_json_object(encoded, "confirmation evidence")
        )


@dataclass(frozen=True)
class VerificationCheck:
    """One machine-readable reason contributing to a recovery verdict."""

    code: str
    outcome: str
    message: str
    expected: object
    observed: object
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not _CHECK_CODE_PATTERN.fullmatch(self.code)
        ):
            raise VerificationError("verification check code is invalid")
        if self.outcome not in CHECK_OUTCOMES:
            raise VerificationError(
                "verification check outcome must be one of {}".format(
                    sorted(CHECK_OUTCOMES)
                )
            )
        object.__setattr__(
            self,
            "message",
            _bounded_text(self.message, "message"),
        )
        _validate_json_value(self.expected, "expected")
        _validate_json_value(self.observed, "observed")
        object.__setattr__(self, "expected", _freeze_json(self.expected))
        object.__setattr__(self, "observed", _freeze_json(self.observed))
        if self.run_id is not None:
            _validate_id(self.run_id, "run_id")

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "outcome": self.outcome,
            "message": self.message,
            "expected": _thaw_json(self.expected),
            "observed": _thaw_json(self.observed),
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "VerificationCheck":
        if not isinstance(payload, dict):
            raise VerificationError("verification check must be an object")
        _reject_unknown_fields(
            payload,
            {"code", "outcome", "message", "expected", "observed", "run_id"},
            "verification check",
        )
        try:
            return cls(
                code=payload["code"],
                outcome=payload["outcome"],
                message=payload["message"],
                expected=payload["expected"],
                observed=payload["observed"],
                run_id=payload["run_id"],
            )
        except KeyError as exc:
            raise VerificationError(
                "verification check is missing a required field"
            ) from exc


@dataclass(frozen=True)
class RecoveryVerification:
    """Immutable aggregate verdict for one candidate and one contract."""

    campaign_id: str
    candidate_id: str
    contract_digest: str
    candidate_config_digest: str
    verdict: str
    required_confirmation_runs: int
    observed_confirmation_runs: int
    confirmation_run_ids: Tuple[str, ...]
    checks: Tuple[VerificationCheck, ...]

    def __post_init__(self) -> None:
        _validate_id(self.campaign_id, "campaign_id")
        _validate_id(self.candidate_id, "candidate_id")
        _validate_digest(self.contract_digest, "contract_digest")
        _validate_digest(
            self.candidate_config_digest,
            "candidate_config_digest",
        )
        if self.verdict not in VERIFICATION_VERDICTS:
            raise VerificationError(
                "verdict must be one of {}".format(
                    sorted(VERIFICATION_VERDICTS)
                )
            )
        object.__setattr__(
            self,
            "required_confirmation_runs",
            _positive_int(
                self.required_confirmation_runs,
                "required_confirmation_runs",
            ),
        )
        object.__setattr__(
            self,
            "observed_confirmation_runs",
            _nonnegative_int(
                self.observed_confirmation_runs,
                "observed_confirmation_runs",
            ),
        )
        try:
            run_ids = tuple(self.confirmation_run_ids)
        except TypeError as exc:
            raise VerificationError(
                "confirmation_run_ids must be an iterable"
            ) from exc
        for run_id in run_ids:
            _validate_id(run_id, "confirmation run_id")
        object.__setattr__(self, "confirmation_run_ids", run_ids)

        try:
            checks = tuple(self.checks)
        except TypeError as exc:
            raise VerificationError("checks must be an iterable") from exc
        if not checks:
            raise VerificationError("a verification report requires checks")
        if any(not isinstance(check, VerificationCheck) for check in checks):
            raise VerificationError(
                "checks must contain VerificationCheck values"
            )
        object.__setattr__(self, "checks", checks)

        expected_verdict = _verdict_from_checks(checks)
        if self.verdict != expected_verdict:
            raise VerificationError(
                "verdict is inconsistent with verification check outcomes"
            )

        if self.observed_confirmation_runs != len(run_ids):
            raise VerificationError(
                "observed_confirmation_runs must equal confirmation_run_ids"
            )

        if self.verdict == "verified":
            if (
                self.observed_confirmation_runs
                < self.required_confirmation_runs
            ):
                raise VerificationError(
                    "a verified report requires every declared confirmation"
                )

            if len(set(run_ids)) != len(run_ids):
                raise VerificationError(
                    "a verified report requires unique confirmation run ids"
                )
    @property
    def verified(self) -> bool:
        return self.verdict == "verified"

    @property
    def failed_checks(self) -> Tuple[VerificationCheck, ...]:
        return tuple(check for check in self.checks if check.outcome == "fail")

    @property
    def missing_checks(self) -> Tuple[VerificationCheck, ...]:
        return tuple(
            check for check in self.checks if check.outcome == "missing"
        )

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": VERIFICATION_REPORT_SCHEMA_NAME,
                "version": VERIFICATION_REPORT_SCHEMA_VERSION,
            },
            "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id,
            "contract_digest": self.contract_digest,
            "candidate_config_digest": self.candidate_config_digest,
            "verdict": self.verdict,
            "verified": self.verified,
            "required_confirmation_runs": self.required_confirmation_runs,
            "observed_confirmation_runs": self.observed_confirmation_runs,
            "confirmation_run_ids": list(self.confirmation_run_ids),
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "RecoveryVerification":
        _validate_schema(
            payload,
            name=VERIFICATION_REPORT_SCHEMA_NAME,
            version=VERIFICATION_REPORT_SCHEMA_VERSION,
            artifact="recovery verification",
        )
        allowed = {
            "schema",
            "campaign_id",
            "candidate_id",
            "contract_digest",
            "candidate_config_digest",
            "verdict",
            "verified",
            "required_confirmation_runs",
            "observed_confirmation_runs",
            "confirmation_run_ids",
            "checks",
        }
        _reject_unknown_fields(payload, allowed, "recovery verification")
        try:
            run_ids = payload["confirmation_run_ids"]
            checks = payload["checks"]
            if not isinstance(run_ids, list):
                raise VerificationError(
                    "confirmation_run_ids must be an array"
                )
            if not isinstance(checks, list):
                raise VerificationError("checks must be an array")
            verified = payload["verified"]
            report = cls(
                campaign_id=payload["campaign_id"],
                candidate_id=payload["candidate_id"],
                contract_digest=payload["contract_digest"],
                candidate_config_digest=payload["candidate_config_digest"],
                verdict=payload["verdict"],
                required_confirmation_runs=payload[
                    "required_confirmation_runs"
                ],
                observed_confirmation_runs=payload[
                    "observed_confirmation_runs"
                ],
                confirmation_run_ids=tuple(run_ids),
                checks=tuple(
                    VerificationCheck.from_dict(check) for check in checks
                ),
            )
        except KeyError as exc:
            raise VerificationError(
                "recovery verification is missing a required field"
            ) from exc
        if not isinstance(verified, bool):
            raise VerificationError("verified must be a boolean")
        if verified != report.verified:
            raise VerificationError(
                "verified is inconsistent with the report verdict"
            )
        return report

    @classmethod
    def from_json(cls, encoded: str) -> "RecoveryVerification":
        return cls.from_dict(
            _load_json_object(encoded, "recovery verification")
        )


def configuration_digest(config: dict) -> str:
    """Return a stable SHA-256 identity for a JSON training configuration."""
    try:
        normalized = validate_config(config)
    except EntrypointError as exc:
        raise VerificationError(str(exc)) from exc
    return hashlib.sha256(_stable_json(normalized).encode("utf-8")).hexdigest()


def evidence_digest(evidence: ConfirmationEvidence) -> str:
    """Return the stable identity of one normalized confirmation artifact."""
    if not isinstance(evidence, ConfirmationEvidence):
        raise VerificationError(
            "evidence must be a ConfirmationEvidence"
        )
    return hashlib.sha256(evidence.to_json().encode("utf-8")).hexdigest()


def verification_digest(report: RecoveryVerification) -> str:
    """Return the stable identity of one deterministic verification report."""
    if not isinstance(report, RecoveryVerification):
        raise VerificationError("report must be a RecoveryVerification")
    return hashlib.sha256(report.to_json().encode("utf-8")).hexdigest()


def verify_recovery(
    contract: RecoveryContract,
    *,
    campaign_id: str,
    candidate_id: str,
    candidate_config: dict,
    confirmations: Iterable[ConfirmationEvidence],
) -> RecoveryVerification:
    """Evaluate confirmation evidence against a predeclared contract.

    All supplied confirmations are evaluated.  The function never selects a
    favorable subset and never uses a candidate's ranking score.  Repeating a
    run id, trial id, or execution manifest is a rejection, not another vote.
    """
    if not isinstance(contract, RecoveryContract):
        raise VerificationError("contract must be a RecoveryContract")
    _validate_id(campaign_id, "campaign_id")
    _validate_id(candidate_id, "candidate_id")
    candidate_digest = configuration_digest(candidate_config)
    expected_contract_digest = contract_digest(contract)

    try:
        evidence_items = tuple(confirmations)
    except TypeError as exc:
        raise VerificationError(
            "confirmations must be an iterable of ConfirmationEvidence"
        ) from exc
    if len(evidence_items) > MAX_CONFIRMATION_EVIDENCE:
        raise VerificationError(
            "confirmations exceed the {}-item verification limit".format(
                MAX_CONFIRMATION_EVIDENCE
            )
        )
    if any(
        not isinstance(item, ConfirmationEvidence)
        for item in evidence_items
    ):
        raise VerificationError(
            "confirmations must contain ConfirmationEvidence values"
        )

    checks = []
    required = contract.verification.confirmation_runs
    observed = len(evidence_items)
    checks.append(
        _check(
            "confirmation.count",
            "pass" if observed >= required else "missing",
            "At least the declared number of confirmation runs must exist.",
            {"minimum": required},
            observed,
        )
    )

    _append_uniqueness_check(
        checks,
        "confirmation.unique_trial_ids",
        [item.trial_id for item in evidence_items],
        "Every confirmation must have a distinct trial id.",
    )
    _append_uniqueness_check(
        checks,
        "confirmation.unique_run_ids",
        [item.run_id for item in evidence_items],
        "Every confirmation must have a distinct recorded run id.",
    )
    _append_uniqueness_check(
        checks,
        "confirmation.unique_requests",
        [item.trial_request_digest for item in evidence_items],
        "Every confirmation must reference a distinct trial request.",
    )
    _append_uniqueness_check(
        checks,
        "confirmation.unique_executions",
        [item.execution_manifest_digest for item in evidence_items],
        "Every confirmation must reference a distinct execution manifest.",
    )

    for item in evidence_items:
        _append_confirmation_checks(
            checks,
            contract=contract,
            evidence=item,
            expected_contract_digest=expected_contract_digest,
            expected_campaign_id=campaign_id,
            expected_candidate_id=candidate_id,
            expected_candidate_config_digest=candidate_digest,
        )

    checks_tuple = tuple(checks)
    return RecoveryVerification(
        campaign_id=campaign_id,
        candidate_id=candidate_id,
        contract_digest=expected_contract_digest,
        candidate_config_digest=candidate_digest,
        verdict=_verdict_from_checks(checks_tuple),
        required_confirmation_runs=required,
        observed_confirmation_runs=observed,
        confirmation_run_ids=tuple(item.run_id for item in evidence_items),
        checks=checks_tuple,
    )


def _append_confirmation_checks(
    checks: list,
    *,
    contract: RecoveryContract,
    evidence: ConfirmationEvidence,
    expected_contract_digest: str,
    expected_campaign_id: str,
    expected_candidate_id: str,
    expected_candidate_config_digest: str,
) -> None:
    run_id = evidence.run_id
    bindings = (
        (
            "binding.contract",
            expected_contract_digest,
            evidence.contract_digest,
            "Confirmation evidence must reference this exact contract.",
        ),
        (
            "binding.campaign",
            expected_campaign_id,
            evidence.campaign_id,
            "Confirmation evidence must belong to this campaign.",
        ),
        (
            "binding.candidate",
            expected_candidate_id,
            evidence.candidate_id,
            "Confirmation evidence must belong to this candidate.",
        ),
        (
            "binding.candidate_config",
            expected_candidate_config_digest,
            evidence.candidate_config_digest,
            "All confirmations must run the exact candidate configuration.",
        ),
        (
            "binding.project",
            contract.project,
            evidence.project,
            "Confirmation evidence must belong to the contract project.",
        ),
        (
            "binding.source_run",
            contract.source_run_id,
            evidence.source_run_id,
            "Confirmation evidence must reference the source OOM run.",
        ),
    )
    for code, expected, observed, message in bindings:
        checks.append(
            _check(
                code,
                "pass" if observed == expected else "fail",
                message,
                expected,
                observed,
                run_id,
            )
        )

    checks.append(
        _check(
            "execution.phase",
            "pass" if evidence.phase == "confirmation" else "fail",
            "Only confirmation-phase executions can prove recovery.",
            "confirmation",
            evidence.phase,
            run_id,
        )
    )
    checks.append(
        _check(
            "execution.status",
            "pass" if evidence.status == "success" else "fail",
            "The confirmation process must finish successfully.",
            "success",
            evidence.status,
            run_id,
        )
    )
    checks.append(
        _check(
            "execution.failure_class",
            "pass" if evidence.failure_class is None else "fail",
            "A confirmation cannot contain an OOM or another failure class.",
            None,
            evidence.failure_class,
            run_id,
        )
    )
    checks.append(
        _check(
            "execution.worker_pid",
            "pass" if evidence.worker_pid is not None else "missing",
            "The isolated worker process identity must be recorded.",
            "positive integer",
            evidence.worker_pid,
            run_id,
        )
    )

    progress = evidence.progress_steps
    minimum_progress = contract.verification.minimum_progress_steps
    if progress is None:
        progress_outcome = "missing"
    elif progress >= minimum_progress:
        progress_outcome = "pass"
    else:
        progress_outcome = "fail"
    checks.append(
        _check(
            "execution.progress",
            progress_outcome,
            "The workload must reach the declared minimum progress.",
            {"minimum_steps": minimum_progress},
            progress,
            run_id,
        )
    )

    for guard in contract.verification.metric_guards:
        checks.append(_metric_check(guard, evidence.metrics, run_id))

    maximum_vram = contract.verification.max_peak_vram_bytes
    if maximum_vram is not None:
        observed_vram = evidence.peak_vram_bytes
        if observed_vram is None:
            vram_outcome = "missing"
        elif observed_vram <= maximum_vram:
            vram_outcome = "pass"
        else:
            vram_outcome = "fail"
        checks.append(
            _check(
                "resource.peak_vram_bytes",
                vram_outcome,
                "Peak VRAM must be observed and stay within the contract.",
                {"maximum": maximum_vram},
                observed_vram,
                run_id,
            )
        )

    expected_identity = contract.verification.workload_identity
    for field_name in expected_identity.known_fields:
        expected_value = getattr(expected_identity, field_name)
        observed_value = getattr(evidence.workload_identity, field_name)
        if observed_value is None:
            outcome = "missing"
        elif observed_value == expected_value:
            outcome = "pass"
        else:
            outcome = "fail"
        checks.append(
            _check(
                "identity.{}".format(field_name),
                outcome,
                "Confirmation workload identity must match the contract.",
                expected_value,
                observed_value,
                run_id,
            )
        )


def _metric_check(
    guard: MetricGuard,
    metrics: Dict[str, float],
    run_id: str,
) -> VerificationCheck:
    observed = metrics.get(guard.name)
    boundary = guard.acceptance_boundary
    if observed is None:
        outcome = "missing"
    elif guard.direction == "maximize":
        outcome = "pass" if observed >= boundary else "fail"
    else:
        outcome = "pass" if observed <= boundary else "fail"
    comparator = ">=" if guard.direction == "maximize" else "<="
    return _check(
        "metric.{}".format(_check_safe_name(guard.name)),
        outcome,
        "Metric must satisfy its predeclared regression boundary.",
        {
            "direction": guard.direction,
            "comparator": comparator,
            "boundary": boundary,
            "baseline": guard.baseline_value,
            "max_regression": guard.max_regression,
            "target": guard.target_value,
        },
        observed,
        run_id,
    )


def _append_uniqueness_check(
    checks: list,
    code: str,
    values: list,
    message: str,
) -> None:
    unique_count = len(set(values))
    checks.append(
        _check(
            code,
            "pass" if unique_count == len(values) else "fail",
            message,
            {"unique": True},
            {"count": len(values), "unique_count": unique_count},
        )
    )


def _check(
    code: str,
    outcome: str,
    message: str,
    expected,
    observed,
    run_id: Optional[str] = None,
) -> VerificationCheck:
    return VerificationCheck(
        code=code,
        outcome=outcome,
        message=message,
        expected=expected,
        observed=observed,
        run_id=run_id,
    )


def _verdict_from_checks(
    checks: Tuple[VerificationCheck, ...],
) -> str:
    if any(check.outcome == "fail" for check in checks):
        return "rejected"
    if any(check.outcome == "missing" for check in checks):
        return "insufficient_evidence"
    return "verified"


def _normalize_metrics(metrics: dict) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        raise VerificationError("metrics must be an object")
    normalized = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise VerificationError("metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise VerificationError(
                "metric {!r} must be a finite number".format(name)
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise VerificationError(
                "metric {!r} must be a finite number".format(name)
            )
        normalized[name] = numeric
    return normalized


def _validate_schema(
    payload: dict,
    *,
    name: str,
    version: str,
    artifact: str,
) -> None:
    if not isinstance(payload, dict):
        raise VerificationError("{} must be an object".format(artifact))
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        raise VerificationError("{} schema must be an object".format(artifact))
    _reject_unknown_fields(schema, {"name", "version"}, "{} schema".format(artifact))
    if schema.get("name") != name:
        raise VerificationError(
            "{} schema.name must be {!r}".format(artifact, name)
        )
    if schema.get("version") != version:
        raise VerificationError(
            "{} schema.version must be {!r}".format(artifact, version)
        )


def _validate_id(value, field_name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise VerificationError("{} is invalid".format(field_name))


def _validate_digest(value, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise VerificationError(
            "{} must be a lowercase SHA-256 digest".format(field_name)
        )


def _positive_int(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise VerificationError("{} must be a positive integer".format(field_name))
    return value


def _nonnegative_int(value, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VerificationError(
            "{} must be a non-negative integer".format(field_name)
        )
    return value


def _bounded_text(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerificationError("{} must be non-empty text".format(field_name))
    normalized = value.strip()
    if len(normalized) > MAX_ERROR_TEXT_LENGTH:
        raise VerificationError("{} is too long".format(field_name))
    return normalized


def _check_safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", name).lower()


def _validate_json_value(value, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            "{} must contain finite JSON data".format(field_name)
        ) from exc


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _stable_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _load_json_object(encoded: str, artifact: str) -> dict:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VerificationError("invalid {} JSON".format(artifact)) from exc
    if not isinstance(payload, dict):
        raise VerificationError("{} JSON must contain an object".format(artifact))
    return payload


def _reject_unknown_fields(
    payload: dict,
    allowed: set,
    artifact: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise VerificationError(
            "{} contains unknown fields: {}".format(artifact, unknown)
        )


__all__ = [
    "CHECK_OUTCOMES",
    "CONFIRMATION_EVIDENCE_SCHEMA_NAME",
    "CONFIRMATION_EVIDENCE_SCHEMA_VERSION",
    "MAX_CONFIRMATION_EVIDENCE",
    "VERIFICATION_REPORT_SCHEMA_NAME",
    "VERIFICATION_REPORT_SCHEMA_VERSION",
    "VERIFICATION_VERDICTS",
    "ConfirmationEvidence",
    "RecoveryVerification",
    "VerificationCheck",
    "VerificationError",
    "configuration_digest",
    "evidence_digest",
    "verification_digest",
    "verify_recovery",
]