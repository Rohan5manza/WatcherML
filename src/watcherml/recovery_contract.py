"""Versioned recovery contracts for WatcherML CUDA OOM campaigns.

A recovery contract is declared before any candidate trial runs.  It defines
the exact training entrypoint and source configuration, compute budgets,
verification requirements, workload identity, metric-regression limits, and
the strongest intervention class the campaign may execute.

This module does not plan interventions, launch trials, consume budget, rank
results, or issue recovery verdicts.  Later campaign and verifier layers use
the immutable contract as their shared source of truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from numbers import Real
from typing import Optional, Tuple

from .entrypoint import EntrypointError, TrainingEntrypoint, validate_config
from .interventions import ResolvedIntervention


RECOVERY_CONTRACT_SCHEMA_NAME = "watcherml.recovery-contract"
RECOVERY_CONTRACT_SCHEMA_VERSION = "1.0"

HARD_MAX_TRIALS = 10
HARD_MAX_CONFIRMATION_RUNS = 3
HARD_MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
HARD_MAX_GPU_SECONDS = 30 * 24 * 60 * 60
MAX_METRIC_GUARDS = 32
MAX_TEXT_LENGTH = 1_024

METRIC_DIRECTIONS = frozenset({"maximize", "minimize"})

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_METRIC_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,127}$")


class RecoveryContractError(ValueError):
    """Raised when a contract is malformed or internally inconsistent."""


class ContractScopeError(RecoveryContractError):
    """Raised when an intervention exceeds a campaign's declared authority."""


@dataclass(frozen=True)
class MetricGuard:
    """Maximum allowed degradation from one declared baseline metric.

    For a metric being maximized, ``baseline_value - max_regression`` is the
    lowest acceptable value.  For a metric being minimized,
    ``baseline_value + max_regression`` is the highest acceptable value.
    ``target_value`` may declare a stricter absolute requirement.
    """

    name: str
    direction: str
    baseline_value: float
    max_regression: float
    target_value: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _METRIC_PATTERN.fullmatch(
            self.name
        ):
            raise RecoveryContractError(
                "metric name must be a safe non-empty identifier"
            )
        if self.direction not in METRIC_DIRECTIONS:
            raise RecoveryContractError(
                "metric direction must be 'maximize' or 'minimize'"
            )
        object.__setattr__(
            self,
            "baseline_value",
            _finite_number(self.baseline_value, "baseline_value"),
        )
        regression = _finite_number(self.max_regression, "max_regression")
        if regression < 0:
            raise RecoveryContractError("max_regression must be non-negative")
        object.__setattr__(self, "max_regression", regression)
        if self.target_value is not None:
            object.__setattr__(
                self,
                "target_value",
                _finite_number(self.target_value, "target_value"),
            )

    @property
    def regression_boundary(self) -> float:
        if self.direction == "maximize":
            return self.baseline_value - self.max_regression
        return self.baseline_value + self.max_regression

    @property
    def acceptance_boundary(self) -> float:
        boundary = self.regression_boundary
        if self.target_value is None:
            return boundary
        if self.direction == "maximize":
            return max(boundary, self.target_value)
        return min(boundary, self.target_value)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "baseline_value": self.baseline_value,
            "max_regression": self.max_regression,
            "target_value": self.target_value,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MetricGuard":
        if not isinstance(payload, dict):
            raise RecoveryContractError("metric guard must be an object")
        _reject_unknown_fields(
            payload,
            {
                "name",
                "direction",
                "baseline_value",
                "max_regression",
                "target_value",
            },
            "metric guard",
        )
        try:
            return cls(
                name=payload["name"],
                direction=payload["direction"],
                baseline_value=payload["baseline_value"],
                max_regression=payload["max_regression"],
                target_value=payload.get("target_value"),
            )
        except KeyError as exc:
            raise RecoveryContractError(
                "metric guard is missing a required field"
            ) from exc


@dataclass(frozen=True)
class WorkloadIdentity:
    """Known source-workload identities that confirmation must preserve.

    Every non-null value becomes a required equality check in the verifier.
    Unknown values remain explicitly null rather than being guessed.
    """

    dataset_fingerprint: Optional[str] = None
    environment_fingerprint: Optional[str] = None
    git_commit: Optional[str] = None
    model_identifier: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_fingerprint",
            "environment_fingerprint",
            "git_commit",
            "model_identifier",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _bounded_text(value, field_name),
                )

    @property
    def known_fields(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name in (
                "dataset_fingerprint",
                "environment_fingerprint",
                "git_commit",
                "model_identifier",
            )
            if getattr(self, name) is not None
        )

    def to_dict(self) -> dict:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "git_commit": self.git_commit,
            "model_identifier": self.model_identifier,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkloadIdentity":
        if not isinstance(payload, dict):
            raise RecoveryContractError("workload identity must be an object")
        allowed = {
            "dataset_fingerprint",
            "environment_fingerprint",
            "git_commit",
            "model_identifier",
        }
        _reject_unknown_fields(payload, allowed, "workload identity")
        return cls(
            dataset_fingerprint=payload.get("dataset_fingerprint"),
            environment_fingerprint=payload.get("environment_fingerprint"),
            git_commit=payload.get("git_commit"),
            model_identifier=payload.get("model_identifier"),
        )


@dataclass(frozen=True)
class RecoveryBudget:
    """Hard campaign limits; all confirmation runs consume this budget."""

    max_trials: int = 10
    max_probe_trials: int = 5
    max_full_trials: int = 2
    probe_steps: int = 30
    trial_timeout_seconds: float = 60 * 60
    campaign_timeout_seconds: float = 4 * 60 * 60
    max_gpu_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        max_trials = _bounded_positive_int(
            self.max_trials,
            "max_trials",
            HARD_MAX_TRIALS,
        )
        max_probe_trials = _bounded_positive_int(
            self.max_probe_trials,
            "max_probe_trials",
            HARD_MAX_TRIALS,
        )
        max_full_trials = _bounded_positive_int(
            self.max_full_trials,
            "max_full_trials",
            HARD_MAX_TRIALS,
        )
        probe_steps = _bounded_positive_int(
            self.probe_steps,
            "probe_steps",
            10_000_000,
        )
        if max_probe_trials > max_trials:
            raise RecoveryContractError(
                "max_probe_trials cannot exceed max_trials"
            )
        if max_full_trials > max_trials:
            raise RecoveryContractError(
                "max_full_trials cannot exceed max_trials"
            )
        object.__setattr__(self, "max_trials", max_trials)
        object.__setattr__(self, "max_probe_trials", max_probe_trials)
        object.__setattr__(self, "max_full_trials", max_full_trials)
        object.__setattr__(self, "probe_steps", probe_steps)

        trial_timeout = _positive_finite(
            self.trial_timeout_seconds,
            "trial_timeout_seconds",
            maximum=HARD_MAX_TIMEOUT_SECONDS,
        )
        campaign_timeout = _positive_finite(
            self.campaign_timeout_seconds,
            "campaign_timeout_seconds",
            maximum=HARD_MAX_TIMEOUT_SECONDS,
        )
        if trial_timeout > campaign_timeout:
            raise RecoveryContractError(
                "trial_timeout_seconds cannot exceed campaign_timeout_seconds"
            )
        object.__setattr__(self, "trial_timeout_seconds", trial_timeout)
        object.__setattr__(self, "campaign_timeout_seconds", campaign_timeout)

        if self.max_gpu_seconds is not None:
            object.__setattr__(
                self,
                "max_gpu_seconds",
                _positive_finite(
                    self.max_gpu_seconds,
                    "max_gpu_seconds",
                    maximum=HARD_MAX_GPU_SECONDS,
                ),
            )

    def to_dict(self) -> dict:
        return {
            "max_trials": self.max_trials,
            "max_probe_trials": self.max_probe_trials,
            "max_full_trials": self.max_full_trials,
            "probe_steps": self.probe_steps,
            "trial_timeout_seconds": self.trial_timeout_seconds,
            "campaign_timeout_seconds": self.campaign_timeout_seconds,
            "max_gpu_seconds": self.max_gpu_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RecoveryBudget":
        if not isinstance(payload, dict):
            raise RecoveryContractError("budget must be an object")
        allowed = {
            "max_trials",
            "max_probe_trials",
            "max_full_trials",
            "probe_steps",
            "trial_timeout_seconds",
            "campaign_timeout_seconds",
            "max_gpu_seconds",
        }
        _reject_unknown_fields(payload, allowed, "budget")
        try:
            return cls(
                max_trials=payload["max_trials"],
                max_probe_trials=payload["max_probe_trials"],
                max_full_trials=payload["max_full_trials"],
                probe_steps=payload["probe_steps"],
                trial_timeout_seconds=payload["trial_timeout_seconds"],
                campaign_timeout_seconds=payload["campaign_timeout_seconds"],
                max_gpu_seconds=payload.get("max_gpu_seconds"),
            )
        except KeyError as exc:
            raise RecoveryContractError("budget is missing a required field") from exc


@dataclass(frozen=True)
class VerificationRequirements:
    """Conditions a completed full/confirmation run must satisfy."""

    minimum_progress_steps: int
    metric_guards: Tuple[MetricGuard, ...]
    confirmation_runs: int = 2
    max_peak_vram_bytes: Optional[int] = None
    workload_identity: WorkloadIdentity = field(default_factory=WorkloadIdentity)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_progress_steps",
            _bounded_positive_int(
                self.minimum_progress_steps,
                "minimum_progress_steps",
                10_000_000_000,
            ),
        )
        try:
            guards = tuple(self.metric_guards)
        except TypeError as exc:
            raise RecoveryContractError(
                "metric_guards must be an iterable of MetricGuard values"
            ) from exc
        if not guards:
            raise RecoveryContractError(
                "at least one metric guard must be declared"
            )
        if len(guards) > MAX_METRIC_GUARDS:
            raise RecoveryContractError(
                "too many metric guards; maximum is {}".format(
                    MAX_METRIC_GUARDS
                )
            )
        if any(not isinstance(guard, MetricGuard) for guard in guards):
            raise RecoveryContractError(
                "metric_guards must contain MetricGuard values"
            )
        names = [guard.name for guard in guards]
        if len(names) != len(set(names)):
            raise RecoveryContractError("metric guard names must be unique")
        object.__setattr__(self, "metric_guards", guards)
        object.__setattr__(
            self,
            "confirmation_runs",
            _bounded_positive_int(
                self.confirmation_runs,
                "confirmation_runs",
                HARD_MAX_CONFIRMATION_RUNS,
            ),
        )
        if self.max_peak_vram_bytes is not None:
            object.__setattr__(
                self,
                "max_peak_vram_bytes",
                _bounded_positive_int(
                    self.max_peak_vram_bytes,
                    "max_peak_vram_bytes",
                    2**63 - 1,
                ),
            )
        if not isinstance(self.workload_identity, WorkloadIdentity):
            raise RecoveryContractError(
                "workload_identity must be a WorkloadIdentity"
            )

    def to_dict(self) -> dict:
        return {
            "minimum_progress_steps": self.minimum_progress_steps,
            "metric_guards": [guard.to_dict() for guard in self.metric_guards],
            "confirmation_runs": self.confirmation_runs,
            "max_peak_vram_bytes": self.max_peak_vram_bytes,
            "workload_identity": self.workload_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "VerificationRequirements":
        if not isinstance(payload, dict):
            raise RecoveryContractError("verification must be an object")
        allowed = {
            "minimum_progress_steps",
            "metric_guards",
            "confirmation_runs",
            "max_peak_vram_bytes",
            "workload_identity",
        }
        _reject_unknown_fields(payload, allowed, "verification")
        try:
            guards = payload["metric_guards"]
            if not isinstance(guards, list):
                raise RecoveryContractError("metric_guards must be an array")
            return cls(
                minimum_progress_steps=payload["minimum_progress_steps"],
                metric_guards=tuple(
                    MetricGuard.from_dict(item) for item in guards
                ),
                confirmation_runs=payload["confirmation_runs"],
                max_peak_vram_bytes=payload.get("max_peak_vram_bytes"),
                workload_identity=WorkloadIdentity.from_dict(
                    payload["workload_identity"]
                ),
            )
        except KeyError as exc:
            raise RecoveryContractError(
                "verification is missing a required field"
            ) from exc


@dataclass(frozen=True)
class InterventionPermissions:
    """Campaign-wide ceilings in addition to proposal-specific approval."""

    allow_approval_required: bool = False
    allow_semantic_changes: bool = False
    allow_high_risk: bool = False

    def __post_init__(self) -> None:
        for name in (
            "allow_approval_required",
            "allow_semantic_changes",
            "allow_high_risk",
        ):
            if not isinstance(getattr(self, name), bool):
                raise RecoveryContractError("{} must be a boolean".format(name))
        if self.allow_semantic_changes and not self.allow_approval_required:
            raise RecoveryContractError(
                "semantic changes require approval-required interventions"
            )
        if self.allow_high_risk and not self.allow_approval_required:
            raise RecoveryContractError(
                "high-risk changes require approval-required interventions"
            )

    def to_dict(self) -> dict:
        return {
            "allow_approval_required": self.allow_approval_required,
            "allow_semantic_changes": self.allow_semantic_changes,
            "allow_high_risk": self.allow_high_risk,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InterventionPermissions":
        if not isinstance(payload, dict):
            raise RecoveryContractError("permissions must be an object")
        allowed = {
            "allow_approval_required",
            "allow_semantic_changes",
            "allow_high_risk",
        }
        _reject_unknown_fields(payload, allowed, "permissions")
        try:
            return cls(
                allow_approval_required=payload["allow_approval_required"],
                allow_semantic_changes=payload["allow_semantic_changes"],
                allow_high_risk=payload["allow_high_risk"],
            )
        except KeyError as exc:
            raise RecoveryContractError(
                "permissions are missing a required field"
            ) from exc


@dataclass(frozen=True, init=False)
class RecoveryContract:
    """Complete immutable contract for one source OOM recovery campaign."""

    project: str
    source_run_id: str
    entrypoint: TrainingEntrypoint
    budget: RecoveryBudget
    verification: VerificationRequirements
    permissions: InterventionPermissions
    _source_config_json: str = field(repr=False)

    def __init__(
        self,
        project: str,
        source_run_id: str,
        entrypoint: TrainingEntrypoint,
        source_config: dict,
        budget: RecoveryBudget,
        verification: VerificationRequirements,
        permissions: Optional[InterventionPermissions] = None,
    ) -> None:
        if not isinstance(project, str) or not project.strip():
            raise RecoveryContractError("project must be a non-empty string")
        normalized_project = project.strip()
        if len(normalized_project) > 128:
            raise RecoveryContractError("project must be at most 128 characters")
        object.__setattr__(self, "project", normalized_project)
        if (
            not isinstance(source_run_id, str)
            or not _ID_PATTERN.fullmatch(source_run_id)
        ):
            raise RecoveryContractError("source_run_id is invalid")
        object.__setattr__(self, "source_run_id", source_run_id)
        if not isinstance(entrypoint, TrainingEntrypoint):
            raise RecoveryContractError(
                "entrypoint must be a TrainingEntrypoint"
            )
        object.__setattr__(self, "entrypoint", entrypoint)
        try:
            normalized_config = validate_config(source_config)
        except EntrypointError as exc:
            raise RecoveryContractError(str(exc)) from exc
        object.__setattr__(
            self,
            "_source_config_json",
            json.dumps(
                normalized_config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        if not isinstance(budget, RecoveryBudget):
            raise RecoveryContractError("budget must be a RecoveryBudget")
        object.__setattr__(self, "budget", budget)
        if not isinstance(verification, VerificationRequirements):
            raise RecoveryContractError(
                "verification must be VerificationRequirements"
            )
        object.__setattr__(self, "verification", verification)
        normalized_permissions = permissions or InterventionPermissions()
        if not isinstance(normalized_permissions, InterventionPermissions):
            raise RecoveryContractError(
                "permissions must be InterventionPermissions"
            )
        object.__setattr__(self, "permissions", normalized_permissions)
        if budget.probe_steps > verification.minimum_progress_steps:
            raise RecoveryContractError(
                "probe_steps cannot exceed minimum_progress_steps"
            )
        reserved_trials = (
            budget.max_probe_trials
            + budget.max_full_trials
            + verification.confirmation_runs
        )
        if reserved_trials > budget.max_trials:
            raise RecoveryContractError(
                "probe, full, and confirmation reservations exceed max_trials"
            )

    @property
    def source_config(self) -> dict:
        """Return a fresh copy so callers cannot mutate the sealed baseline."""
        return json.loads(self._source_config_json)

    @property
    def reserved_trials(self) -> int:
        return (
            self.budget.max_probe_trials
            + self.budget.max_full_trials
            + self.verification.confirmation_runs
        )

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": RECOVERY_CONTRACT_SCHEMA_NAME,
                "version": RECOVERY_CONTRACT_SCHEMA_VERSION,
            },
            "project": self.project,
            "source_run_id": self.source_run_id,
            "entrypoint": self.entrypoint.to_dict(),
            "source_config": self.source_config,
            "budget": self.budget.to_dict(),
            "verification": self.verification.to_dict(),
            "permissions": self.permissions.to_dict(),
            "invariants": {
                "failure_class": "cuda_out_of_memory",
                "fresh_process": True,
                "source_config_immutable": True,
                "no_oom_required": True,
                "confirmation_required": True,
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "RecoveryContract":
        if not isinstance(payload, dict):
            raise RecoveryContractError("recovery contract must be an object")
        schema = payload.get("schema") or {}
        if not isinstance(schema, dict):
            raise RecoveryContractError("schema must be an object")
        _reject_unknown_fields(schema, {"name", "version"}, "schema")
        if schema.get("name") != RECOVERY_CONTRACT_SCHEMA_NAME:
            raise RecoveryContractError(
                "schema.name must be {!r}".format(
                    RECOVERY_CONTRACT_SCHEMA_NAME
                )
            )
        if schema.get("version") != RECOVERY_CONTRACT_SCHEMA_VERSION:
            raise RecoveryContractError(
                "schema.version must be {!r}".format(
                    RECOVERY_CONTRACT_SCHEMA_VERSION
                )
            )
        allowed = {
            "schema",
            "project",
            "source_run_id",
            "entrypoint",
            "source_config",
            "budget",
            "verification",
            "permissions",
            "invariants",
        }
        _reject_unknown_fields(payload, allowed, "recovery contract")
        _validate_invariants(payload.get("invariants"))
        try:
            return cls(
                project=payload["project"],
                source_run_id=payload["source_run_id"],
                entrypoint=TrainingEntrypoint.from_dict(payload["entrypoint"]),
                source_config=payload["source_config"],
                budget=RecoveryBudget.from_dict(payload["budget"]),
                verification=VerificationRequirements.from_dict(
                    payload["verification"]
                ),
                permissions=InterventionPermissions.from_dict(
                    payload["permissions"]
                ),
            )
        except KeyError as exc:
            raise RecoveryContractError(
                "recovery contract is missing a required field"
            ) from exc
        except EntrypointError as exc:
            raise RecoveryContractError(str(exc)) from exc

    @classmethod
    def from_json(cls, encoded: str) -> "RecoveryContract":
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RecoveryContractError("invalid recovery contract JSON") from exc
        return cls.from_dict(payload)


def contract_digest(contract: RecoveryContract) -> str:
    """Return the stable SHA-256 identity persisted with campaign records."""
    if not isinstance(contract, RecoveryContract):
        raise RecoveryContractError("contract must be a RecoveryContract")
    return hashlib.sha256(contract.to_json().encode("utf-8")).hexdigest()


def validate_intervention_scope(
    contract: RecoveryContract,
    intervention: ResolvedIntervention,
) -> None:
    """Reject an otherwise valid intervention that exceeds this contract.

    Proposal-specific approval remains mandatory in ``interventions.py``.
    These checks are an additional campaign-wide ceiling and never authorize a
    proposal by themselves.
    """
    if not isinstance(contract, RecoveryContract):
        raise ContractScopeError("contract must be a RecoveryContract")
    if not isinstance(intervention, ResolvedIntervention):
        raise ContractScopeError(
            "intervention must be a ResolvedIntervention"
        )
    permissions = contract.permissions
    if (
        intervention.approval_required
        and not permissions.allow_approval_required
    ):
        raise ContractScopeError(
            "contract does not allow approval-required interventions"
        )
    if intervention.semantic_change and not permissions.allow_semantic_changes:
        raise ContractScopeError(
            "contract does not allow semantic changes"
        )
    if intervention.maximum_risk == "high" and not permissions.allow_high_risk:
        raise ContractScopeError(
            "contract does not allow high-risk interventions"
        )


def _validate_invariants(payload) -> None:
    expected = {
        "failure_class": "cuda_out_of_memory",
        "fresh_process": True,
        "source_config_immutable": True,
        "no_oom_required": True,
        "confirmation_required": True,
    }
    if payload != expected:
        raise RecoveryContractError(
            "recovery contract safety invariants cannot be changed"
        )


def _bounded_positive_int(value, field_name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise RecoveryContractError(
            "{} must be an integer from 1 to {}".format(field_name, maximum)
        )
    return value


def _finite_number(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RecoveryContractError("{} must be a finite number".format(field_name))
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RecoveryContractError("{} must be a finite number".format(field_name))
    return normalized


def _positive_finite(
    value,
    field_name: str,
    *,
    maximum: float,
) -> float:
    normalized = _finite_number(value, field_name)
    if normalized <= 0 or normalized > maximum:
        raise RecoveryContractError(
            "{} must be greater than zero and at most {}".format(
                field_name,
                maximum,
            )
        )
    return normalized


def _bounded_text(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryContractError(
            "{} must be a non-empty string or null".format(field_name)
        )
    normalized = value.strip()
    if len(normalized) > MAX_TEXT_LENGTH:
        raise RecoveryContractError(
            "{} exceeds the {}-character limit".format(
                field_name,
                MAX_TEXT_LENGTH,
            )
        )
    return normalized


def _reject_unknown_fields(
    payload: dict,
    allowed: set,
    artifact_name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RecoveryContractError(
            "{} contains unknown fields: {}".format(artifact_name, unknown)
        )


__all__ = [
    "HARD_MAX_CONFIRMATION_RUNS",
    "HARD_MAX_GPU_SECONDS",
    "HARD_MAX_TIMEOUT_SECONDS",
    "HARD_MAX_TRIALS",
    "METRIC_DIRECTIONS",
    "RECOVERY_CONTRACT_SCHEMA_NAME",
    "RECOVERY_CONTRACT_SCHEMA_VERSION",
    "ContractScopeError",
    "InterventionPermissions",
    "MetricGuard",
    "RecoveryBudget",
    "RecoveryContract",
    "RecoveryContractError",
    "VerificationRequirements",
    "WorkloadIdentity",
    "contract_digest",
    "validate_intervention_scope",
]