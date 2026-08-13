"""Deterministic candidate ordering for WatcherML OOM recovery campaigns.

Ranking happens after full candidate trials and before confirmation.  It is a
provisional scheduling decision, never a recovery verdict.  Candidates first
pass contract-derived eligibility gates.  Eligible candidates are then ordered
lexicographically by a versioned, explicit preference policy; no opaque
weighted score or LLM judgment is used.

The verifier remains the only component that may declare recovery.
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

from .recovery_contract import (
    MetricGuard,
    RecoveryContract,
    WorkloadIdentity,
    contract_digest,
)


RANKING_POLICY_SCHEMA_NAME = "watcherml.ranking-policy"
RANKING_POLICY_SCHEMA_VERSION = "1.0"
RANKING_CANDIDATE_SCHEMA_NAME = "watcherml.ranking-candidate"
RANKING_CANDIDATE_SCHEMA_VERSION = "1.0"
RANKING_REPORT_SCHEMA_NAME = "watcherml.candidate-ranking"
RANKING_REPORT_SCHEMA_VERSION = "1.0"

CANDIDATE_ELIGIBILITY = frozenset(
    {"eligible", "rejected", "insufficient_evidence"}
)
RANKING_REASON_OUTCOMES = frozenset({"pass", "fail", "missing"})
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
INTERVENTION_RISKS = frozenset({"low", "medium", "high"})

RANKING_FACTORS = frozenset(
    {
        "primary_metric",
        "peak_vram_bytes",
        "throughput",
        "intervention_risk",
        "semantic_change",
        "approval_required",
        "change_count",
    }
)
DEFAULT_PREFERENCE_ORDER = (
    "primary_metric",
    "peak_vram_bytes",
    "throughput",
    "intervention_risk",
    "semantic_change",
    "approval_required",
    "change_count",
)

MAX_RANKING_CANDIDATES = 64
MAX_REASON_TEXT_LENGTH = 4_000

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_METRIC_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,127}$")
_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_RISK_RANK = {"low": 0, "medium": 1, "high": 2}


class RankingError(ValueError):
    """Raised when ranking input or a serialized ranking artifact is invalid."""


@dataclass(frozen=True)
class RankingPolicy:
    """Explicit lexicographic preferences applied after contract gates.

    ``primary_metric`` must name one of the recovery contract's metric guards,
    so its optimization direction and acceptable boundary were declared before
    trials.  ``throughput_metric`` is optional and is only used when the
    ``throughput`` factor appears in ``preference_order``.
    """

    primary_metric: str
    throughput_metric: Optional[str] = None
    preference_order: Tuple[str, ...] = DEFAULT_PREFERENCE_ORDER

    def __post_init__(self) -> None:
        _validate_metric_name(self.primary_metric, "primary_metric")
        if self.throughput_metric is not None:
            _validate_metric_name(self.throughput_metric, "throughput_metric")
            if self.throughput_metric == self.primary_metric:
                raise RankingError(
                    "throughput_metric must differ from primary_metric"
                )
        try:
            order = tuple(self.preference_order)
        except TypeError as exc:
            raise RankingError("preference_order must be an iterable") from exc
        if not order:
            raise RankingError("preference_order cannot be empty")
        if len(order) != len(set(order)):
            raise RankingError("preference_order factors must be unique")
        unknown = sorted(set(order) - RANKING_FACTORS)
        if unknown:
            raise RankingError(
                "preference_order contains unknown factors: {}".format(unknown)
            )
        if "primary_metric" not in order:
            raise RankingError("preference_order must include primary_metric")
        if "throughput" in order and self.throughput_metric is None:
            # The default policy remains convenient without requiring a
            # throughput metric: omit the inactive default factor.  Explicit
            # custom orders containing it remain invalid.
            if order == DEFAULT_PREFERENCE_ORDER:
                order = tuple(item for item in order if item != "throughput")
            else:
                raise RankingError(
                    "throughput factor requires throughput_metric"
                )
        if self.throughput_metric is not None and "throughput" not in order:
            raise RankingError(
                "throughput_metric requires the throughput preference factor"
            )
        object.__setattr__(self, "preference_order", order)

    def validate_against(self, contract: RecoveryContract) -> MetricGuard:
        if not isinstance(contract, RecoveryContract):
            raise RankingError("contract must be a RecoveryContract")
        guards = {
            guard.name: guard for guard in contract.verification.metric_guards
        }
        guard = guards.get(self.primary_metric)
        if guard is None:
            raise RankingError(
                "primary_metric must name a metric guard in the contract"
            )
        return guard

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": RANKING_POLICY_SCHEMA_NAME,
                "version": RANKING_POLICY_SCHEMA_VERSION,
            },
            "primary_metric": self.primary_metric,
            "throughput_metric": self.throughput_metric,
            "preference_order": list(self.preference_order),
            "algorithm": "constraint_first_lexicographic",
            "weighted_score": False,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "RankingPolicy":
        _validate_schema(
            payload,
            RANKING_POLICY_SCHEMA_NAME,
            RANKING_POLICY_SCHEMA_VERSION,
            "ranking policy",
        )
        allowed = {
            "schema",
            "primary_metric",
            "throughput_metric",
            "preference_order",
            "algorithm",
            "weighted_score",
        }
        _reject_unknown_fields(payload, allowed, "ranking policy")
        if payload.get("algorithm") != "constraint_first_lexicographic":
            raise RankingError("ranking algorithm cannot be changed")
        if payload.get("weighted_score") is not False:
            raise RankingError("weighted_score must remain false")
        try:
            order = payload["preference_order"]
            if not isinstance(order, list):
                raise RankingError("preference_order must be an array")
            return cls(
                primary_metric=payload["primary_metric"],
                throughput_metric=payload["throughput_metric"],
                preference_order=tuple(order),
            )
        except KeyError as exc:
            raise RankingError("ranking policy is missing a required field") from exc

    @classmethod
    def from_json(cls, encoded: str) -> "RankingPolicy":
        return cls.from_dict(_load_json_object(encoded, "ranking policy"))


@dataclass(frozen=True)
class RankingCandidate:
    """Full-trial evidence considered for provisional confirmation order."""

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
    failure_class: Optional[str]
    intervention_risk: str
    approval_required: bool
    semantic_change: bool
    change_count: int

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
            raise RankingError("project must be a non-empty string")
        normalized_project = self.project.strip()
        if len(normalized_project) > 128:
            raise RankingError("project must be at most 128 characters")
        object.__setattr__(self, "project", normalized_project)
        for field_name in (
            "contract_digest",
            "candidate_config_digest",
            "trial_request_digest",
            "execution_manifest_digest",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.phase not in TRIAL_PHASES:
            raise RankingError("phase must be one of {}".format(sorted(TRIAL_PHASES)))
        if self.status not in TRIAL_STATUSES:
            raise RankingError(
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
            raise RankingError(
                "workload_identity must be a WorkloadIdentity"
            )
        if self.worker_pid is not None:
            object.__setattr__(
                self,
                "worker_pid",
                _positive_int(self.worker_pid, "worker_pid"),
            )
        if self.failure_class is not None:
            object.__setattr__(
                self,
                "failure_class",
                _bounded_text(self.failure_class, "failure_class"),
            )
        if self.intervention_risk not in INTERVENTION_RISKS:
            raise RankingError(
                "intervention_risk must be one of {}".format(
                    sorted(INTERVENTION_RISKS)
                )
            )
        for field_name in ("approval_required", "semantic_change"):
            if not isinstance(getattr(self, field_name), bool):
                raise RankingError("{} must be a boolean".format(field_name))
        object.__setattr__(
            self,
            "change_count",
            _positive_int(self.change_count, "change_count"),
        )

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": RANKING_CANDIDATE_SCHEMA_NAME,
                "version": RANKING_CANDIDATE_SCHEMA_VERSION,
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
            "intervention": {
                "risk": self.intervention_risk,
                "approval_required": self.approval_required,
                "semantic_change": self.semantic_change,
                "change_count": self.change_count,
            },
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "RankingCandidate":
        _validate_schema(
            payload,
            RANKING_CANDIDATE_SCHEMA_NAME,
            RANKING_CANDIDATE_SCHEMA_VERSION,
            "ranking candidate",
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
            "intervention",
        }
        _reject_unknown_fields(payload, allowed, "ranking candidate")
        try:
            intervention = payload["intervention"]
            if not isinstance(intervention, dict):
                raise RankingError("intervention must be an object")
            _reject_unknown_fields(
                intervention,
                {"risk", "approval_required", "semantic_change", "change_count"},
                "ranking candidate intervention",
            )
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
                execution_manifest_digest=payload["execution_manifest_digest"],
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
                intervention_risk=intervention["risk"],
                approval_required=intervention["approval_required"],
                semantic_change=intervention["semantic_change"],
                change_count=intervention["change_count"],
            )
        except KeyError as exc:
            raise RankingError("ranking candidate is missing a required field") from exc
        except ValueError as exc:
            if isinstance(exc, RankingError):
                raise
            raise RankingError(str(exc)) from exc

    @classmethod
    def from_json(cls, encoded: str) -> "RankingCandidate":
        return cls.from_dict(_load_json_object(encoded, "ranking candidate"))


@dataclass(frozen=True)
class RankingReason:
    """One eligibility decision used before provisional ordering."""

    code: str
    outcome: str
    message: str
    expected: object
    observed: object

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not _REASON_CODE_PATTERN.fullmatch(self.code)
        ):
            raise RankingError("ranking reason code is invalid")
        if self.outcome not in RANKING_REASON_OUTCOMES:
            raise RankingError("ranking reason outcome is invalid")
        object.__setattr__(self, "message", _bounded_text(self.message, "message"))
        _validate_json_value(self.expected, "expected")
        _validate_json_value(self.observed, "observed")
        object.__setattr__(self, "expected", _freeze_json(self.expected))
        object.__setattr__(self, "observed", _freeze_json(self.observed))

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "outcome": self.outcome,
            "message": self.message,
            "expected": _thaw_json(self.expected),
            "observed": _thaw_json(self.observed),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RankingReason":
        if not isinstance(payload, dict):
            raise RankingError("ranking reason must be an object")
        _reject_unknown_fields(
            payload,
            {"code", "outcome", "message", "expected", "observed"},
            "ranking reason",
        )
        try:
            return cls(
                code=payload["code"],
                outcome=payload["outcome"],
                message=payload["message"],
                expected=payload["expected"],
                observed=payload["observed"],
            )
        except KeyError as exc:
            raise RankingError("ranking reason is missing a required field") from exc


@dataclass(frozen=True)
class CandidateAssessment:
    """Eligibility and provisional position for one candidate."""

    candidate_id: str
    run_id: str
    eligibility: str
    rank: Optional[int]
    preference_values: dict
    deciding_factor: Optional[str]
    reasons: Tuple[RankingReason, ...]

    def __post_init__(self) -> None:
        _validate_id(self.candidate_id, "candidate_id")
        _validate_id(self.run_id, "run_id")
        if self.eligibility not in CANDIDATE_ELIGIBILITY:
            raise RankingError("candidate eligibility is invalid")
        if self.rank is not None:
            object.__setattr__(self, "rank", _positive_int(self.rank, "rank"))
        if self.eligibility == "eligible" and self.rank is None:
            raise RankingError("eligible candidates require a provisional rank")
        if self.eligibility != "eligible" and self.rank is not None:
            raise RankingError("ineligible candidates cannot have a rank")
        if not isinstance(self.preference_values, dict):
            raise RankingError("preference_values must be an object")
        _validate_json_value(self.preference_values, "preference_values")
        object.__setattr__(
            self,
            "preference_values",
            MappingProxyType(deepcopy_json(self.preference_values)),
        )
        if self.deciding_factor is not None and self.deciding_factor not in RANKING_FACTORS:
            raise RankingError("deciding_factor is invalid")
        try:
            reasons = tuple(self.reasons)
        except TypeError as exc:
            raise RankingError("reasons must be an iterable") from exc
        if not reasons or any(not isinstance(reason, RankingReason) for reason in reasons):
            raise RankingError("reasons must contain RankingReason values")
        object.__setattr__(self, "reasons", reasons)
        expected_eligibility = _eligibility_from_reasons(reasons)
        if self.eligibility != expected_eligibility:
            raise RankingError("eligibility is inconsistent with reason outcomes")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "run_id": self.run_id,
            "eligibility": self.eligibility,
            "rank": self.rank,
            "preference_values": dict(self.preference_values),
            "deciding_factor": self.deciding_factor,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CandidateAssessment":
        if not isinstance(payload, dict):
            raise RankingError("candidate assessment must be an object")
        _reject_unknown_fields(
            payload,
            {
                "candidate_id",
                "run_id",
                "eligibility",
                "rank",
                "preference_values",
                "deciding_factor",
                "reasons",
            },
            "candidate assessment",
        )
        try:
            reasons = payload["reasons"]
            if not isinstance(reasons, list):
                raise RankingError("reasons must be an array")
            return cls(
                candidate_id=payload["candidate_id"],
                run_id=payload["run_id"],
                eligibility=payload["eligibility"],
                rank=payload["rank"],
                preference_values=payload["preference_values"],
                deciding_factor=payload["deciding_factor"],
                reasons=tuple(RankingReason.from_dict(item) for item in reasons),
            )
        except KeyError as exc:
            raise RankingError("candidate assessment is missing a required field") from exc


@dataclass(frozen=True)
class CandidateRanking:
    """Versioned provisional confirmation order for a candidate set."""

    campaign_id: str
    contract_digest: str
    policy: RankingPolicy
    confirmation_order: Tuple[str, ...]
    assessments: Tuple[CandidateAssessment, ...]

    def __post_init__(self) -> None:
        _validate_id(self.campaign_id, "campaign_id")
        _validate_digest(self.contract_digest, "contract_digest")
        if not isinstance(self.policy, RankingPolicy):
            raise RankingError("policy must be a RankingPolicy")
        try:
            order = tuple(self.confirmation_order)
            assessments = tuple(self.assessments)
        except TypeError as exc:
            raise RankingError("ranking collections must be iterable") from exc
        for candidate_id in order:
            _validate_id(candidate_id, "confirmation candidate_id")
        if len(order) != len(set(order)):
            raise RankingError("confirmation_order must contain unique ids")
        if any(not isinstance(item, CandidateAssessment) for item in assessments):
            raise RankingError("assessments must contain CandidateAssessment values")
        assessment_ids = [item.candidate_id for item in assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise RankingError("assessment candidate ids must be unique")
        eligible = sorted(
            (item for item in assessments if item.eligibility == "eligible"),
            key=lambda item: item.rank,
        )
        if [item.rank for item in eligible] != list(range(1, len(eligible) + 1)):
            raise RankingError("eligible ranks must be contiguous from one")
        if tuple(item.candidate_id for item in eligible) != order:
            raise RankingError("confirmation_order must match eligible ranks")
        object.__setattr__(self, "confirmation_order", order)
        object.__setattr__(self, "assessments", assessments)

    @property
    def next_confirmation_candidate_id(self) -> Optional[str]:
        return self.confirmation_order[0] if self.confirmation_order else None

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": RANKING_REPORT_SCHEMA_NAME,
                "version": RANKING_REPORT_SCHEMA_VERSION,
            },
            "campaign_id": self.campaign_id,
            "contract_digest": self.contract_digest,
            "policy": self.policy.to_dict(),
            "confirmation_order": list(self.confirmation_order),
            "next_confirmation_candidate_id": self.next_confirmation_candidate_id,
            "assessments": [item.to_dict() for item in self.assessments],
            "provisional": True,
            "recovery_verdict": None,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "CandidateRanking":
        _validate_schema(
            payload,
            RANKING_REPORT_SCHEMA_NAME,
            RANKING_REPORT_SCHEMA_VERSION,
            "candidate ranking",
        )
        allowed = {
            "schema",
            "campaign_id",
            "contract_digest",
            "policy",
            "confirmation_order",
            "next_confirmation_candidate_id",
            "assessments",
            "provisional",
            "recovery_verdict",
        }
        _reject_unknown_fields(payload, allowed, "candidate ranking")
        if payload.get("provisional") is not True:
            raise RankingError("candidate ranking must remain provisional")
        if payload.get("recovery_verdict") is not None:
            raise RankingError("ranking cannot contain a recovery verdict")
        try:
            order = payload["confirmation_order"]
            assessments = payload["assessments"]
            if not isinstance(order, list):
                raise RankingError("confirmation_order must be an array")
            if not isinstance(assessments, list):
                raise RankingError("assessments must be an array")
            report = cls(
                campaign_id=payload["campaign_id"],
                contract_digest=payload["contract_digest"],
                policy=RankingPolicy.from_dict(payload["policy"]),
                confirmation_order=tuple(order),
                assessments=tuple(
                    CandidateAssessment.from_dict(item) for item in assessments
                ),
            )
            next_candidate = payload["next_confirmation_candidate_id"]
        except KeyError as exc:
            raise RankingError("candidate ranking is missing a required field") from exc
        if next_candidate != report.next_confirmation_candidate_id:
            raise RankingError("next confirmation candidate is inconsistent")
        return report

    @classmethod
    def from_json(cls, encoded: str) -> "CandidateRanking":
        return cls.from_dict(_load_json_object(encoded, "candidate ranking"))


def ranking_digest(ranking: CandidateRanking) -> str:
    if not isinstance(ranking, CandidateRanking):
        raise RankingError("ranking must be a CandidateRanking")
    return hashlib.sha256(ranking.to_json().encode("utf-8")).hexdigest()


def rank_candidates(
    contract: RecoveryContract,
    *,
    campaign_id: str,
    policy: RankingPolicy,
    candidates: Iterable[RankingCandidate],
) -> CandidateRanking:
    """Gate and order all supplied full-trial candidates deterministically."""
    if not isinstance(contract, RecoveryContract):
        raise RankingError("contract must be a RecoveryContract")
    _validate_id(campaign_id, "campaign_id")
    if not isinstance(policy, RankingPolicy):
        raise RankingError("policy must be a RankingPolicy")
    primary_guard = policy.validate_against(contract)
    try:
        items = tuple(candidates)
    except TypeError as exc:
        raise RankingError("candidates must be an iterable") from exc
    if len(items) > MAX_RANKING_CANDIDATES:
        raise RankingError(
            "candidates exceed the {}-item ranking limit".format(
                MAX_RANKING_CANDIDATES
            )
        )
    if any(not isinstance(item, RankingCandidate) for item in items):
        raise RankingError("candidates must contain RankingCandidate values")
    _validate_unique_candidate_artifacts(items)

    expected_contract_digest = contract_digest(contract)
    provisional = []
    for candidate in items:
        reasons = _eligibility_reasons(
            contract,
            campaign_id,
            expected_contract_digest,
            candidate,
        )
        eligibility = _eligibility_from_reasons(reasons)
        values = _preference_values(policy, candidate)
        provisional.append((candidate, reasons, eligibility, values))

    eligible = [item for item in provisional if item[2] == "eligible"]
    eligible.sort(
        key=lambda item: _preference_key(
            policy,
            primary_guard,
            item[0],
        )
    )
    ranks = {item[0].candidate_id: index + 1 for index, item in enumerate(eligible)}
    deciding = _deciding_factors(policy, eligible)

    assessments = []
    for candidate, reasons, eligibility, values in provisional:
        assessments.append(
            CandidateAssessment(
                candidate_id=candidate.candidate_id,
                run_id=candidate.run_id,
                eligibility=eligibility,
                rank=ranks.get(candidate.candidate_id),
                preference_values=values,
                deciding_factor=deciding.get(candidate.candidate_id),
                reasons=reasons,
            )
        )
    assessments.sort(
        key=lambda item: (
            0 if item.rank is not None else 1,
            item.rank if item.rank is not None else 0,
            item.candidate_id,
        )
    )
    order = tuple(item[0].candidate_id for item in eligible)
    return CandidateRanking(
        campaign_id=campaign_id,
        contract_digest=expected_contract_digest,
        policy=policy,
        confirmation_order=order,
        assessments=tuple(assessments),
    )


def _eligibility_reasons(
    contract: RecoveryContract,
    campaign_id: str,
    expected_contract_digest: str,
    candidate: RankingCandidate,
) -> Tuple[RankingReason, ...]:
    reasons = []
    bindings = (
        ("binding.contract", expected_contract_digest, candidate.contract_digest),
        ("binding.campaign", campaign_id, candidate.campaign_id),
        ("binding.project", contract.project, candidate.project),
        ("binding.source_run", contract.source_run_id, candidate.source_run_id),
    )
    for code, expected, observed in bindings:
        reasons.append(
            _reason(
                code,
                "pass" if observed == expected else "fail",
                "Candidate evidence must match its declared campaign contract.",
                expected,
                observed,
            )
        )
    reasons.append(
        _reason(
            "execution.phase",
            "pass" if candidate.phase == "full" else "fail",
            "Only a completed full trial is eligible for confirmation.",
            "full",
            candidate.phase,
        )
    )
    reasons.append(
        _reason(
            "execution.status",
            "pass" if candidate.status == "success" else "fail",
            "The full trial must finish successfully.",
            "success",
            candidate.status,
        )
    )
    reasons.append(
        _reason(
            "execution.failure_class",
            "pass" if candidate.failure_class is None else "fail",
            "The full trial cannot contain an OOM or another failure class.",
            None,
            candidate.failure_class,
        )
    )
    reasons.append(
        _reason(
            "execution.worker_pid",
            "pass" if candidate.worker_pid is not None else "missing",
            "The isolated worker process identity must be recorded.",
            "positive integer",
            candidate.worker_pid,
        )
    )
    permissions = contract.permissions
    reasons.append(
        _reason(
            "intervention.approval_scope",
            (
                "pass"
                if not candidate.approval_required
                or permissions.allow_approval_required
                else "fail"
            ),
            "The candidate intervention must stay inside contract authority.",
            permissions.allow_approval_required,
            candidate.approval_required,
        )
    )
    reasons.append(
        _reason(
            "intervention.semantic_scope",
            (
                "pass"
                if not candidate.semantic_change
                or permissions.allow_semantic_changes
                else "fail"
            ),
            "Semantic intervention changes require contract authority.",
            permissions.allow_semantic_changes,
            candidate.semantic_change,
        )
    )
    reasons.append(
        _reason(
            "intervention.high_risk_scope",
            (
                "pass"
                if candidate.intervention_risk != "high"
                or permissions.allow_high_risk
                else "fail"
            ),
            "High-risk intervention changes require contract authority.",
            permissions.allow_high_risk,
            candidate.intervention_risk == "high",
        )
    )
    minimum = contract.verification.minimum_progress_steps
    if candidate.progress_steps is None:
        progress_outcome = "missing"
    elif candidate.progress_steps >= minimum:
        progress_outcome = "pass"
    else:
        progress_outcome = "fail"
    reasons.append(
        _reason(
            "execution.progress",
            progress_outcome,
            "The full trial must reach the contract's minimum progress.",
            {"minimum_steps": minimum},
            candidate.progress_steps,
        )
    )
    for guard in contract.verification.metric_guards:
        reasons.append(_metric_reason(guard, candidate.metrics))
    maximum_vram = contract.verification.max_peak_vram_bytes
    if maximum_vram is not None:
        if candidate.peak_vram_bytes is None:
            vram_outcome = "missing"
        elif candidate.peak_vram_bytes <= maximum_vram:
            vram_outcome = "pass"
        else:
            vram_outcome = "fail"
        reasons.append(
            _reason(
                "resource.peak_vram_bytes",
                vram_outcome,
                "Peak VRAM must be observed and stay inside the contract.",
                {"maximum": maximum_vram},
                candidate.peak_vram_bytes,
            )
        )
    expected_identity = contract.verification.workload_identity
    for field_name in expected_identity.known_fields:
        expected_value = getattr(expected_identity, field_name)
        observed_value = getattr(candidate.workload_identity, field_name)
        if observed_value is None:
            outcome = "missing"
        elif observed_value == expected_value:
            outcome = "pass"
        else:
            outcome = "fail"
        reasons.append(
            _reason(
                "identity.{}".format(field_name),
                outcome,
                "Full-trial workload identity must match the contract.",
                expected_value,
                observed_value,
            )
        )
    return tuple(reasons)


def _metric_reason(guard: MetricGuard, metrics: Dict[str, float]) -> RankingReason:
    observed = metrics.get(guard.name)
    boundary = guard.acceptance_boundary
    if observed is None:
        outcome = "missing"
    elif guard.direction == "maximize":
        outcome = "pass" if observed >= boundary else "fail"
    else:
        outcome = "pass" if observed <= boundary else "fail"
    return _reason(
        "metric.{}".format(_safe_factor_name(guard.name)),
        outcome,
        "Metric must satisfy its contract boundary before ranking.",
        {
            "direction": guard.direction,
            "boundary": boundary,
        },
        observed,
    )


def _preference_values(policy: RankingPolicy, candidate: RankingCandidate) -> dict:
    return {
        "primary_metric": candidate.metrics.get(policy.primary_metric),
        "peak_vram_bytes": candidate.peak_vram_bytes,
        "throughput": (
            candidate.metrics.get(policy.throughput_metric)
            if policy.throughput_metric is not None
            else None
        ),
        "intervention_risk": candidate.intervention_risk,
        "semantic_change": candidate.semantic_change,
        "approval_required": candidate.approval_required,
        "change_count": candidate.change_count,
    }


def _preference_key(
    policy: RankingPolicy,
    primary_guard: MetricGuard,
    candidate: RankingCandidate,
) -> tuple:
    values = _preference_values(policy, candidate)
    key = []
    for factor in policy.preference_order:
        value = values[factor]
        if factor == "primary_metric":
            key.append(-value if primary_guard.direction == "maximize" else value)
        elif factor == "peak_vram_bytes":
            key.append(value if value is not None else math.inf)
        elif factor == "throughput":
            key.append(-value if value is not None else math.inf)
        elif factor == "intervention_risk":
            key.append(_RISK_RANK[value])
        elif factor in {"semantic_change", "approval_required"}:
            key.append(1 if value else 0)
        elif factor == "change_count":
            key.append(value)
    key.append(candidate.candidate_id)
    return tuple(key)


def _deciding_factors(policy: RankingPolicy, eligible: list) -> dict:
    factors = {}
    previous = None
    for candidate, _, _, values in eligible:
        if previous is None:
            factors[candidate.candidate_id] = policy.preference_order[0]
        else:
            previous_values = previous[3]
            deciding = next(
                (
                    factor
                    for factor in policy.preference_order
                    if values[factor] != previous_values[factor]
                ),
                None,
            )
            factors[candidate.candidate_id] = deciding
        previous = (candidate, None, None, values)
    return factors


def _validate_unique_candidate_artifacts(items: tuple) -> None:
    fields = (
        "candidate_id",
        "trial_id",
        "run_id",
        "candidate_config_digest",
        "trial_request_digest",
        "execution_manifest_digest",
    )
    for field_name in fields:
        values = [getattr(item, field_name) for item in items]
        if len(values) != len(set(values)):
            raise RankingError(
                "candidate inputs contain duplicate {} values".format(field_name)
            )


def _eligibility_from_reasons(reasons: Tuple[RankingReason, ...]) -> str:
    if any(reason.outcome == "fail" for reason in reasons):
        return "rejected"
    if any(reason.outcome == "missing" for reason in reasons):
        return "insufficient_evidence"
    return "eligible"


def _reason(code, outcome, message, expected, observed) -> RankingReason:
    return RankingReason(code, outcome, message, expected, observed)


def deepcopy_json(value: dict) -> dict:
    return json.loads(_stable_json(value))


def _normalize_metrics(metrics: dict) -> Dict[str, float]:
    if not isinstance(metrics, dict):
        raise RankingError("metrics must be an object")
    normalized = {}
    for name, value in metrics.items():
        _validate_metric_name(name, "metric name")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RankingError("metric {!r} must be a finite number".format(name))
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RankingError("metric {!r} must be a finite number".format(name))
        normalized[name] = numeric
    return normalized


def _validate_schema(payload, name, version, artifact) -> None:
    if not isinstance(payload, dict):
        raise RankingError("{} must be an object".format(artifact))
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        raise RankingError("{} schema must be an object".format(artifact))
    _reject_unknown_fields(schema, {"name", "version"}, "{} schema".format(artifact))
    if schema.get("name") != name:
        raise RankingError("{} schema.name must be {!r}".format(artifact, name))
    if schema.get("version") != version:
        raise RankingError("{} schema.version must be {!r}".format(artifact, version))


def _validate_metric_name(value, field_name) -> None:
    if not isinstance(value, str) or not _METRIC_PATTERN.fullmatch(value):
        raise RankingError("{} is invalid".format(field_name))


def _validate_id(value, field_name) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise RankingError("{} is invalid".format(field_name))


def _validate_digest(value, field_name) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise RankingError("{} must be a lowercase SHA-256 digest".format(field_name))


def _positive_int(value, field_name) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RankingError("{} must be a positive integer".format(field_name))
    return value


def _nonnegative_int(value, field_name) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RankingError("{} must be a non-negative integer".format(field_name))
    return value


def _bounded_text(value, field_name) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankingError("{} must be non-empty text".format(field_name))
    normalized = value.strip()
    if len(normalized) > MAX_REASON_TEXT_LENGTH:
        raise RankingError("{} is too long".format(field_name))
    return normalized


def _validate_json_value(value, field_name) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RankingError("{} must contain finite JSON data".format(field_name)) from exc


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _safe_factor_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", value).lower()


def _stable_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _load_json_object(encoded, artifact) -> dict:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RankingError("invalid {} JSON".format(artifact)) from exc
    if not isinstance(payload, dict):
        raise RankingError("{} JSON must contain an object".format(artifact))
    return payload


def _reject_unknown_fields(payload, allowed, artifact) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RankingError("{} contains unknown fields: {}".format(artifact, unknown))


__all__ = [
    "CANDIDATE_ELIGIBILITY",
    "DEFAULT_PREFERENCE_ORDER",
    "INTERVENTION_RISKS",
    "MAX_RANKING_CANDIDATES",
    "RANKING_CANDIDATE_SCHEMA_NAME",
    "RANKING_CANDIDATE_SCHEMA_VERSION",
    "RANKING_FACTORS",
    "RANKING_POLICY_SCHEMA_NAME",
    "RANKING_POLICY_SCHEMA_VERSION",
    "RANKING_REASON_OUTCOMES",
    "RANKING_REPORT_SCHEMA_NAME",
    "RANKING_REPORT_SCHEMA_VERSION",
    "CandidateAssessment",
    "CandidateRanking",
    "RankingCandidate",
    "RankingError",
    "RankingPolicy",
    "RankingReason",
    "rank_candidates",
    "ranking_digest",
]