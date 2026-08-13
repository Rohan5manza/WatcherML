"""Typed, evidence-linked intervention proposals for WatcherML recovery.

Capabilities describe what a project can control.  This module describes one
bounded proposal, resolves its canonical capability names to the project's
actual config/environment targets, and materializes trial inputs without
mutating the source configuration.

It intentionally does not choose proposals, rank candidates, run trials, or
declare recovery.  The deterministic OOM policy chooses proposals; the trial
runner executes them; the verifier decides whether any result is a recovery.

V1 intervention surfaces remain deliberately narrow:

* JSON configuration values already declared or safely detected;
* explicitly allowlisted environment capabilities from ``capabilities.py``.

Code, dependencies, datasets, shell commands, arbitrary environment variables,
and serialized Python objects cannot be represented by this schema.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from .capabilities import (
    CapabilityError,
    CapabilityManifest,
    CapabilityTransitionError,
    UnsupportedCapabilityError,
    get_config_value,
)
from .entrypoint import EntrypointError, validate_config


INTERVENTION_PROPOSAL_SCHEMA_NAME = "watcherml.intervention-proposal"
INTERVENTION_PROPOSAL_SCHEMA_VERSION = "1.0"
INTERVENTION_RESOLUTION_SCHEMA_NAME = "watcherml.intervention-resolution"
INTERVENTION_RESOLUTION_SCHEMA_VERSION = "1.0"
INTERVENTION_AUTHORIZATION_SCHEMA_NAME = "watcherml.intervention-authorization"
INTERVENTION_AUTHORIZATION_SCHEMA_VERSION = "1.0"

MAX_CHANGES_PER_INTERVENTION = 4
MAX_EVIDENCE_REFERENCES = 32
MAX_TEXT_LENGTH = 4_000

INTERVENTION_OPERATIONS = frozenset(
    {"decrease", "increase", "enable", "disable", "set"}
)
INTERVENTION_PROPOSERS = frozenset({"deterministic_policy", "user"})

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RULE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_EVIDENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")

_PERMISSION_RANK = {
    "automatic": 0,
    "approval_required": 1,
    "disabled": 2,
}
_RISK_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

Scalar = Union[str, int, float, bool]


class InterventionError(ValueError):
    """Base exception for an invalid intervention artifact."""


class InterventionResolutionError(InterventionError):
    """Raised when a proposal cannot resolve against current capabilities."""


class StaleInterventionError(InterventionResolutionError):
    """Raised when config/environment facts changed after discovery."""


class InterventionAuthorizationError(InterventionError):
    """Raised when an approval-required proposal lacks valid authorization."""


@dataclass(frozen=True)
class InterventionChange:
    """One canonical capability transition requested by a proposal."""

    capability_id: str
    operation: str
    proposed_value: Scalar

    def __post_init__(self) -> None:
        _validate_id(self.capability_id, "capability_id")
        if self.operation not in INTERVENTION_OPERATIONS:
            raise InterventionError(
                "operation must be one of {}".format(
                    sorted(INTERVENTION_OPERATIONS)
                )
            )
        object.__setattr__(
            self,
            "proposed_value",
            _validate_scalar(self.proposed_value, "proposed_value"),
        )

    def to_dict(self) -> dict:
        return {
            "capability_id": self.capability_id,
            "operation": self.operation,
            "proposed_value": self.proposed_value,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InterventionChange":
        if not isinstance(payload, dict):
            raise InterventionError("intervention change must be an object")
        _reject_unknown_fields(
            payload,
            {"capability_id", "operation", "proposed_value"},
            "intervention change",
        )
        try:
            return cls(
                capability_id=payload["capability_id"],
                operation=payload["operation"],
                proposed_value=payload["proposed_value"],
            )
        except (KeyError, TypeError) as exc:
            raise InterventionError(
                "intervention change is missing a required field"
            ) from exc


@dataclass(frozen=True)
class InterventionProposal:
    """An auditable proposal, not a recovery claim or trial result."""

    proposal_id: str
    policy_rule: str
    changes: Tuple[InterventionChange, ...]
    rationale: str
    expected_effect: str
    evidence_refs: Tuple[str, ...]
    proposer: str = "deterministic_policy"

    def __post_init__(self) -> None:
        _validate_id(self.proposal_id, "proposal_id")
        if (
            not isinstance(self.policy_rule, str)
            or not _RULE_PATTERN.fullmatch(self.policy_rule)
        ):
            raise InterventionError(
                "policy_rule must be a lowercase machine-readable rule id"
            )
        if self.proposer not in INTERVENTION_PROPOSERS:
            raise InterventionError(
                "proposer must be one of {}".format(
                    sorted(INTERVENTION_PROPOSERS)
                )
            )

        try:
            changes = tuple(self.changes)
        except TypeError as exc:
            raise InterventionError("changes must be an iterable") from exc
        if not changes:
            raise InterventionError("an intervention requires at least one change")
        if len(changes) > MAX_CHANGES_PER_INTERVENTION:
            raise InterventionError(
                "an intervention may contain at most {} coupled changes".format(
                    MAX_CHANGES_PER_INTERVENTION
                )
            )
        if any(not isinstance(change, InterventionChange) for change in changes):
            raise InterventionError(
                "changes must contain InterventionChange values"
            )
        capability_ids = [change.capability_id for change in changes]
        if len(capability_ids) != len(set(capability_ids)):
            raise InterventionError(
                "a proposal may change each capability at most once"
            )
        object.__setattr__(self, "changes", changes)

        object.__setattr__(
            self,
            "rationale",
            _validate_text(self.rationale, "rationale"),
        )
        object.__setattr__(
            self,
            "expected_effect",
            _validate_text(self.expected_effect, "expected_effect"),
        )

        try:
            evidence_refs = tuple(self.evidence_refs)
        except TypeError as exc:
            raise InterventionError("evidence_refs must be an iterable") from exc
        if not evidence_refs:
            raise InterventionError(
                "an intervention requires at least one evidence reference"
            )
        if len(evidence_refs) > MAX_EVIDENCE_REFERENCES:
            raise InterventionError(
                "an intervention may cite at most {} evidence references".format(
                    MAX_EVIDENCE_REFERENCES
                )
            )
        if len(evidence_refs) != len(set(evidence_refs)):
            raise InterventionError("evidence references must be unique")
        for reference in evidence_refs:
            if (
                not isinstance(reference, str)
                or not _EVIDENCE_PATTERN.fullmatch(reference)
            ):
                raise InterventionError(
                    "invalid evidence reference {!r}".format(reference)
                )
        object.__setattr__(self, "evidence_refs", evidence_refs)

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": INTERVENTION_PROPOSAL_SCHEMA_NAME,
                "version": INTERVENTION_PROPOSAL_SCHEMA_VERSION,
            },
            "proposal_id": self.proposal_id,
            "policy_rule": self.policy_rule,
            "proposer": self.proposer,
            "changes": [change.to_dict() for change in self.changes],
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "evidence_refs": list(self.evidence_refs),
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "InterventionProposal":
        _validate_schema(
            payload,
            name=INTERVENTION_PROPOSAL_SCHEMA_NAME,
            version=INTERVENTION_PROPOSAL_SCHEMA_VERSION,
        )
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "proposal_id",
                "policy_rule",
                "proposer",
                "changes",
                "rationale",
                "expected_effect",
                "evidence_refs",
            },
            "intervention proposal",
        )
        changes = payload.get("changes")
        evidence_refs = payload.get("evidence_refs")
        if not isinstance(changes, list):
            raise InterventionError("changes must be an array")
        if not isinstance(evidence_refs, list):
            raise InterventionError("evidence_refs must be an array")
        try:
            return cls(
                proposal_id=payload["proposal_id"],
                policy_rule=payload["policy_rule"],
                proposer=payload["proposer"],
                changes=tuple(
                    InterventionChange.from_dict(change) for change in changes
                ),
                rationale=payload["rationale"],
                expected_effect=payload["expected_effect"],
                evidence_refs=tuple(evidence_refs),
            )
        except KeyError as exc:
            raise InterventionError(
                "intervention proposal is missing a required field"
            ) from exc

    @classmethod
    def from_json(cls, encoded: str) -> "InterventionProposal":
        return cls.from_dict(_load_json_object(encoded, "intervention proposal"))


@dataclass(frozen=True)
class ResolvedChange:
    """A capability transition bound to one concrete project target."""

    capability_id: str
    operation: str
    location: str
    target: str
    before: Scalar
    after: Scalar
    permission: str
    risk: str
    semantic_change: bool
    expected_effect: str

    def __post_init__(self) -> None:
        _validate_id(self.capability_id, "capability_id")
        if self.operation not in INTERVENTION_OPERATIONS:
            raise InterventionError("resolved change has an invalid operation")
        if self.location not in {"config", "environment"}:
            raise InterventionError("resolved change has an invalid location")
        if not isinstance(self.target, str) or not self.target:
            raise InterventionError("resolved change target must be non-empty")
        _validate_scalar(self.before, "before")
        _validate_scalar(self.after, "after")
        if self.permission not in _PERMISSION_RANK:
            raise InterventionError("resolved change has an invalid permission")
        if self.risk not in _RISK_RANK:
            raise InterventionError("resolved change has an invalid risk")
        if not isinstance(self.semantic_change, bool):
            raise InterventionError("semantic_change must be a boolean")
        _validate_text(self.expected_effect, "expected_effect")

    def to_dict(self) -> dict:
        return {
            "capability_id": self.capability_id,
            "operation": self.operation,
            "location": self.location,
            "target": self.target,
            "before": self.before,
            "after": self.after,
            "permission": self.permission,
            "risk": self.risk,
            "semantic_change": self.semantic_change,
            "expected_effect": self.expected_effect,
        }


@dataclass(frozen=True)
class ResolvedIntervention:
    """A proposal resolved against a specific capability manifest/baseline."""

    proposal: InterventionProposal
    changes: Tuple[ResolvedChange, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, InterventionProposal):
            raise InterventionError("proposal must be an InterventionProposal")
        try:
            changes = tuple(self.changes)
        except TypeError as exc:
            raise InterventionError("resolved changes must be an iterable") from exc
        if any(not isinstance(change, ResolvedChange) for change in changes):
            raise InterventionError(
                "changes must contain ResolvedChange values"
            )
        if len(changes) != len(self.proposal.changes):
            raise InterventionError(
                "resolved changes must correspond one-to-one with proposal changes"
            )
        proposal_ids = tuple(
            change.capability_id for change in self.proposal.changes
        )
        resolved_ids = tuple(change.capability_id for change in changes)
        if resolved_ids != proposal_ids:
            raise InterventionError(
                "resolved changes must preserve proposal change order"
            )
        object.__setattr__(self, "changes", changes)

    @property
    def required_permission(self) -> str:
        return max(
            (change.permission for change in self.changes),
            key=lambda value: _PERMISSION_RANK[value],
        )

    @property
    def maximum_risk(self) -> str:
        return max(
            (change.risk for change in self.changes),
            key=lambda value: _RISK_RANK[value],
        )

    @property
    def semantic_change(self) -> bool:
        return any(change.semantic_change for change in self.changes)

    @property
    def approval_required(self) -> bool:
        return self.required_permission == "approval_required"

    @property
    def config_patch(self) -> Dict[str, Scalar]:
        return {
            change.target: change.after
            for change in self.changes
            if change.location == "config"
        }

    @property
    def environment_patch(self) -> Dict[str, str]:
        return {
            change.target: str(change.after)
            for change in self.changes
            if change.location == "environment"
        }

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": INTERVENTION_RESOLUTION_SCHEMA_NAME,
                "version": INTERVENTION_RESOLUTION_SCHEMA_VERSION,
            },
            "proposal": self.proposal.to_dict(),
            "proposal_digest": proposal_digest(self.proposal),
            "required_permission": self.required_permission,
            "maximum_risk": self.maximum_risk,
            "semantic_change": self.semantic_change,
            "changes": [change.to_dict() for change in self.changes],
            "config_patch": self.config_patch,
            "environment_patch": self.environment_patch,
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())


@dataclass(frozen=True)
class InterventionAuthorization:
    """Proposal-specific human authorization for a gated intervention."""

    proposal_id: str
    proposal_digest: str
    approved_by: str
    reason: str
    approved_at: float

    def __post_init__(self) -> None:
        _validate_id(self.proposal_id, "proposal_id")
        if (
            not isinstance(self.proposal_digest, str)
            or not _SHA256_PATTERN.fullmatch(self.proposal_digest)
        ):
            raise InterventionError(
                "proposal_digest must be a lowercase SHA-256 digest"
            )
        object.__setattr__(
            self,
            "approved_by",
            _validate_text(self.approved_by, "approved_by", maximum=256),
        )
        object.__setattr__(
            self,
            "reason",
            _validate_text(self.reason, "reason"),
        )
        if (
            isinstance(self.approved_at, bool)
            or not isinstance(self.approved_at, Real)
            or not math.isfinite(float(self.approved_at))
            or float(self.approved_at) <= 0
        ):
            raise InterventionError("approved_at must be a positive timestamp")
        object.__setattr__(self, "approved_at", float(self.approved_at))

    @classmethod
    def approve(
        cls,
        proposal: InterventionProposal,
        *,
        approved_by: str,
        reason: str,
        approved_at: Optional[float] = None,
    ) -> "InterventionAuthorization":
        """Create an authorization bound to the exact proposal bytes."""
        if not isinstance(proposal, InterventionProposal):
            raise InterventionError("proposal must be an InterventionProposal")
        return cls(
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal_digest(proposal),
            approved_by=approved_by,
            reason=reason,
            approved_at=time.time() if approved_at is None else approved_at,
        )

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": INTERVENTION_AUTHORIZATION_SCHEMA_NAME,
                "version": INTERVENTION_AUTHORIZATION_SCHEMA_VERSION,
            },
            "proposal_id": self.proposal_id,
            "proposal_digest": self.proposal_digest,
            "approved_by": self.approved_by,
            "reason": self.reason,
            "approved_at": self.approved_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "InterventionAuthorization":
        _validate_schema(
            payload,
            name=INTERVENTION_AUTHORIZATION_SCHEMA_NAME,
            version=INTERVENTION_AUTHORIZATION_SCHEMA_VERSION,
        )
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "proposal_id",
                "proposal_digest",
                "approved_by",
                "reason",
                "approved_at",
            },
            "intervention authorization",
        )
        try:
            return cls(
                proposal_id=payload["proposal_id"],
                proposal_digest=payload["proposal_digest"],
                approved_by=payload["approved_by"],
                reason=payload["reason"],
                approved_at=payload["approved_at"],
            )
        except KeyError as exc:
            raise InterventionError(
                "authorization is missing a required field"
            ) from exc


@dataclass(frozen=True)
class InterventionApplication:
    """Trial inputs produced from a resolved, authorized intervention."""

    proposal_id: str
    config: dict
    environment_patch: Dict[str, str] = field(default_factory=dict)
    authorization: Optional[InterventionAuthorization] = None

    def __post_init__(self) -> None:
        _validate_id(self.proposal_id, "proposal_id")
        try:
            normalized_config = validate_config(self.config)
        except EntrypointError as exc:
            raise InterventionError(str(exc)) from exc
        object.__setattr__(self, "config", normalized_config)

        if not isinstance(self.environment_patch, dict):
            raise InterventionError("environment_patch must be an object")
        normalized_environment = {}
        for key, value in self.environment_patch.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise InterventionError(
                    "environment_patch must contain string keys and values"
                )
            normalized_environment[key] = value
        object.__setattr__(self, "environment_patch", normalized_environment)

        if self.authorization is not None and not isinstance(
            self.authorization,
            InterventionAuthorization,
        ):
            raise InterventionError(
                "authorization must be an InterventionAuthorization or None"
            )


def resolve_intervention(
    proposal: InterventionProposal,
    manifest: CapabilityManifest,
    source_config: dict,
    *,
    source_environment: Optional[Mapping[str, str]] = None,
) -> ResolvedIntervention:
    """Resolve and validate a proposal against current baseline facts.

    Resolution fails if a capability is absent, disabled, has an illegal
    transition, or no longer matches the supplied source config/environment.
    No source object is mutated.
    """
    if not isinstance(proposal, InterventionProposal):
        raise InterventionResolutionError(
            "proposal must be an InterventionProposal"
        )
    if not isinstance(manifest, CapabilityManifest):
        raise InterventionResolutionError(
            "manifest must be a CapabilityManifest"
        )
    try:
        normalized_config = validate_config(source_config)
    except EntrypointError as exc:
        raise InterventionResolutionError(str(exc)) from exc

    resolved_changes = []
    for requested in proposal.changes:
        try:
            capability = manifest.require(requested.capability_id)
        except (UnsupportedCapabilityError, CapabilityError) as exc:
            raise InterventionResolutionError(str(exc)) from exc

        current = _current_surface_value(
            capability.location,
            capability.target,
            normalized_config,
            source_environment,
        )
        if not _same_scalar(current, capability.current_value):
            raise StaleInterventionError(
                "baseline for {!r} changed after capability discovery: "
                "manifest={!r}, current={!r}".format(
                    capability.capability_id,
                    capability.current_value,
                    current,
                )
            )
        try:
            after = capability.validate_transition(
                requested.operation,
                requested.proposed_value,
            )
        except CapabilityTransitionError as exc:
            raise InterventionResolutionError(str(exc)) from exc

        resolved_changes.append(
            ResolvedChange(
                capability_id=capability.capability_id,
                operation=requested.operation,
                location=capability.location,
                target=capability.target,
                before=capability.current_value,
                after=after,
                permission=capability.permission,
                risk=capability.risk,
                semantic_change=capability.semantic_change,
                expected_effect=capability.expected_effect,
            )
        )

    resolved = ResolvedIntervention(
        proposal=proposal,
        changes=tuple(resolved_changes),
    )
    if resolved.required_permission == "disabled":
        raise InterventionResolutionError(
            "proposal contains a disabled capability"
        )
    return resolved


def materialize_intervention(
    resolved: ResolvedIntervention,
    manifest: CapabilityManifest,
    source_config: dict,
    *,
    source_environment: Optional[Mapping[str, str]] = None,
    authorization: Optional[InterventionAuthorization] = None,
) -> InterventionApplication:
    """Create complete trial config plus the minimal environment patch.

    Approval-required proposals need a proposal-specific authorization.  This
    function rechecks baseline values to prevent applying a previously resolved
    proposal to a changed configuration.  It returns only changed environment
    keys, never a copy of the parent environment or its credentials.
    """
    if not isinstance(resolved, ResolvedIntervention):
        raise InterventionError("resolved must be a ResolvedIntervention")
    # ResolvedChange is a public audit type, so never trust a caller-created
    # instance by itself. Re-resolution binds every field back to the sealed
    # capability manifest and prevents forged automatic/config targets.
    canonical = resolve_intervention(
        resolved.proposal,
        manifest,
        source_config,
        source_environment=source_environment,
    )
    if canonical != resolved:
        raise InterventionResolutionError(
            "resolved intervention does not match the capability manifest"
        )
    _validate_authorization(resolved, authorization)

    try:
        trial_config = validate_config(source_config)
    except EntrypointError as exc:
        raise InterventionError(str(exc)) from exc

    environment_patch = {}
    for change in resolved.changes:
        current = _current_surface_value(
            change.location,
            change.target,
            trial_config,
            source_environment,
        )
        if not _same_scalar(current, change.before):
            raise StaleInterventionError(
                "baseline target {!r} changed before materialization".format(
                    change.target
                )
            )
        if change.location == "config":
            _set_existing_config_value(
                trial_config,
                change.target,
                change.after,
            )
        elif change.location == "environment":
            environment_patch[change.target] = str(change.after)
        else:
            raise InterventionError(
                "unsupported intervention location {!r}".format(
                    change.location
                )
            )

    # Revalidate the complete patched config before it reaches TrialRequest.
    trial_config = validate_config(trial_config)
    return InterventionApplication(
        proposal_id=resolved.proposal.proposal_id,
        config=trial_config,
        environment_patch=environment_patch,
        authorization=authorization,
    )


def _validate_authorization(
    resolved: ResolvedIntervention,
    authorization: Optional[InterventionAuthorization],
) -> None:
    if authorization is not None:
        if not isinstance(authorization, InterventionAuthorization):
            raise InterventionAuthorizationError(
                "authorization must be an InterventionAuthorization"
            )
        if authorization.proposal_id != resolved.proposal.proposal_id:
            raise InterventionAuthorizationError(
                "authorization proposal_id does not match the intervention"
            )
        if authorization.proposal_digest != proposal_digest(resolved.proposal):
            raise InterventionAuthorizationError(
                "authorization digest does not match the intervention proposal"
            )
    if resolved.approval_required and authorization is None:
        raise InterventionAuthorizationError(
            "proposal {!r} requires explicit authorization".format(
                resolved.proposal.proposal_id
            )
        )


def _current_surface_value(
    location: str,
    target: str,
    source_config: dict,
    source_environment: Optional[Mapping[str, str]],
):
    if location == "config":
        try:
            return get_config_value(source_config, target)
        except CapabilityError as exc:
            raise StaleInterventionError(str(exc)) from exc
    if location == "environment":
        if source_environment is None:
            raise StaleInterventionError(
                "source_environment is required for environment capability {!r}".format(
                    target
                )
            )
        if not isinstance(source_environment, Mapping):
            raise InterventionResolutionError(
                "source_environment must be a string mapping"
            )
        if target not in source_environment:
            raise StaleInterventionError(
                "environment target {!r} is missing from the baseline".format(
                    target
                )
            )
        value = source_environment[target]
        if not isinstance(value, str):
            raise InterventionResolutionError(
                "environment baseline values must be strings"
            )
        return value
    raise InterventionResolutionError(
        "unsupported capability location {!r}".format(location)
    )


def _set_existing_config_value(config: dict, target: str, value: Scalar) -> None:
    segments = target.split(".")
    parent = config
    for segment in segments[:-1]:
        child = parent.get(segment)
        if not isinstance(child, dict):
            raise StaleInterventionError(
                "config target {!r} no longer exists".format(target)
            )
        parent = child
    leaf = segments[-1]
    if leaf not in parent:
        raise StaleInterventionError(
            "config target {!r} no longer exists".format(target)
        )
    parent[leaf] = value


def _same_scalar(left, right) -> bool:
    # bool is an int subclass. Exact types prevent True from matching 1 in a
    # stale-baseline check.
    return type(left) is type(right) and left == right


def _validate_scalar(value, field_name: str) -> Scalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise InterventionError(
                "{} must be a finite JSON scalar".format(field_name)
            )
        return normalized
    if isinstance(value, str):
        if len(value) > MAX_TEXT_LENGTH:
            raise InterventionError(
                "{} exceeds the {}-character limit".format(
                    field_name,
                    MAX_TEXT_LENGTH,
                )
            )
        return value
    raise InterventionError(
        "{} must be a string, number, or boolean".format(field_name)
    )


def _validate_text(
    value: str,
    field_name: str,
    *,
    maximum: int = MAX_TEXT_LENGTH,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InterventionError("{} must be a non-empty string".format(field_name))
    normalized = value.strip()
    if len(normalized) > maximum:
        raise InterventionError(
            "{} exceeds the {}-character limit".format(field_name, maximum)
        )
    return normalized


def _validate_id(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise InterventionError(
            "{} must match {}".format(field_name, _ID_PATTERN.pattern)
        )


def _validate_schema(payload: dict, *, name: str, version: str) -> None:
    if not isinstance(payload, dict):
        raise InterventionError("payload must be an object")
    schema = payload.get("schema") or {}
    if schema.get("name") != name:
        raise InterventionError("schema.name must be {!r}".format(name))
    if schema.get("version") != version:
        raise InterventionError("schema.version must be {!r}".format(version))


def _reject_unknown_fields(
    payload: dict,
    allowed: set,
    artifact_name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise InterventionError(
            "{} contains unknown fields: {}".format(artifact_name, unknown)
        )


def _stable_json(payload: dict) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InterventionError("artifact is not valid JSON: {}".format(exc)) from exc


def proposal_digest(proposal: InterventionProposal) -> str:
    """Return the stable SHA-256 identity used by approval records."""
    if not isinstance(proposal, InterventionProposal):
        raise InterventionError("proposal must be an InterventionProposal")
    return hashlib.sha256(proposal.to_json().encode("utf-8")).hexdigest()


def _load_json_object(encoded: str, artifact_name: str) -> dict:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise InterventionError("invalid {} JSON".format(artifact_name)) from exc
    if not isinstance(payload, dict):
        raise InterventionError("{} JSON must contain an object".format(artifact_name))
    return payload


__all__ = [
    "INTERVENTION_AUTHORIZATION_SCHEMA_NAME",
    "INTERVENTION_AUTHORIZATION_SCHEMA_VERSION",
    "INTERVENTION_PROPOSAL_SCHEMA_NAME",
    "INTERVENTION_PROPOSAL_SCHEMA_VERSION",
    "INTERVENTION_RESOLUTION_SCHEMA_NAME",
    "INTERVENTION_RESOLUTION_SCHEMA_VERSION",
    "MAX_CHANGES_PER_INTERVENTION",
    "InterventionApplication",
    "InterventionAuthorization",
    "InterventionAuthorizationError",
    "InterventionChange",
    "InterventionError",
    "InterventionProposal",
    "InterventionResolutionError",
    "ResolvedChange",
    "ResolvedIntervention",
    "StaleInterventionError",
    "materialize_intervention",
    "proposal_digest",
    "resolve_intervention",
]