"""Public deterministic CUDA OOM recovery integration for WatcherML v1.

This module is intentionally an integration layer, not another recovery
engine.  It connects the independently tested v1 components:

``failure capsule -> capabilities -> OOM policy -> authorized interventions
-> immutable contract -> isolated campaign -> independent verifier``

There is no in-process ``train_fn`` fallback, LLM diagnosis, hidden retry,
weighted recovery score, or unverified "best run".  A recovery is exposed as
verified only when ``verifier.py`` has accepted the complete confirmation set.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

from .campaign import (
    CampaignCandidate,
    CampaignResult,
    ExecutedTrial,
    RunObservation,
    run_campaign,
)
from .capabilities import (
    CapabilityManifest,
    DeclarationInput,
    discover_capabilities,
)
from .entrypoint import (
    TrainingEntrypoint,
    validate_entrypoint,
)
from .interventions import (
    InterventionAuthorization,
    InterventionProposal,
    materialize_intervention,
    resolve_intervention,
)
from .oom_policy import (
    DEFAULT_MAX_PROPOSALS,
    OOMPolicyPlan,
    plan_oom_interventions,
)
from .ranking import RankingPolicy
from .recovery_contract import (
    InterventionPermissions,
    RecoveryBudget,
    RecoveryContract,
    VerificationRequirements,
    WorkloadIdentity,
    contract_digest,
    validate_intervention_scope,
)
from .storage import Storage
from .trial_protocol import TrialRequest, atomic_write_json
from .trial_runner import (
    DEFAULT_TERMINATION_GRACE_SECONDS,
    TrialExecution,
    run_trial,
)


RECOVERY_PREPARATION_SCHEMA_NAME = "watcherml.recovery-preparation"
RECOVERY_PREPARATION_SCHEMA_VERSION = "1.0"
RECOVERY_RESULT_SCHEMA_NAME = "watcherml.recovery-result"
RECOVERY_RESULT_SCHEMA_VERSION = "1.0"

PROPOSAL_SKIP_CODES = frozenset(
    {
        "authorization_missing",
        "contract_scope_excluded",
    }
)

DEFAULT_PROGRESS_METRIC = "steps_completed"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_METRIC_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class RecoveryIntegrationError(RuntimeError):
    """Raised when the public recovery workflow cannot proceed safely."""


@dataclass(frozen=True)
class ProposalSkip:
    """One policy proposal deliberately excluded before GPU execution."""

    proposal_id: str
    policy_rule: str
    code: str
    reason: str

    def __post_init__(self) -> None:
        _validate_id(self.proposal_id, "proposal_id")
        if not isinstance(self.policy_rule, str) or not self.policy_rule:
            raise RecoveryIntegrationError("policy_rule must be non-empty")
        if self.code not in PROPOSAL_SKIP_CODES:
            raise RecoveryIntegrationError("proposal skip code is invalid")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise RecoveryIntegrationError("proposal skip reason must be non-empty")
        object.__setattr__(self, "reason", self.reason.strip())

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "policy_rule": self.policy_rule,
            "code": self.code,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProposalSkip":
        if not isinstance(payload, dict):
            raise RecoveryIntegrationError("proposal skip must be an object")
        _reject_unknown_fields(
            payload,
            {"proposal_id", "policy_rule", "code", "reason"},
            "proposal skip",
        )
        try:
            return cls(**payload)
        except KeyError as exc:
            raise RecoveryIntegrationError(
                "proposal skip is missing a required field"
            ) from exc


@dataclass(frozen=True, init=False)
class RecoveryPreparation:
    """Zero-compute, serializable review of one source OOM recovery plan."""

    contract: RecoveryContract
    capability_manifest: CapabilityManifest
    policy_plan: OOMPolicyPlan
    capsule_digest: str
    _source_environment_json: str = field(repr=False)
    _entrypoint_validation_json: str = field(repr=False)

    def __init__(
        self,
        *,
        contract: RecoveryContract,
        capability_manifest: CapabilityManifest,
        policy_plan: OOMPolicyPlan,
        capsule_digest: str,
        source_environment: Mapping[str, str],
        entrypoint_validation: dict,
    ) -> None:
        if not isinstance(contract, RecoveryContract):
            raise RecoveryIntegrationError("contract must be a RecoveryContract")
        if not isinstance(capability_manifest, CapabilityManifest):
            raise RecoveryIntegrationError(
                "capability_manifest must be a CapabilityManifest"
            )
        if not isinstance(policy_plan, OOMPolicyPlan):
            raise RecoveryIntegrationError("policy_plan must be an OOMPolicyPlan")
        if policy_plan.run_id != contract.source_run_id:
            raise RecoveryIntegrationError(
                "policy plan and recovery contract reference different source runs"
            )
        _validate_digest(capsule_digest, "capsule_digest")
        normalized_environment = _string_mapping(
            source_environment,
            "source_environment",
        )
        normalized_validation = _entrypoint_validation(entrypoint_validation)
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "capability_manifest", capability_manifest)
        object.__setattr__(self, "policy_plan", policy_plan)
        object.__setattr__(self, "capsule_digest", capsule_digest)
        object.__setattr__(
            self,
            "_source_environment_json",
            _stable_json(normalized_environment),
        )
        object.__setattr__(
            self,
            "_entrypoint_validation_json",
            _stable_json(normalized_validation),
        )

    @property
    def source_environment(self) -> Dict[str, str]:
        return json.loads(self._source_environment_json)

    @property
    def entrypoint_validation(self) -> dict:
        return json.loads(self._entrypoint_validation_json)

    @property
    def automatic_proposal_ids(self) -> Tuple[str, ...]:
        return self.policy_plan.automatic_proposal_ids

    @property
    def approval_required_proposal_ids(self) -> Tuple[str, ...]:
        return self.policy_plan.approval_required_proposal_ids

    def proposal(self, proposal_id: str) -> InterventionProposal:
        _validate_id(proposal_id, "proposal_id")
        matches = [
            item
            for item in self.policy_plan.proposals
            if item.proposal_id == proposal_id
        ]
        if len(matches) != 1:
            raise RecoveryIntegrationError(
                "proposal_id {!r} is not in this preparation".format(
                    proposal_id
                )
            )
        return matches[0]

    def authorize(
        self,
        proposal_id: str,
        *,
        approved_by: str,
        reason: str,
        approved_at: Optional[float] = None,
    ) -> InterventionAuthorization:
        """Create an authorization bound to one visible approval proposal."""
        if proposal_id not in self.approval_required_proposal_ids:
            raise RecoveryIntegrationError(
                "only approval-required proposals can be explicitly authorized"
            )
        return InterventionAuthorization.approve(
            self.proposal(proposal_id),
            approved_by=approved_by,
            reason=reason,
            approved_at=approved_at,
        )

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": RECOVERY_PREPARATION_SCHEMA_NAME,
                "version": RECOVERY_PREPARATION_SCHEMA_VERSION,
            },
            "contract": self.contract.to_dict(),
            "contract_digest": contract_digest(self.contract),
            "capability_manifest": self.capability_manifest.to_dict(),
            "policy_plan": self.policy_plan.to_dict(),
            "capsule_digest": self.capsule_digest,
            "source_environment": self.source_environment,
            "entrypoint_validation": self.entrypoint_validation,
            "compute_started": False,
            "invariants": {
                "deterministic_policy": True,
                "authorization_is_proposal_specific": True,
                "source_capsule_is_sealed": True,
            },
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "RecoveryPreparation":
        _validate_schema(
            payload,
            RECOVERY_PREPARATION_SCHEMA_NAME,
            RECOVERY_PREPARATION_SCHEMA_VERSION,
            "recovery preparation",
        )
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "contract",
                "contract_digest",
                "capability_manifest",
                "policy_plan",
                "capsule_digest",
                "source_environment",
                "entrypoint_validation",
                "compute_started",
                "invariants",
            },
            "recovery preparation",
        )
        if payload.get("compute_started") is not False:
            raise RecoveryIntegrationError(
                "recovery preparation cannot claim compute execution"
            )
        if payload.get("invariants") != {
            "deterministic_policy": True,
            "authorization_is_proposal_specific": True,
            "source_capsule_is_sealed": True,
        }:
            raise RecoveryIntegrationError(
                "recovery preparation invariants cannot be changed"
            )
        try:
            contract = RecoveryContract.from_dict(payload["contract"])
            preparation = cls(
                contract=contract,
                capability_manifest=CapabilityManifest.from_dict(
                    payload["capability_manifest"]
                ),
                policy_plan=OOMPolicyPlan.from_dict(payload["policy_plan"]),
                capsule_digest=payload["capsule_digest"],
                source_environment=payload["source_environment"],
                entrypoint_validation=payload["entrypoint_validation"],
            )
            encoded_contract_digest = payload["contract_digest"]
        except KeyError as exc:
            raise RecoveryIntegrationError(
                "recovery preparation is missing a required field"
            ) from exc
        except ValueError as exc:
            if isinstance(exc, RecoveryIntegrationError):
                raise
            raise RecoveryIntegrationError(str(exc)) from exc
        if encoded_contract_digest != contract_digest(contract):
            raise RecoveryIntegrationError("contract_digest is inconsistent")
        return preparation

    @classmethod
    def from_json(cls, encoded: str) -> "RecoveryPreparation":
        return cls.from_dict(_load_json(encoded, "recovery preparation"))


@dataclass(frozen=True)
class RecoveryResult:
    """Complete public result for one prepared and executed recovery."""

    preparation: RecoveryPreparation
    campaign: CampaignResult
    executed_proposal_ids: Tuple[str, ...]
    skipped_proposals: Tuple[ProposalSkip, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.preparation, RecoveryPreparation):
            raise RecoveryIntegrationError(
                "preparation must be a RecoveryPreparation"
            )
        if not isinstance(self.campaign, CampaignResult):
            raise RecoveryIntegrationError("campaign must be a CampaignResult")
        if self.campaign.contract_digest != contract_digest(
            self.preparation.contract
        ):
            raise RecoveryIntegrationError(
                "campaign result belongs to another recovery contract"
            )
        executed = tuple(self.executed_proposal_ids)
        for proposal_id in executed:
            _validate_id(proposal_id, "executed proposal_id")
        if len(executed) != len(set(executed)):
            raise RecoveryIntegrationError(
                "executed_proposal_ids must contain unique ids"
            )
        if executed != self.campaign.planned_candidate_ids:
            raise RecoveryIntegrationError(
                "executed proposal ids must match campaign candidates"
            )
        object.__setattr__(self, "executed_proposal_ids", executed)
        skipped = tuple(self.skipped_proposals)
        if any(not isinstance(item, ProposalSkip) for item in skipped):
            raise RecoveryIntegrationError(
                "skipped_proposals must contain ProposalSkip values"
            )
        skipped_ids = tuple(item.proposal_id for item in skipped)
        if len(skipped_ids) != len(set(skipped_ids)):
            raise RecoveryIntegrationError(
                "each proposal may be skipped at most once"
            )
        planned_ids = tuple(
            item.proposal_id for item in self.preparation.policy_plan.proposals
        )
        if set(executed) & set(skipped_ids):
            raise RecoveryIntegrationError(
                "a proposal cannot be both executed and skipped"
            )
        if set(executed) | set(skipped_ids) != set(planned_ids):
            raise RecoveryIntegrationError(
                "every policy proposal must be executed or explicitly skipped"
            )
        object.__setattr__(self, "skipped_proposals", skipped)

    @property
    def verified(self) -> bool:
        return self.campaign.verified

    @property
    def campaign_id(self) -> str:
        return self.campaign.campaign_id

    @property
    def verified_candidate_id(self) -> Optional[str]:
        return self.campaign.verified_candidate_id

    @property
    def verified_run_ids(self) -> Tuple[str, ...]:
        return self.campaign.verified_run_ids

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": RECOVERY_RESULT_SCHEMA_NAME,
                "version": RECOVERY_RESULT_SCHEMA_VERSION,
            },
            "preparation": self.preparation.to_dict(),
            "preparation_digest": preparation_digest(self.preparation),
            "campaign": self.campaign.to_dict(),
            "executed_proposal_ids": list(self.executed_proposal_ids),
            "skipped_proposals": [
                item.to_dict() for item in self.skipped_proposals
            ],
            "verified": self.verified,
            "invariants": {
                "subprocess_trials_only": True,
                "ranking_is_not_a_verdict": True,
                "verifier_is_only_recovery_authority": True,
                "no_unrecorded_retry": True,
            },
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict) -> "RecoveryResult":
        _validate_schema(
            payload,
            RECOVERY_RESULT_SCHEMA_NAME,
            RECOVERY_RESULT_SCHEMA_VERSION,
            "recovery result",
        )
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "preparation",
                "preparation_digest",
                "campaign",
                "executed_proposal_ids",
                "skipped_proposals",
                "verified",
                "invariants",
            },
            "recovery result",
        )
        if payload.get("invariants") != {
            "subprocess_trials_only": True,
            "ranking_is_not_a_verdict": True,
            "verifier_is_only_recovery_authority": True,
            "no_unrecorded_retry": True,
        }:
            raise RecoveryIntegrationError(
                "recovery result invariants cannot be changed"
            )
        try:
            preparation = RecoveryPreparation.from_dict(payload["preparation"])
            executed = payload["executed_proposal_ids"]
            skipped = payload["skipped_proposals"]
            if not isinstance(executed, list) or not isinstance(skipped, list):
                raise RecoveryIntegrationError(
                    "recovery result collections must be arrays"
                )
            result = cls(
                preparation=preparation,
                campaign=CampaignResult.from_dict(payload["campaign"]),
                executed_proposal_ids=tuple(executed),
                skipped_proposals=tuple(
                    ProposalSkip.from_dict(item) for item in skipped
                ),
            )
            encoded_preparation_digest = payload["preparation_digest"]
            encoded_verified = payload["verified"]
        except KeyError as exc:
            raise RecoveryIntegrationError(
                "recovery result is missing a required field"
            ) from exc
        except ValueError as exc:
            if isinstance(exc, RecoveryIntegrationError):
                raise
            raise RecoveryIntegrationError(str(exc)) from exc
        if encoded_preparation_digest != preparation_digest(preparation):
            raise RecoveryIntegrationError(
                "preparation_digest is inconsistent"
            )
        if not isinstance(encoded_verified, bool) or encoded_verified != result.verified:
            raise RecoveryIntegrationError("verified flag is inconsistent")
        return result

    @classmethod
    def from_json(cls, encoded: str) -> "RecoveryResult":
        return cls.from_dict(_load_json(encoded, "recovery result"))


class IsolatedTrialExecutor:
    """Campaign adapter around the parent-side subprocess runner."""

    def __init__(
        self,
        *,
        storage: Storage,
        project_root: Union[str, Path],
        trials_root: Optional[Union[str, Path]] = None,
        progress_metric: str = DEFAULT_PROGRESS_METRIC,
        python_executable: Optional[Union[str, Path]] = None,
        termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    ) -> None:
        if not isinstance(storage, Storage):
            raise RecoveryIntegrationError("storage must be a Storage")
        self.storage = storage
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise RecoveryIntegrationError(
                "project_root is not an existing directory: {}".format(
                    self.project_root
                )
            )
        self.trials_root = trials_root
        self.progress_metric = _metric_name(progress_metric, "progress_metric")
        self.python_executable = python_executable
        self.termination_grace_seconds = _nonnegative_finite(
            termination_grace_seconds,
            "termination_grace_seconds",
        )

    def __call__(
        self,
        request: TrialRequest,
        timeout_seconds: float,
    ) -> ExecutedTrial:
        execution = run_trial(
            request,
            project_root=self.project_root,
            storage_root=self.storage.root,
            trials_root=self.trials_root,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=self.termination_grace_seconds,
            python_executable=self.python_executable,
        )
        return ExecutedTrial(
            execution=execution,
            observation=_observe_trial(
                self.storage,
                request,
                execution,
                progress_metric=self.progress_metric,
            ),
        )


def prepare_oom_recovery(
    failed_run_id: str,
    entrypoint: Union[str, TrainingEntrypoint],
    verification: VerificationRequirements,
    *,
    budget: Optional[RecoveryBudget] = None,
    permissions: Optional[InterventionPermissions] = None,
    storage: Optional[Storage] = None,
    project_root: Union[str, Path] = ".",
    capability_declarations: Optional[DeclarationInput] = None,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    include_approval_required: bool = True,
) -> RecoveryPreparation:
    """Prepare and validate an OOM plan without launching any trial.

    The entrypoint is imported and signature-validated in the caller process.
    It must accept ``max_steps`` so a probe cannot silently become a full run.
    """
    storage = Storage() if storage is None else storage
    if not isinstance(storage, Storage):
        raise RecoveryIntegrationError("storage must be a Storage")
    _validate_id(failed_run_id, "failed_run_id")
    spec = _entrypoint(entrypoint)
    if not isinstance(verification, VerificationRequirements):
        raise RecoveryIntegrationError(
            "verification must be VerificationRequirements"
        )
    normalized_budget = RecoveryBudget() if budget is None else budget
    if not isinstance(normalized_budget, RecoveryBudget):
        raise RecoveryIntegrationError("budget must be a RecoveryBudget")
    normalized_permissions = (
        InterventionPermissions() if permissions is None else permissions
    )
    if not isinstance(normalized_permissions, InterventionPermissions):
        raise RecoveryIntegrationError(
            "permissions must be InterventionPermissions"
        )

    row, capsule = _source_oom(storage, failed_run_id)
    project = row["project"]
    config = (capsule.get("evidence") or {}).get("config")
    if not isinstance(config, dict) or not config:
        raise RecoveryIntegrationError(
            "the OOM capsule does not contain a usable source configuration"
        )
    source_environment = _captured_source_environment(capsule)
    manifest = discover_capabilities(
        config,
        declarations=capability_declarations,
        environment=source_environment,
    )
    policy_plan = plan_oom_interventions(
        capsule,
        manifest,
        max_proposals=max_proposals,
        include_approval_required=include_approval_required,
    )
    validation = validate_entrypoint(
        spec,
        project_root=str(Path(project_root).expanduser().resolve()),
        require_max_steps=True,
    )
    contract = RecoveryContract(
        project=project,
        source_run_id=failed_run_id,
        entrypoint=spec,
        source_config=config,
        budget=normalized_budget,
        verification=verification,
        permissions=normalized_permissions,
    )
    return RecoveryPreparation(
        contract=contract,
        capability_manifest=manifest,
        policy_plan=policy_plan,
        capsule_digest=_digest(capsule),
        source_environment=source_environment,
        entrypoint_validation=validation.to_dict(),
    )


def run_prepared_recovery(
    preparation: RecoveryPreparation,
    *,
    ranking_policy: Optional[RankingPolicy] = None,
    authorizations: Optional[Mapping[str, InterventionAuthorization]] = None,
    storage: Optional[Storage] = None,
    project_root: Union[str, Path] = ".",
    trials_root: Optional[Union[str, Path]] = None,
    progress_metric: str = DEFAULT_PROGRESS_METRIC,
    python_executable: Optional[Union[str, Path]] = None,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    campaign_id: Optional[str] = None,
    print_summary: bool = True,
) -> RecoveryResult:
    """Execute a sealed preparation through isolated trials and verification."""
    if not isinstance(preparation, RecoveryPreparation):
        raise RecoveryIntegrationError(
            "preparation must be a RecoveryPreparation"
        )
    storage = Storage() if storage is None else storage
    if not isinstance(storage, Storage):
        raise RecoveryIntegrationError("storage must be a Storage")
    normalized_authorizations = _authorizations(authorizations)
    plan_ids = {
        proposal.proposal_id for proposal in preparation.policy_plan.proposals
    }
    unknown_authorizations = sorted(set(normalized_authorizations) - plan_ids)
    if unknown_authorizations:
        raise RecoveryIntegrationError(
            "authorizations reference unknown proposal ids: {}".format(
                unknown_authorizations
            )
        )
    automatic_ids = set(preparation.automatic_proposal_ids)
    unexpected_automatic = sorted(set(normalized_authorizations) & automatic_ids)
    if unexpected_automatic:
        raise RecoveryIntegrationError(
            "automatic proposals must not receive explicit authorization: {}".format(
                unexpected_automatic
            )
        )

    _, current_capsule = _source_oom(
        storage,
        preparation.contract.source_run_id,
    )
    if _digest(current_capsule) != preparation.capsule_digest:
        raise RecoveryIntegrationError(
            "the source OOM capsule changed after recovery preparation"
        )
    validate_entrypoint(
        preparation.contract.entrypoint,
        project_root=str(Path(project_root).expanduser().resolve()),
        require_max_steps=True,
    )
    policy = (
        RankingPolicy(
            preparation.contract.verification.metric_guards[0].name
        )
        if ranking_policy is None
        else ranking_policy
    )
    if not isinstance(policy, RankingPolicy):
        raise RecoveryIntegrationError(
            "ranking_policy must be a RankingPolicy"
        )
    policy.validate_against(preparation.contract)

    candidates = []
    skipped = []
    source_config = preparation.contract.source_config
    source_environment = preparation.source_environment
    approval_ids = set(preparation.approval_required_proposal_ids)
    for proposal in preparation.policy_plan.proposals:
        authorization = normalized_authorizations.get(proposal.proposal_id)
        if proposal.proposal_id in approval_ids and authorization is None:
            skipped.append(
                ProposalSkip(
                    proposal.proposal_id,
                    proposal.policy_rule,
                    "authorization_missing",
                    "The proposal requires explicit authorization bound to its digest.",
                )
            )
            continue
        resolved = resolve_intervention(
            proposal,
            preparation.capability_manifest,
            source_config,
            source_environment=source_environment,
        )
        try:
            validate_intervention_scope(preparation.contract, resolved)
        except ValueError as exc:
            if authorization is not None:
                raise RecoveryIntegrationError(
                    "authorized proposal {!r} exceeds contract scope: {}".format(
                        proposal.proposal_id,
                        exc,
                    )
                ) from exc
            skipped.append(
                ProposalSkip(
                    proposal.proposal_id,
                    proposal.policy_rule,
                    "contract_scope_excluded",
                    str(exc),
                )
            )
            continue
        application = materialize_intervention(
            resolved,
            preparation.capability_manifest,
            source_config,
            source_environment=source_environment,
            authorization=authorization,
        )
        candidates.append(CampaignCandidate(resolved, application))

    normalized_campaign_id = campaign_id or "recovery-{}".format(
        uuid.uuid4().hex[:12]
    )
    _validate_id(normalized_campaign_id, "campaign_id")
    started_at = time.time()
    storage.create_recovery_campaign(
        normalized_campaign_id,
        preparation.contract.project,
        preparation.contract.source_run_id,
        preparation.contract.to_dict(),
        started_at,
    )
    executor = IsolatedTrialExecutor(
        storage=storage,
        project_root=project_root,
        trials_root=trials_root,
        progress_metric=progress_metric,
        python_executable=python_executable,
        termination_grace_seconds=termination_grace_seconds,
    )
    try:
        campaign = run_campaign(
            preparation.contract,
            campaign_id=normalized_campaign_id,
            candidates=tuple(candidates),
            ranking_policy=policy,
            executor=executor,
        )
        result = RecoveryResult(
            preparation=preparation,
            campaign=campaign,
            executed_proposal_ids=tuple(
                candidate.candidate_id for candidate in candidates
            ),
            skipped_proposals=tuple(skipped),
        )
        _persist_recovery_result(
            storage,
            result,
            candidates=tuple(candidates),
            ended_at=time.time(),
        )
    except Exception as exc:
        _finish_integration_error(
            storage,
            normalized_campaign_id,
            preparation,
            exc,
        )
        raise

    if result.verified:
        storage.set_run_resolved(
            preparation.contract.source_run_id,
            True,
            "Verified by recovery campaign {} using candidate {}.".format(
                result.campaign_id,
                result.verified_candidate_id,
            ),
        )
    if print_summary:
        print_recovery_summary(result)
    return result


def recover_from_oom(
    failed_run_id: str,
    entrypoint: Union[str, TrainingEntrypoint],
    verification: VerificationRequirements,
    *,
    budget: Optional[RecoveryBudget] = None,
    permissions: Optional[InterventionPermissions] = None,
    ranking_policy: Optional[RankingPolicy] = None,
    authorizations: Optional[Mapping[str, InterventionAuthorization]] = None,
    storage: Optional[Storage] = None,
    project_root: Union[str, Path] = ".",
    trials_root: Optional[Union[str, Path]] = None,
    progress_metric: str = DEFAULT_PROGRESS_METRIC,
    capability_declarations: Optional[DeclarationInput] = None,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    include_approval_required: bool = True,
    python_executable: Optional[Union[str, Path]] = None,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    campaign_id: Optional[str] = None,
    print_summary: bool = True,
) -> RecoveryResult:
    """Prepare and run one deterministic isolated CUDA OOM campaign.

    For approval-required interventions, use ``prepare_oom_recovery`` first,
    inspect its plan, call ``preparation.authorize(...)`` for chosen proposal
    ids, then pass those authorizations to ``run_prepared_recovery``.

    The training entrypoint must return all guarded metrics and an integral
    ``steps_completed`` metric (or the configured ``progress_metric``) for full
    and confirmation runs.
    """
    shared_storage = Storage() if storage is None else storage
    preparation = prepare_oom_recovery(
        failed_run_id,
        entrypoint,
        verification,
        budget=budget,
        permissions=permissions,
        storage=shared_storage,
        project_root=project_root,
        capability_declarations=capability_declarations,
        max_proposals=max_proposals,
        include_approval_required=include_approval_required,
    )
    return run_prepared_recovery(
        preparation,
        ranking_policy=ranking_policy,
        authorizations=authorizations,
        storage=shared_storage,
        project_root=project_root,
        trials_root=trials_root,
        progress_metric=progress_metric,
        python_executable=python_executable,
        termination_grace_seconds=termination_grace_seconds,
        campaign_id=campaign_id,
        print_summary=print_summary,
    )


def preparation_digest(preparation: RecoveryPreparation) -> str:
    if not isinstance(preparation, RecoveryPreparation):
        raise RecoveryIntegrationError(
            "preparation must be a RecoveryPreparation"
        )
    return hashlib.sha256(preparation.to_json().encode("utf-8")).hexdigest()


def recovery_result_digest(result: RecoveryResult) -> str:
    if not isinstance(result, RecoveryResult):
        raise RecoveryIntegrationError("result must be a RecoveryResult")
    return hashlib.sha256(result.to_json().encode("utf-8")).hexdigest()


def print_recovery_summary(result: RecoveryResult) -> None:
    """Print claims that exactly match the verifier-backed artifact."""
    if not isinstance(result, RecoveryResult):
        raise RecoveryIntegrationError("result must be a RecoveryResult")
    campaign = result.campaign
    print("\nWatcherML OOM recovery campaign: {}".format(campaign.campaign_id))
    print("Status: {} ({})".format(campaign.status, campaign.stopped_reason))
    print(
        "Trials: {}/{}  [probe={}, full={}, confirmation={}]".format(
            campaign.usage.attempted_trials,
            result.preparation.contract.budget.max_trials,
            campaign.usage.probe_trials,
            campaign.usage.full_trials,
            campaign.usage.confirmation_trials,
        )
    )
    print(
        "Policy proposals: {} executed, {} skipped".format(
            len(result.executed_proposal_ids),
            len(result.skipped_proposals),
        )
    )
    if result.verified:
        print("Verified recovery: {}".format(result.verified_candidate_id))
        print(
            "Independent confirmation runs: {}".format(
                ", ".join(result.verified_run_ids)
            )
        )
    elif campaign.ranking and campaign.ranking.confirmation_order:
        print(
            "No verified recovery. Provisional confirmation order was: {}".format(
                ", ".join(campaign.ranking.confirmation_order)
            )
        )
    else:
        print("No verified recovery was produced.")
    print("Inspect: watcher recovery {}".format(campaign.campaign_id))


def _observe_trial(
    storage: Storage,
    request: TrialRequest,
    execution: TrialExecution,
    *,
    progress_metric: str,
) -> RunObservation:
    row = storage.get_run(execution.run_id)
    metrics = execution.result.metrics if execution.result is not None else {}
    progress = None
    if execution.status == "success" and request.phase == "probe":
        progress = request.max_steps
    elif progress_metric in metrics:
        progress = _integral_metric(metrics[progress_metric], progress_metric)
    if progress is None and row is not None:
        logged_steps = [
            item["step"]
            for item in storage.get_metrics(execution.run_id)
            if item["step"] is not None
        ]
        if logged_steps:
            progress = max(logged_steps)

    resource = _row_json(row, "resource_json")
    peak_mib = resource.get("vram_used_mib_peak")
    peak_vram_bytes = None
    if _finite_nonnegative(peak_mib) is not None:
        peak_vram_bytes = int(round(float(peak_mib) * 1024 * 1024))

    gpu = _row_json(row, "gpu_json")
    if gpu.get("available") is True:
        gpu_seconds = execution.duration_seconds
    elif gpu.get("available") is False:
        gpu_seconds = 0.0
    else:
        gpu_seconds = None

    config = _row_json(row, "config_json") or request.config
    identity = WorkloadIdentity(
        dataset_fingerprint=(
            row["dataset_fingerprint"] if row is not None else None
        ) or _nested_value(config, "dataset.fingerprint"),
        environment_fingerprint=_row_json(row, "env_json").get("fingerprint"),
        git_commit=_row_json(row, "git_json").get("commit"),
        model_identifier=_model_identifier(config),
    )
    return RunObservation(
        progress_steps=progress,
        peak_vram_bytes=peak_vram_bytes,
        workload_identity=identity,
        gpu_seconds=gpu_seconds,
    )


def _persist_recovery_result(
    storage: Storage,
    result: RecoveryResult,
    *,
    candidates: Tuple[CampaignCandidate, ...],
    ended_at: float,
) -> None:
    candidates_by_id = {item.candidate_id: item for item in candidates}
    verified_runs = set(result.verified_run_ids)
    created_at = time.time()
    for index, trial in enumerate(result.campaign.trials):
        candidate = candidates_by_id[trial.candidate_id]
        patch = dict(candidate.resolved.config_patch)
        if candidate.resolved.environment_patch:
            patch["__environment__"] = candidate.resolved.environment_patch
        storage.save_recovery_trial(
            result.campaign_id,
            trial.run_id,
            trial.phase,
            {
                "proposal_id": candidate.candidate_id,
                "policy_rule": candidate.resolved.proposal.policy_rule,
                "evidence_refs": list(
                    candidate.resolved.proposal.evidence_refs
                ),
            },
            patch,
            candidate.resolved.proposal.rationale,
            None,
            trial.failure_class or trial.status,
            None,
            trial.run_id in verified_runs,
            created_at + (index * 0.000001),
        )

    report = result.to_dict()
    storage.finish_recovery_campaign(
        result.campaign_id,
        ended_at,
        result.campaign.stopped_reason,
        None,
        report,
    )
    artifact_path = storage.artifact_path(
        result.campaign_id,
        "recovery-result.json",
    )
    atomic_write_json(artifact_path, report)
    payload = Path(artifact_path).read_bytes()
    storage.log_artifact(
        result.campaign_id,
        artifact_path,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


def _finish_integration_error(
    storage: Storage,
    campaign_id: str,
    preparation: RecoveryPreparation,
    error: Exception,
) -> None:
    report = {
        "schema": {
            "name": RECOVERY_RESULT_SCHEMA_NAME,
            "version": RECOVERY_RESULT_SCHEMA_VERSION,
        },
        "campaign_id": campaign_id,
        "preparation_digest": preparation_digest(preparation),
        "status": "integration_error",
        "verified": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    try:
        storage.finish_recovery_campaign(
            campaign_id,
            time.time(),
            "integration_error",
            None,
            report,
        )
    except Exception:
        pass


def _source_oom(storage: Storage, run_id: str):
    row = storage.get_run(run_id)
    if row is None:
        raise RecoveryIntegrationError("run {!r} was not found".format(run_id))
    capsule = storage.get_failure_capsule(run_id)
    if capsule is None:
        raise RecoveryIntegrationError(
            "run {!r} has no failure capsule".format(run_id)
        )
    failure_class = capsule.get("failure_class") or (
        capsule.get("failure") or {}
    ).get("class")
    if failure_class != "cuda_out_of_memory":
        raise RecoveryIntegrationError(
            "run {!r} failed as {!r}, not cuda_out_of_memory".format(
                run_id,
                failure_class,
            )
        )
    if capsule.get("run_id") != run_id:
        raise RecoveryIntegrationError(
            "failure capsule run_id does not match the requested run"
        )
    if capsule.get("project") not in {None, row["project"]}:
        raise RecoveryIntegrationError(
            "failure capsule project does not match the recorded run"
        )
    return row, capsule


def _captured_source_environment(capsule: dict) -> Dict[str, str]:
    framework = (capsule.get("evidence") or {}).get("framework") or {}
    value = framework.get("allocator_config")
    if value is None:
        return {}
    if not isinstance(value, str):
        raise RecoveryIntegrationError(
            "captured allocator_config must be a string"
        )
    return {"PYTORCH_CUDA_ALLOC_CONF": value}


def _authorizations(value) -> Dict[str, InterventionAuthorization]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RecoveryIntegrationError("authorizations must be a mapping")
    normalized = {}
    for proposal_id, authorization in value.items():
        _validate_id(proposal_id, "authorization proposal_id")
        if not isinstance(authorization, InterventionAuthorization):
            raise RecoveryIntegrationError(
                "authorization values must be InterventionAuthorization objects"
            )
        if authorization.proposal_id != proposal_id:
            raise RecoveryIntegrationError(
                "authorization mapping key does not match proposal_id"
            )
        normalized[proposal_id] = authorization
    return normalized


def _entrypoint(value) -> TrainingEntrypoint:
    if isinstance(value, TrainingEntrypoint):
        return value
    if isinstance(value, str):
        return TrainingEntrypoint(value)
    raise RecoveryIntegrationError(
        "entrypoint must be TrainingEntrypoint or 'module:function' string"
    )


def _entrypoint_validation(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RecoveryIntegrationError("entrypoint_validation must be an object")
    allowed = {
        "target",
        "signature",
        "supports_max_steps",
        "project_root",
        "working_directory",
    }
    _reject_unknown_fields(payload, allowed, "entrypoint validation")
    if set(payload) != allowed:
        raise RecoveryIntegrationError(
            "entrypoint_validation is missing required fields"
        )
    for name in ("target", "signature", "project_root", "working_directory"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise RecoveryIntegrationError(
                "entrypoint validation {} must be non-empty".format(name)
            )
    if payload["supports_max_steps"] is not True:
        raise RecoveryIntegrationError(
            "recovery entrypoint must support bounded max_steps probes"
        )
    return dict(payload)


def _row_json(row, column: str) -> dict:
    if row is None:
        return {}
    try:
        value = row[column]
    except (IndexError, KeyError):
        return {}
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _integral_metric(value, name: str) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RecoveryIntegrationError(
            "progress metric {!r} must be an integer".format(name)
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or not normalized.is_integer():
        raise RecoveryIntegrationError(
            "progress metric {!r} must be a finite non-negative integer".format(
                name
            )
        )
    return int(normalized)


def _model_identifier(config: dict) -> Optional[str]:
    paths = (
        "model.name",
        "model.name_or_path",
        "model.model_name_or_path",
        "model_name_or_path",
        "model_name",
        "model_id",
        "model_identifier",
    )
    for path in paths:
        value = _nested_value(config, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    model = config.get("model") if isinstance(config, dict) else None
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _nested_value(payload: dict, path: str):
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _metric_name(value, name: str) -> str:
    if not isinstance(value, str) or not _METRIC_PATTERN.fullmatch(value):
        raise RecoveryIntegrationError("{} is not a valid metric name".format(name))
    return value


def _finite_nonnegative(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) and normalized >= 0 else None


def _nonnegative_finite(value, name: str) -> float:
    normalized = _finite_nonnegative(value)
    if normalized is None:
        raise RecoveryIntegrationError(
            "{} must be a finite non-negative number".format(name)
        )
    return normalized


def _string_mapping(value, name: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise RecoveryIntegrationError("{} must be a string mapping".format(name))
    normalized = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise RecoveryIntegrationError(
                "{} must contain string keys and values".format(name)
            )
        normalized[key] = item
    return normalized


def _validate_id(value, name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise RecoveryIntegrationError("{} is invalid".format(name))


def _validate_digest(value, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise RecoveryIntegrationError(
            "{} must be a lowercase SHA-256 digest".format(name)
        )


def _digest(payload) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _stable_json(payload) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecoveryIntegrationError("value is not stable JSON") from exc


def _load_json(encoded: str, artifact: str) -> dict:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RecoveryIntegrationError("invalid {} JSON".format(artifact)) from exc
    if not isinstance(payload, dict):
        raise RecoveryIntegrationError("{} must be an object".format(artifact))
    return payload


def _validate_schema(payload: dict, name: str, version: str, artifact: str) -> None:
    if not isinstance(payload, dict):
        raise RecoveryIntegrationError("{} must be an object".format(artifact))
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        raise RecoveryIntegrationError("{} schema must be an object".format(artifact))
    _reject_unknown_fields(schema, {"name", "version"}, "schema")
    if schema.get("name") != name or schema.get("version") != version:
        raise RecoveryIntegrationError("{} schema is unsupported".format(artifact))


def _reject_unknown_fields(payload: dict, allowed: set, artifact: str) -> None:
    if not isinstance(payload, dict):
        raise RecoveryIntegrationError("{} must be an object".format(artifact))
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RecoveryIntegrationError(
            "{} contains unknown fields: {}".format(artifact, unknown)
        )