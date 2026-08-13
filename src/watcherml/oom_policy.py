"""Deterministic, evidence-gated CUDA OOM intervention policy.

This module converts one versioned CUDA OOM capsule plus one capability
manifest into an ordered plan of typed ``InterventionProposal`` objects.

The policy is deliberately not an agent and performs no trial execution,
candidate scoring, search, or recovery verification.  It makes no network
calls and imports no ML framework.  Its output says only which bounded trials
are justified by the captured evidence and which proposals require approval.

Ordering is part of the public V1 behavior:

1. halve micro-batch size while preserving effective batch size when possible;
2. enable gradient checkpointing;
3. combine those two interventions only after their single variants;
4. surface broader, evidence-gated and approval-required controls.

Code, dependencies, datasets, shell commands, arbitrary environment variables,
and unregistered configuration paths remain outside the policy vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .capabilities import (
    Capability,
    CapabilityError,
    CapabilityManifest,
    get_config_value,
)
from .capsule import build_evidence_index
from .capsule_schema import OOM_FAILURE_CLASS, validate_capsule
from .interventions import (
    InterventionChange,
    InterventionError,
    InterventionProposal,
    InterventionResolutionError,
    ResolvedIntervention,
    resolve_intervention,
)


OOM_POLICY_SCHEMA_NAME = "watcherml.oom-policy-plan"
OOM_POLICY_SCHEMA_VERSION = "1.0"
OOM_POLICY_RULE_VERSION = "1.0"

DEFAULT_MAX_PROPOSALS = 16
HARD_MAX_PROPOSALS = 32

POLICY_RULE_ORDER = (
    "halve_batch_preserve_effective_batch",
    "halve_micro_batch",
    "enable_gradient_checkpointing",
    "halve_batch_and_checkpoint",
    "allocator_fragmentation_mitigation",
    "disable_training_model_cache",
    "enable_memory_efficient_attention",
    "use_sdpa_attention",
    "use_lower_memory_precision",
    "halve_sequence_length",
    "enable_activation_offload",
    "enable_optimizer_state_offload",
    "use_8bit_optimizer_state",
    "enable_parameter_offload",
)

AUTOMATIC_POLICY_RULES = frozenset(
    {
        "halve_batch_preserve_effective_batch",
        "halve_micro_batch",
        "enable_gradient_checkpointing",
        "halve_batch_and_checkpoint",
    }
)

_POLICY_RULE_RANK = {
    rule: index for index, rule in enumerate(POLICY_RULE_ORDER)
}

POLICY_SKIP_CODES = frozenset(
    {
        "capability_unavailable",
        "already_enabled",
        "already_disabled",
        "already_minimal",
        "unsupported_baseline",
        "missing_evidence",
        "approval_filtered",
        "proposal_limit",
        "invalid_transition",
    }
)

_EVIDENCE_CATEGORY_IDS = {
    "config": "EV-1",
    "training_state": "EV-2",
    "runtime": "EV-3",
    "resource_state_at_failure": "EV-4",
    "gpu": "EV-5",
    "framework": "EV-6",
    "git": "EV-7",
    "environment": "EV-8",
    "dataset": "EV-9",
    "recent_metrics": "EV-10",
    "notebook_cells_executed": "EV-11",
}


class OOMPolicyError(ValueError):
    """Raised when policy inputs cannot safely produce an OOM plan."""


@dataclass(frozen=True)
class PolicySkip:
    """One policy rule that could not or must not become a proposal."""

    policy_rule: str
    capability_ids: Tuple[str, ...]
    code: str
    reason: str
    evidence_refs: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_rule, str) or not self.policy_rule:
            raise OOMPolicyError("policy_rule must be a non-empty string")
        if self.policy_rule not in _POLICY_RULE_RANK:
            raise OOMPolicyError("policy skip contains an unknown OOM rule")
        capability_ids = tuple(self.capability_ids)
        if not capability_ids or any(
            not isinstance(value, str) or not value
            for value in capability_ids
        ):
            raise OOMPolicyError(
                "capability_ids must contain non-empty strings"
            )
        if len(capability_ids) != len(set(capability_ids)):
            raise OOMPolicyError("capability_ids must be unique")
        object.__setattr__(self, "capability_ids", capability_ids)
        if self.code not in POLICY_SKIP_CODES:
            raise OOMPolicyError(
                "invalid policy skip code {!r}".format(self.code)
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise OOMPolicyError("policy skip reason must be non-empty")
        object.__setattr__(self, "reason", self.reason.strip())
        evidence_refs = tuple(self.evidence_refs)
        if len(evidence_refs) != len(set(evidence_refs)):
            raise OOMPolicyError("skip evidence references must be unique")
        if any(not isinstance(value, str) or not value for value in evidence_refs):
            raise OOMPolicyError(
                "skip evidence references must be non-empty strings"
            )
        object.__setattr__(self, "evidence_refs", evidence_refs)

    def to_dict(self) -> dict:
        return {
            "policy_rule": self.policy_rule,
            "capability_ids": list(self.capability_ids),
            "code": self.code,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PolicySkip":
        if not isinstance(payload, dict):
            raise OOMPolicyError("policy skip must be an object")
        allowed = {
            "policy_rule",
            "capability_ids",
            "code",
            "reason",
            "evidence_refs",
        }
        _reject_unknown_fields(payload, allowed, "policy skip")
        try:
            capability_ids = payload["capability_ids"]
            evidence_refs = payload["evidence_refs"]
            if not isinstance(capability_ids, list):
                raise OOMPolicyError("capability_ids must be an array")
            if not isinstance(evidence_refs, list):
                raise OOMPolicyError("evidence_refs must be an array")
            return cls(
                policy_rule=payload["policy_rule"],
                capability_ids=tuple(capability_ids),
                code=payload["code"],
                reason=payload["reason"],
                evidence_refs=tuple(evidence_refs),
            )
        except KeyError as exc:
            raise OOMPolicyError("policy skip is missing a field") from exc


@dataclass(frozen=True)
class OOMPolicyPlan:
    """Versioned result of deterministic policy evaluation."""

    run_id: str
    proposals: Tuple[InterventionProposal, ...]
    automatic_proposal_ids: Tuple[str, ...]
    approval_required_proposal_ids: Tuple[str, ...]
    skipped: Tuple[PolicySkip, ...] = field(default_factory=tuple)
    failure_class: str = OOM_FAILURE_CLASS
    policy_rule_version: str = OOM_POLICY_RULE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise OOMPolicyError("run_id must be a non-empty string")
        if self.failure_class != OOM_FAILURE_CLASS:
            raise OOMPolicyError("OOM policy plan has an invalid failure class")
        if self.policy_rule_version != OOM_POLICY_RULE_VERSION:
            raise OOMPolicyError("OOM policy plan has an invalid rule version")

        proposals = tuple(self.proposals)
        if any(
            not isinstance(proposal, InterventionProposal)
            for proposal in proposals
        ):
            raise OOMPolicyError(
                "proposals must contain InterventionProposal values"
            )
        proposal_ids = tuple(proposal.proposal_id for proposal in proposals)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise OOMPolicyError("proposal ids must be unique")
        policy_rules = tuple(proposal.policy_rule for proposal in proposals)
        if len(policy_rules) != len(set(policy_rules)):
            raise OOMPolicyError("policy rules must be unique within a plan")
        if any(rule not in _POLICY_RULE_RANK for rule in policy_rules):
            raise OOMPolicyError("plan contains an unknown OOM policy rule")
        ranks = tuple(_POLICY_RULE_RANK[rule] for rule in policy_rules)
        if any(left >= right for left, right in zip(ranks, ranks[1:])):
            raise OOMPolicyError(
                "proposals do not follow the deterministic policy order"
            )
        if any(
            proposal.proposer != "deterministic_policy"
            for proposal in proposals
        ):
            raise OOMPolicyError(
                "OOM policy plans may contain deterministic proposals only"
            )
        for proposal in proposals:
            expected_id = _proposal_id(
                self.run_id,
                proposal.policy_rule,
                proposal.changes,
            )
            if proposal.proposal_id != expected_id:
                raise OOMPolicyError(
                    "proposal_id is not bound to its run, rule, and changes"
                )
        object.__setattr__(self, "proposals", proposals)

        automatic = tuple(self.automatic_proposal_ids)
        approval = tuple(self.approval_required_proposal_ids)
        if set(automatic) & set(approval):
            raise OOMPolicyError(
                "a proposal cannot be both automatic and approval-required"
            )
        if set(automatic) | set(approval) != set(proposal_ids):
            raise OOMPolicyError(
                "permission id lists must partition every proposal"
            )
        if len(automatic) != len(set(automatic)) or len(approval) != len(set(approval)):
            raise OOMPolicyError("permission proposal ids must be unique")
        proposal_by_id = {
            proposal.proposal_id: proposal for proposal in proposals
        }
        if any(
            proposal_by_id[proposal_id].policy_rule
            not in AUTOMATIC_POLICY_RULES
            for proposal_id in automatic
        ):
            raise OOMPolicyError(
                "an approval-only policy rule cannot be classified automatic"
            )
        automatic_set = set(automatic)
        approval_set = set(approval)
        expected_automatic_order = tuple(
            proposal.proposal_id
            for proposal in proposals
            if proposal.proposal_id in automatic_set
        )
        expected_approval_order = tuple(
            proposal.proposal_id
            for proposal in proposals
            if proposal.proposal_id in approval_set
        )
        if (
            automatic != expected_automatic_order
            or approval != expected_approval_order
        ):
            raise OOMPolicyError(
                "permission proposal ids must preserve policy order"
            )
        object.__setattr__(self, "automatic_proposal_ids", automatic)
        object.__setattr__(self, "approval_required_proposal_ids", approval)

        skipped = tuple(self.skipped)
        if any(not isinstance(item, PolicySkip) for item in skipped):
            raise OOMPolicyError("skipped must contain PolicySkip values")
        skipped_rules = tuple(item.policy_rule for item in skipped)
        if len(skipped_rules) != len(set(skipped_rules)):
            raise OOMPolicyError("a policy rule may be skipped at most once")
        if set(skipped_rules) & set(policy_rules):
            raise OOMPolicyError(
                "a policy rule cannot be both proposed and skipped"
            )
        skipped_ranks = tuple(
            _POLICY_RULE_RANK[rule] for rule in skipped_rules
        )
        if any(
            left >= right
            for left, right in zip(skipped_ranks, skipped_ranks[1:])
        ):
            raise OOMPolicyError(
                "skipped rules do not follow deterministic policy order"
            )
        object.__setattr__(self, "skipped", skipped)
        if len(proposals) > HARD_MAX_PROPOSALS:
            raise OOMPolicyError("plan exceeds the hard proposal cap")

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": OOM_POLICY_SCHEMA_NAME,
                "version": OOM_POLICY_SCHEMA_VERSION,
            },
            "run_id": self.run_id,
            "failure_class": self.failure_class,
            "policy_rule_version": self.policy_rule_version,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "automatic_proposal_ids": list(self.automatic_proposal_ids),
            "approval_required_proposal_ids": list(
                self.approval_required_proposal_ids
            ),
            "skipped": [item.to_dict() for item in self.skipped],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "OOMPolicyPlan":
        if not isinstance(payload, dict):
            raise OOMPolicyError("OOM policy plan must be an object")
        schema = payload.get("schema") or {}
        if schema.get("name") != OOM_POLICY_SCHEMA_NAME:
            raise OOMPolicyError(
                "schema.name must be {!r}".format(OOM_POLICY_SCHEMA_NAME)
            )
        if schema.get("version") != OOM_POLICY_SCHEMA_VERSION:
            raise OOMPolicyError(
                "schema.version must be {!r}".format(OOM_POLICY_SCHEMA_VERSION)
            )
        allowed = {
            "schema",
            "run_id",
            "failure_class",
            "policy_rule_version",
            "proposals",
            "automatic_proposal_ids",
            "approval_required_proposal_ids",
            "skipped",
        }
        _reject_unknown_fields(payload, allowed, "OOM policy plan")
        try:
            proposals = payload["proposals"]
            automatic = payload["automatic_proposal_ids"]
            approval = payload["approval_required_proposal_ids"]
            skipped = payload["skipped"]
            for name, value in (
                ("proposals", proposals),
                ("automatic_proposal_ids", automatic),
                ("approval_required_proposal_ids", approval),
                ("skipped", skipped),
            ):
                if not isinstance(value, list):
                    raise OOMPolicyError("{} must be an array".format(name))
            return cls(
                run_id=payload["run_id"],
                failure_class=payload["failure_class"],
                policy_rule_version=payload["policy_rule_version"],
                proposals=tuple(
                    InterventionProposal.from_dict(item) for item in proposals
                ),
                automatic_proposal_ids=tuple(automatic),
                approval_required_proposal_ids=tuple(approval),
                skipped=tuple(PolicySkip.from_dict(item) for item in skipped),
            )
        except KeyError as exc:
            raise OOMPolicyError("OOM policy plan is missing a field") from exc
        except InterventionError as exc:
            raise OOMPolicyError(str(exc)) from exc

    @classmethod
    def from_json(cls, encoded: str) -> "OOMPolicyPlan":
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise OOMPolicyError("invalid OOM policy plan JSON") from exc
        return cls.from_dict(payload)


@dataclass
class _Planner:
    capsule: dict
    manifest: CapabilityManifest
    config: dict
    source_environment: Dict[str, str]
    evidence_ids: Dict[str, str]
    max_proposals: int
    include_approval_required: bool
    proposals: List[InterventionProposal] = field(default_factory=list)
    automatic_ids: List[str] = field(default_factory=list)
    approval_ids: List[str] = field(default_factory=list)
    skipped: List[PolicySkip] = field(default_factory=list)

    def capability(self, capability_id: str) -> Optional[Capability]:
        return self.manifest.get(capability_id)

    def refs(self, *categories: str) -> Tuple[str, ...]:
        return tuple(
            self.evidence_ids[category]
            for category in categories
            if category in self.evidence_ids
        )

    def skip(
        self,
        policy_rule: str,
        capability_ids: Sequence[str],
        code: str,
        reason: str,
        evidence_refs: Sequence[str] = (),
    ) -> None:
        self.skipped.append(
            PolicySkip(
                policy_rule=policy_rule,
                capability_ids=tuple(capability_ids),
                code=code,
                reason=reason,
                evidence_refs=tuple(evidence_refs),
            )
        )

    def add(
        self,
        *,
        policy_rule: str,
        changes: Sequence[InterventionChange],
        rationale: str,
        expected_effect: str,
        evidence_refs: Sequence[str],
    ) -> Optional[ResolvedIntervention]:
        capability_ids = tuple(change.capability_id for change in changes)
        if not evidence_refs:
            self.skip(
                policy_rule,
                capability_ids,
                "missing_evidence",
                "The captured capsule has no evidence category required by "
                "this policy rule.",
            )
            return None
        proposal = InterventionProposal(
            proposal_id=_proposal_id(
                self.capsule["run_id"],
                policy_rule,
                changes,
            ),
            policy_rule=policy_rule,
            changes=tuple(changes),
            rationale=rationale,
            expected_effect=expected_effect,
            evidence_refs=tuple(evidence_refs),
        )
        try:
            resolved = resolve_intervention(
                proposal,
                self.manifest,
                self.config,
                source_environment=self.source_environment,
            )
        except InterventionResolutionError as exc:
            self.skip(
                policy_rule,
                capability_ids,
                "invalid_transition",
                str(exc),
                evidence_refs,
            )
            return None

        if resolved.approval_required and not self.include_approval_required:
            self.skip(
                policy_rule,
                capability_ids,
                "approval_filtered",
                "The proposal requires approval and gated proposals were "
                "excluded by the caller.",
                evidence_refs,
            )
            return None
        if len(self.proposals) >= self.max_proposals:
            self.skip(
                policy_rule,
                capability_ids,
                "proposal_limit",
                "The deterministic proposal limit was reached.",
                evidence_refs,
            )
            return None

        self.proposals.append(proposal)
        if resolved.approval_required:
            self.approval_ids.append(proposal.proposal_id)
        else:
            self.automatic_ids.append(proposal.proposal_id)
        return resolved


def plan_oom_interventions(
    capsule: dict,
    manifest: CapabilityManifest,
    *,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    include_approval_required: bool = True,
) -> OOMPolicyPlan:
    """Return an ordered deterministic intervention plan for one OOM capsule.

    Every proposal is resolved once against the captured source config before
    it enters the plan.  This catches stale manifests and invalid transitions
    before campaign code can spend GPU compute.  Approval-required proposals
    are clearly partitioned; planning them does not authorize execution.
    """
    normalized_capsule = _validate_policy_inputs(
        capsule,
        manifest,
        max_proposals,
        include_approval_required,
    )
    evidence = normalized_capsule["evidence"]
    config = evidence["config"]
    framework = evidence.get("framework") or {}
    source_environment = {}
    if framework.get("allocator_config") is not None:
        source_environment["PYTORCH_CUDA_ALLOC_CONF"] = str(
            framework["allocator_config"]
        )
    evidence_ids = _present_evidence_ids(normalized_capsule)
    _validate_manifest_baseline(
        manifest,
        config,
        source_environment,
    )

    planner = _Planner(
        capsule=normalized_capsule,
        manifest=manifest,
        config=config,
        source_environment=source_environment,
        evidence_ids=evidence_ids,
        max_proposals=max_proposals,
        include_approval_required=include_approval_required,
    )

    batch_changes = _plan_batch_reduction(planner)
    checkpoint_change = _plan_gradient_checkpointing(planner)
    _plan_combined_batch_checkpointing(
        planner,
        batch_changes,
        checkpoint_change,
    )

    # Broader controls are ordered from targeted/runtime changes toward more
    # intrusive semantic or offload changes. Each remains approval-gated by
    # its canonical capability definition.
    _plan_allocator_configuration(planner, framework)
    _plan_model_cache(planner)
    _plan_memory_efficient_attention(planner)
    _plan_attention_backend(planner)
    _plan_precision(planner, framework)
    _plan_sequence_length(planner)
    _plan_activation_offload(planner)
    _plan_optimizer_state_offload(planner)
    _plan_optimizer_bits(planner)
    _plan_parameter_offload(planner)

    return OOMPolicyPlan(
        run_id=normalized_capsule["run_id"],
        proposals=tuple(planner.proposals),
        automatic_proposal_ids=tuple(planner.automatic_ids),
        approval_required_proposal_ids=tuple(planner.approval_ids),
        skipped=tuple(planner.skipped),
    )


def _plan_batch_reduction(
    planner: _Planner,
) -> Optional[Tuple[InterventionChange, ...]]:
    rule = "halve_batch_preserve_effective_batch"
    batch = planner.capability("micro_batch_size")
    if batch is None:
        planner.skip(
            rule,
            ("micro_batch_size", "gradient_accumulation_steps"),
            "capability_unavailable",
            "No unambiguous micro-batch capability was discovered.",
        )
        return None
    if not isinstance(batch.current_value, int) or batch.current_value <= 1:
        planner.skip(
            rule,
            ("micro_batch_size",),
            "already_minimal",
            "Micro-batch size is already at its minimum supported value.",
            planner.refs("config", "training_state"),
        )
        return None

    new_batch = max(1, batch.current_value // 2)
    changes = [
        InterventionChange("micro_batch_size", "decrease", new_batch)
    ]
    accumulation = planner.capability("gradient_accumulation_steps")
    if (
        accumulation is not None
        and isinstance(accumulation.current_value, int)
        and accumulation.current_value >= 1
    ):
        original_effective = batch.current_value * accumulation.current_value
        new_accumulation = int(math.ceil(original_effective / new_batch))
        if new_accumulation > accumulation.current_value:
            changes.append(
                InterventionChange(
                    "gradient_accumulation_steps",
                    "increase",
                    new_accumulation,
                )
            )
        rationale = (
            "The deterministic CUDA OOM rule matched and the capsule records "
            "micro-batch size {} with {} accumulation step(s). Halve the "
            "per-device batch and raise accumulation to preserve at least the "
            "original effective batch size.".format(
                batch.current_value,
                accumulation.current_value,
            )
        )
        expected = (
            "Lower per-step activation memory while preserving effective "
            "batch size as closely as integer accumulation permits."
        )
    else:
        rule = "halve_micro_batch"
        rationale = (
            "The deterministic CUDA OOM rule matched and the capsule records "
            "micro-batch size {}. No unambiguous gradient-accumulation "
            "capability is available, so this proposal changes only the "
            "micro-batch size.".format(batch.current_value)
        )
        expected = (
            "Lower per-step activation memory; effective batch size may change "
            "because accumulation cannot be controlled."
        )

    resolved = planner.add(
        policy_rule=rule,
        changes=changes,
        rationale=rationale,
        expected_effect=expected,
        evidence_refs=planner.refs(
            "config",
            "training_state",
            "resource_state_at_failure",
            "framework",
        ),
    )
    return tuple(changes) if resolved is not None else None


def _plan_gradient_checkpointing(
    planner: _Planner,
) -> Optional[InterventionChange]:
    rule = "enable_gradient_checkpointing"
    capability = planner.capability("gradient_checkpointing")
    if capability is None:
        planner.skip(
            rule,
            ("gradient_checkpointing",),
            "capability_unavailable",
            "No unambiguous gradient-checkpointing capability was discovered.",
        )
        return None
    if capability.current_value is True:
        planner.skip(
            rule,
            ("gradient_checkpointing",),
            "already_enabled",
            "Gradient checkpointing is already enabled.",
            planner.refs("config", "training_state"),
        )
        return None
    change = InterventionChange(
        "gradient_checkpointing",
        "enable",
        True,
    )
    resolved = planner.add(
        policy_rule=rule,
        changes=(change,),
        rationale=(
            "The deterministic CUDA OOM rule matched and the captured "
            "configuration shows gradient checkpointing disabled."
        ),
        expected_effect=(
            "Trade recomputation time for lower retained activation memory."
        ),
        evidence_refs=planner.refs("config", "training_state", "framework"),
    )
    return change if resolved is not None else None


def _plan_combined_batch_checkpointing(
    planner: _Planner,
    batch_changes: Optional[Tuple[InterventionChange, ...]],
    checkpoint_change: Optional[InterventionChange],
) -> None:
    rule = "halve_batch_and_checkpoint"
    if not batch_changes or checkpoint_change is None:
        planner.skip(
            rule,
            (
                "micro_batch_size",
                "gradient_checkpointing",
            ),
            "capability_unavailable",
            "The combined proposal requires both earlier single-policy "
            "variants to be valid.",
        )
        return
    planner.add(
        policy_rule=rule,
        changes=tuple(batch_changes) + (checkpoint_change,),
        rationale=(
            "Combine the valid batch-reduction and checkpointing transitions "
            "only after presenting each lower-complexity intervention alone."
        ),
        expected_effect=(
            "Reduce per-step activation memory through both smaller "
            "micro-batches and activation recomputation."
        ),
        evidence_refs=planner.refs(
            "config",
            "training_state",
            "resource_state_at_failure",
            "framework",
        ),
    )


def _plan_allocator_configuration(planner: _Planner, framework: dict) -> None:
    rule = "allocator_fragmentation_mitigation"
    capability = planner.capability("allocator_configuration")
    if capability is None:
        planner.skip(
            rule,
            ("allocator_configuration",),
            "capability_unavailable",
            "No allowlisted allocator capability was captured.",
        )
        return
    fragmentation, detail = _allocator_fragmentation_signal(
        planner.capsule,
        framework,
    )
    if not fragmentation:
        planner.skip(
            rule,
            ("allocator_configuration",),
            "missing_evidence",
            detail,
            planner.refs("framework", "resource_state_at_failure"),
        )
        return
    current = str(capability.current_value)
    if "expandable_segments:true" in current.replace(" ", "").lower():
        planner.skip(
            rule,
            ("allocator_configuration",),
            "already_enabled",
            "Expandable allocator segments are already enabled.",
            planner.refs("framework"),
        )
        return
    after = (
        current.rstrip(",") + ",expandable_segments:True"
        if current
        else "expandable_segments:True"
    )
    planner.add(
        policy_rule=rule,
        changes=(
            InterventionChange(
                "allocator_configuration",
                "set",
                after,
            ),
        ),
        rationale=detail,
        expected_effect=(
            "Mitigate allocator fragmentation without changing model, data, "
            "or training semantics."
        ),
        evidence_refs=planner.refs("framework", "resource_state_at_failure"),
    )


def _plan_model_cache(planner: _Planner) -> None:
    _plan_boolean(
        planner,
        capability_id="model_cache",
        operation="disable",
        proposed_value=False,
        policy_rule="disable_training_model_cache",
        already_code="already_disabled",
        already_value=False,
        rationale=(
            "The source configuration enables a model cache during a training "
            "workload. Disable it for one bounded trial."
        ),
        expected_effect="Avoid retaining inference-style cache tensors.",
        categories=("config", "framework"),
    )


def _plan_memory_efficient_attention(planner: _Planner) -> None:
    _plan_boolean(
        planner,
        capability_id="memory_efficient_attention",
        operation="enable",
        proposed_value=True,
        policy_rule="enable_memory_efficient_attention",
        already_code="already_enabled",
        already_value=True,
        rationale=(
            "The workload explicitly exposes a memory-efficient attention "
            "capability and the captured configuration shows it disabled."
        ),
        expected_effect="Reduce materialized attention intermediates.",
        categories=("config", "framework"),
    )


def _plan_attention_backend(planner: _Planner) -> None:
    rule = "use_sdpa_attention"
    capability = planner.capability("attention_backend")
    if capability is None:
        planner.skip(
            rule,
            ("attention_backend",),
            "capability_unavailable",
            "No unambiguous attention-backend capability was discovered.",
        )
        return
    current = str(capability.current_value).lower()
    if current == "sdpa":
        planner.skip(
            rule,
            ("attention_backend",),
            "already_enabled",
            "The SDPA attention backend is already selected.",
            planner.refs("config", "framework"),
        )
        return
    if current != "eager":
        planner.skip(
            rule,
            ("attention_backend",),
            "unsupported_baseline",
            "The policy only changes an explicitly captured eager backend to "
            "SDPA; it will not guess between other implementations.",
            planner.refs("config", "framework"),
        )
        return
    planner.add(
        policy_rule=rule,
        changes=(InterventionChange("attention_backend", "set", "sdpa"),),
        rationale=(
            "The workload exposes attention-backend selection and the capsule "
            "records the eager implementation."
        ),
        expected_effect=(
            "Use the workload-supported SDPA path to reduce attention-memory "
            "intermediates where supported by the installed framework."
        ),
        evidence_refs=planner.refs("config", "framework"),
    )


def _plan_precision(planner: _Planner, framework: dict) -> None:
    rule = "use_lower_memory_precision"
    capability = planner.capability("precision")
    if capability is None:
        planner.skip(
            rule,
            ("precision",),
            "capability_unavailable",
            "No unambiguous precision capability was discovered.",
        )
        return
    current = str(capability.current_value).lower()
    if current in {"bf16", "bfloat16", "fp16", "float16"}:
        planner.skip(
            rule,
            ("precision",),
            "already_enabled",
            "A lower-memory floating-point precision is already selected.",
            planner.refs("config", "framework"),
        )
        return
    if current not in {"fp32", "float32", "tf32"}:
        planner.skip(
            rule,
            ("precision",),
            "unsupported_baseline",
            "The policy does not recognize the captured precision as a safe "
            "source for a deterministic transition.",
            planner.refs("config", "framework"),
        )
        return
    bf16_supported = bool(
        framework.get("bf16_supported")
        or framework.get("is_bf16_supported")
    )
    cuda_available = bool(framework.get("cuda_available"))
    if bf16_supported:
        after = "bf16"
        support_detail = "The framework evidence explicitly reports BF16 support."
    elif cuda_available:
        after = "fp16"
        support_detail = (
            "The framework evidence reports CUDA availability but does not "
            "explicitly establish BF16 support."
        )
    else:
        planner.skip(
            rule,
            ("precision",),
            "missing_evidence",
            "No captured CUDA or BF16-support evidence justifies selecting a "
            "lower-memory precision.",
            planner.refs("config", "framework", "gpu"),
        )
        return
    planner.add(
        policy_rule=rule,
        changes=(InterventionChange("precision", "set", after),),
        rationale=(
            "The source precision is {}. {} This numerical change remains "
            "approval-required.".format(current, support_detail)
        ),
        expected_effect=(
            "Reduce floating-point tensor memory, subject to metric and "
            "numerical-stability verification."
        ),
        evidence_refs=planner.refs("config", "framework", "gpu"),
    )


def _plan_sequence_length(planner: _Planner) -> None:
    rule = "halve_sequence_length"
    capability = planner.capability("sequence_length")
    if capability is None:
        planner.skip(
            rule,
            ("sequence_length",),
            "capability_unavailable",
            "No unambiguous sequence-length capability was discovered.",
        )
        return
    current = capability.current_value
    if not isinstance(current, int) or current <= 1:
        planner.skip(
            rule,
            ("sequence_length",),
            "already_minimal",
            "Sequence length is already at its minimum supported value.",
            planner.refs("config", "training_state"),
        )
        return
    after = max(1, current // 2)
    planner.add(
        policy_rule=rule,
        changes=(InterventionChange("sequence_length", "decrease", after),),
        rationale=(
            "The capsule records sequence length {}. Halving it is a bounded "
            "but semantic change, so the proposal requires approval.".format(
                current
            )
        ),
        expected_effect="Reduce attention and activation-memory growth.",
        evidence_refs=planner.refs(
            "config",
            "training_state",
            "resource_state_at_failure",
        ),
    )


def _plan_activation_offload(planner: _Planner) -> None:
    _plan_boolean(
        planner,
        capability_id="activation_offload",
        operation="enable",
        proposed_value=True,
        policy_rule="enable_activation_offload",
        already_code="already_enabled",
        already_value=True,
        rationale=(
            "The workload explicitly exposes activation offload and the "
            "captured configuration shows it disabled."
        ),
        expected_effect="Trade host transfer and RAM for lower activation VRAM.",
        categories=("config", "resource_state_at_failure", "framework"),
    )


def _plan_optimizer_state_offload(planner: _Planner) -> None:
    _plan_boolean(
        planner,
        capability_id="optimizer_state_offload",
        operation="enable",
        proposed_value=True,
        policy_rule="enable_optimizer_state_offload",
        already_code="already_enabled",
        already_value=True,
        rationale=(
            "The workload explicitly exposes optimizer-state offload and the "
            "captured configuration shows it disabled."
        ),
        expected_effect="Trade host transfer and RAM for lower optimizer VRAM.",
        categories=("config", "resource_state_at_failure", "framework"),
    )


def _plan_optimizer_bits(planner: _Planner) -> None:
    rule = "use_8bit_optimizer_state"
    capability = planner.capability("optimizer_bits")
    if capability is None:
        planner.skip(
            rule,
            ("optimizer_bits",),
            "capability_unavailable",
            "No optimizer-bit capability was discovered.",
        )
        return
    current = capability.current_value
    if not isinstance(current, int) or current <= 8:
        planner.skip(
            rule,
            ("optimizer_bits",),
            "already_minimal",
            "Optimizer state is already at the minimum allowlisted bit width.",
            planner.refs("config", "framework"),
        )
        return
    planner.add(
        policy_rule=rule,
        changes=(InterventionChange("optimizer_bits", "decrease", 8),),
        rationale=(
            "The workload explicitly exposes optimizer-state bit width and the "
            "capsule records {} bits.".format(current)
        ),
        expected_effect=(
            "Reduce persistent optimizer-state memory, subject to metric and "
            "optimizer-behavior verification."
        ),
        evidence_refs=planner.refs("config", "resource_state_at_failure"),
    )


def _plan_parameter_offload(planner: _Planner) -> None:
    _plan_boolean(
        planner,
        capability_id="parameter_offload",
        operation="enable",
        proposed_value=True,
        policy_rule="enable_parameter_offload",
        already_code="already_enabled",
        already_value=True,
        rationale=(
            "The workload explicitly exposes parameter offload and the "
            "captured configuration shows it disabled."
        ),
        expected_effect=(
            "Trade substantial host transfer and RAM for lower persistent "
            "parameter VRAM."
        ),
        categories=("config", "resource_state_at_failure", "framework"),
    )


def _plan_boolean(
    planner: _Planner,
    *,
    capability_id: str,
    operation: str,
    proposed_value: bool,
    policy_rule: str,
    already_code: str,
    already_value: bool,
    rationale: str,
    expected_effect: str,
    categories: Sequence[str],
) -> None:
    capability = planner.capability(capability_id)
    if capability is None:
        planner.skip(
            policy_rule,
            (capability_id,),
            "capability_unavailable",
            "Capability {!r} was not declared or safely detected.".format(
                capability_id
            ),
        )
        return
    if capability.current_value is already_value:
        planner.skip(
            policy_rule,
            (capability_id,),
            already_code,
            "Capability {!r} is already {}.".format(
                capability_id,
                "enabled" if already_value else "disabled",
            ),
            planner.refs(*categories),
        )
        return
    planner.add(
        policy_rule=policy_rule,
        changes=(
            InterventionChange(
                capability_id,
                operation,
                proposed_value,
            ),
        ),
        rationale=rationale,
        expected_effect=expected_effect,
        evidence_refs=planner.refs(*categories),
    )


def _allocator_fragmentation_signal(
    capsule: dict,
    framework: Mapping[str, object],
) -> Tuple[bool, str]:
    allocated = _finite_nonnegative(framework.get("allocated_bytes"))
    reserved = _finite_nonnegative(framework.get("reserved_bytes"))
    message = str((capsule.get("failure") or {}).get("message") or "").lower()
    message_signal = (
        "reserved" in message
        and ("unallocated" in message or "fragment" in message)
    )
    numeric_signal = False
    gap = None
    if allocated is not None and reserved is not None and reserved >= allocated:
        gap = reserved - allocated
        numeric_signal = (
            gap >= 256 * 1024 * 1024
            and reserved >= max(1.2 * max(allocated, 1), allocated + 1)
        )
    if numeric_signal:
        return (
            True,
            "Allocator evidence records {} reserved bytes versus {} allocated "
            "bytes, leaving a {}-byte gap consistent with fragmentation."
            .format(int(reserved), int(allocated), int(gap)),
        )
    if message_signal:
        return (
            True,
            "The captured CUDA OOM message explicitly reports reserved but "
            "unallocated or fragmented memory.",
        )
    return (
        False,
        "The capsule does not contain the deterministic reserved-versus-"
        "allocated signal required for an allocator intervention.",
    )


def _validate_policy_inputs(
    capsule: dict,
    manifest: CapabilityManifest,
    max_proposals: int,
    include_approval_required: bool,
) -> dict:
    if not isinstance(capsule, dict):
        raise OOMPolicyError("capsule must be an object")
    errors = validate_capsule(capsule)
    if errors:
        raise OOMPolicyError(
            "invalid failure capsule: {}".format("; ".join(errors))
        )
    failure = capsule.get("failure") or {}
    classification = failure.get("classification") or {}
    failure_class = failure.get("class") or capsule.get("failure_class")
    if failure_class != OOM_FAILURE_CLASS:
        raise OOMPolicyError(
            "OOM policy requires failure class {!r}, got {!r}".format(
                OOM_FAILURE_CLASS,
                failure_class,
            )
        )
    if classification.get("rule") != OOM_FAILURE_CLASS:
        raise OOMPolicyError(
            "capsule classification does not match the CUDA OOM rule"
        )
    if classification.get("match_kind") != "deterministic":
        raise OOMPolicyError(
            "OOM policy requires a deterministic classification"
        )
    if classification.get("recoverable_by_bounded_trial") is not True:
        raise OOMPolicyError(
            "capsule is not marked recoverable by a bounded trial"
        )
    if not isinstance(manifest, CapabilityManifest):
        raise OOMPolicyError("manifest must be a CapabilityManifest")
    if (
        isinstance(max_proposals, bool)
        or not isinstance(max_proposals, int)
        or max_proposals < 1
        or max_proposals > HARD_MAX_PROPOSALS
    ):
        raise OOMPolicyError(
            "max_proposals must be an integer from 1 to {}".format(
                HARD_MAX_PROPOSALS
            )
        )
    if not isinstance(include_approval_required, bool):
        raise OOMPolicyError("include_approval_required must be a boolean")
    evidence = capsule.get("evidence")
    if not isinstance(evidence, dict) or not isinstance(
        evidence.get("config"),
        dict,
    ):
        raise OOMPolicyError("capsule evidence.config must be an object")
    return capsule


def _validate_manifest_baseline(
    manifest: CapabilityManifest,
    config: dict,
    source_environment: Mapping[str, str],
) -> None:
    for capability in manifest.capabilities:
        try:
            if capability.location == "config":
                current = get_config_value(config, capability.target)
            elif capability.location == "environment":
                if capability.target not in source_environment:
                    raise OOMPolicyError(
                        "capability manifest expects environment target {!r}, "
                        "but the capsule did not capture its baseline".format(
                            capability.target
                        )
                    )
                current = source_environment[capability.target]
            else:
                raise OOMPolicyError(
                    "capability has unsupported location {!r}".format(
                        capability.location
                    )
                )
        except CapabilityError as exc:
            raise OOMPolicyError(str(exc)) from exc
        if type(current) is not type(capability.current_value) or (
            current != capability.current_value
        ):
            raise OOMPolicyError(
                "capability manifest is stale for {!r}: manifest={!r}, "
                "capsule={!r}".format(
                    capability.capability_id,
                    capability.current_value,
                    current,
                )
            )


def _present_evidence_ids(capsule: dict) -> Dict[str, str]:
    evidence = capsule.get("evidence") or {}
    raw_index = capsule.get("evidence_index")
    index = raw_index if isinstance(raw_index, list) else build_evidence_index(evidence)
    present = {}
    for item in index:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        evidence_id = item.get("id")
        if (
            category in _EVIDENCE_CATEGORY_IDS
            and evidence_id == _EVIDENCE_CATEGORY_IDS[category]
            and evidence.get(category) not in (None, {}, [], "")
        ):
            present[category] = evidence_id
    return present


def _proposal_id(
    run_id: str,
    policy_rule: str,
    changes: Iterable[InterventionChange],
) -> str:
    payload = {
        "run_id": run_id,
        "policy_rule": policy_rule,
        "changes": [change.to_dict() for change in changes],
        "policy_rule_version": OOM_POLICY_RULE_VERSION,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "oom-{}".format(hashlib.sha256(encoded).hexdigest()[:20])


def _finite_nonnegative(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        return None
    return normalized


def _reject_unknown_fields(
    payload: dict,
    allowed: set,
    artifact_name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise OOMPolicyError(
            "{} contains unknown fields: {}".format(artifact_name, unknown)
        )


__all__ = [
    "AUTOMATIC_POLICY_RULES",
    "DEFAULT_MAX_PROPOSALS",
    "HARD_MAX_PROPOSALS",
    "OOM_POLICY_RULE_VERSION",
    "OOM_POLICY_SCHEMA_NAME",
    "OOM_POLICY_SCHEMA_VERSION",
    "OOMPolicyError",
    "OOMPolicyPlan",
    "POLICY_RULE_ORDER",
    "POLICY_SKIP_CODES",
    "PolicySkip",
    "plan_oom_interventions",
]