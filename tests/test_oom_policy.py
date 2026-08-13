"""Acceptance tests for WatcherML's deterministic CUDA OOM policy."""
from __future__ import annotations

import json
import unittest
from copy import deepcopy

from watcherml.capabilities import (
    CapabilityDeclaration,
    CapabilityManifest,
    discover_capabilities,
)
from watcherml.capsule import build_evidence_index
from watcherml.oom_policy import (
    AUTOMATIC_POLICY_RULES,
    DEFAULT_MAX_PROPOSALS,
    HARD_MAX_PROPOSALS,
    OOM_POLICY_RULE_VERSION,
    OOM_POLICY_SCHEMA_NAME,
    OOM_POLICY_SCHEMA_VERSION,
    POLICY_RULE_ORDER,
    OOMPolicyError,
    OOMPolicyPlan,
    PolicySkip,
    plan_oom_interventions,
)


FULL_RULE_ORDER = (
    "halve_batch_preserve_effective_batch",
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


def _full_config() -> dict:
    return {
        "trainer": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0002,
        },
        "model": {
            "gradient_checkpointing": False,
            "max_seq_length": 2048,
            "use_cache": True,
            "memory_efficient_attention": False,
            "activation_offload": False,
            "parameter_offload": False,
            "attn_implementation": "eager",
            "name": "example/model",
        },
        "runtime": {
            "precision": "fp32",
        },
        "optimizer": {
            "offload_optimizer": False,
            "optimizer_bits": 32,
        },
        "dataset": {
            "fingerprint": "dataset-example",
        },
    }


def _framework(**overrides) -> dict:
    values = {
        "python_version": "3.12.6",
        "torch_available": True,
        "cuda_available": True,
        "bf16_supported": True,
        "allocated_bytes": 8 * 1024**3,
        "reserved_bytes": 12 * 1024**3,
        "allocator_config": "max_split_size_mb:128",
    }
    values.update(overrides)
    return values


def _capsule(
    config: dict,
    *,
    framework=None,
    message: str = "CUDA out of memory. Tried to allocate 2.00 GiB",
    failure_class: str = "cuda_out_of_memory",
    match_kind: str = "deterministic",
    recoverable: bool = True,
    run_id: str = "source-oom-run",
) -> dict:
    framework = _framework() if framework is None else framework
    evidence = {
        "config": deepcopy(config),
        "training_state": {
            "last_logged_step": 417,
        },
        "runtime": {
            "pid": 1234,
            "working_directory": "/project",
        },
        "resource_state_at_failure": {
            "vram_used_mib_peak": 15_500,
        },
        "gpu": {
            "available": True,
            "gpus": [{"name": "test-gpu", "memory_total_mib": 16_384}],
        },
        "framework": deepcopy(framework),
        "environment": {
            "fingerprint": "environment-fingerprint",
        },
        "git": {
            "available": True,
            "commit": "abc123",
        },
        "dataset": {
            "fingerprint": "dataset-example",
        },
        "recent_metrics": [
            {"name": "loss", "value": 0.8, "step": 417},
        ],
        "notebook_cells_executed": None,
    }
    classification = {
        "rule": failure_class,
        "rule_version": "1.0",
        "match_kind": match_kind,
        "recoverable_by_bounded_trial": recoverable,
        "evidence_ids": ["EV-1", "EV-2", "EV-4", "EV-5", "EV-6"],
    }
    failure = {
        "class": failure_class,
        "exception_type": "RuntimeError",
        "message": message,
        "traceback": "Traceback: CUDA out of memory",
        "classification": classification,
    }
    return {
        "schema": {
            "name": "watcherml.failure-capsule",
            "version": "1.0",
        },
        "run_id": run_id,
        "project": "policy-tests",
        "captured_at": 1_800_000_000.0,
        "failure": failure,
        "failure_class": failure_class,
        "evidence": evidence,
        "evidence_index": build_evidence_index(evidence),
        "capture": {
            "score": 10,
            "maximum": 10,
            "present": [],
            "missing": [],
        },
    }


def _manifest_for(capsule: dict, declarations=None):
    framework = capsule["evidence"].get("framework") or {}
    environment = {}
    if framework.get("allocator_config") is not None:
        environment["PYTORCH_CUDA_ALLOC_CONF"] = str(
            framework["allocator_config"]
        )
    return discover_capabilities(
        capsule["evidence"]["config"],
        declarations=declarations,
        environment=environment,
    )


def _proposal(plan: OOMPolicyPlan, rule: str):
    return next(item for item in plan.proposals if item.policy_rule == rule)


def _skip(plan: OOMPolicyPlan, rule: str):
    return next(item for item in plan.skipped if item.policy_rule == rule)


class FullPolicyPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = _full_config()
        self.capsule = _capsule(self.config)
        self.manifest = _manifest_for(self.capsule)

    def test_full_capability_plan_has_documented_order_and_breadth(self):
        plan = plan_oom_interventions(self.capsule, self.manifest)

        self.assertEqual(
            tuple(proposal.policy_rule for proposal in plan.proposals),
            FULL_RULE_ORDER,
        )
        self.assertEqual(len(plan.proposals), 13)
        self.assertGreaterEqual(DEFAULT_MAX_PROPOSALS, len(plan.proposals))
        self.assertEqual(plan.skipped, ())

    def test_only_original_low_risk_ladder_is_automatic(self):
        plan = plan_oom_interventions(self.capsule, self.manifest)

        self.assertEqual(
            plan.automatic_proposal_ids,
            tuple(proposal.proposal_id for proposal in plan.proposals[:3]),
        )
        self.assertEqual(
            plan.approval_required_proposal_ids,
            tuple(proposal.proposal_id for proposal in plan.proposals[3:]),
        )
        self.assertTrue(
            all(
                proposal.policy_rule in AUTOMATIC_POLICY_RULES
                for proposal in plan.proposals[:3]
            )
        )
        self.assertTrue(
            all(
                proposal.policy_rule not in AUTOMATIC_POLICY_RULES
                for proposal in plan.proposals[3:]
            )
        )

    def test_original_three_rules_preserve_order_and_effective_batch(self):
        plan = plan_oom_interventions(self.capsule, self.manifest)
        batch = plan.proposals[0]
        checkpoint = plan.proposals[1]
        combined = plan.proposals[2]

        self.assertEqual(
            [change.capability_id for change in batch.changes],
            ["micro_batch_size", "gradient_accumulation_steps"],
        )
        self.assertEqual(
            [change.proposed_value for change in batch.changes],
            [16, 2],
        )
        self.assertEqual(
            checkpoint.changes,
            (
                checkpoint.changes[0],
            ),
        )
        self.assertEqual(
            checkpoint.changes[0].capability_id,
            "gradient_checkpointing",
        )
        self.assertEqual(
            [change.capability_id for change in combined.changes],
            [
                "micro_batch_size",
                "gradient_accumulation_steps",
                "gradient_checkpointing",
            ],
        )

    def test_every_proposal_cites_only_present_capsule_evidence(self):
        plan = plan_oom_interventions(self.capsule, self.manifest)
        present = {
            item["id"] for item in self.capsule["evidence_index"]
        }

        for proposal in plan.proposals:
            with self.subTest(rule=proposal.policy_rule):
                self.assertTrue(proposal.evidence_refs)
                self.assertTrue(set(proposal.evidence_refs) <= present)
                self.assertEqual(
                    len(proposal.evidence_refs),
                    len(set(proposal.evidence_refs)),
                )

    def test_repeated_planning_and_json_round_trip_are_deterministic(self):
        first = plan_oom_interventions(self.capsule, self.manifest)
        second = plan_oom_interventions(self.capsule, self.manifest)

        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(OOMPolicyPlan.from_json(first.to_json()), first)
        self.assertEqual(
            len({proposal.proposal_id for proposal in first.proposals}),
            len(first.proposals),
        )
        payload = json.loads(first.to_json())
        self.assertEqual(
            payload["schema"],
            {
                "name": OOM_POLICY_SCHEMA_NAME,
                "version": OOM_POLICY_SCHEMA_VERSION,
            },
        )
        self.assertEqual(
            payload["policy_rule_version"],
            OOM_POLICY_RULE_VERSION,
        )

    def test_planning_never_mutates_capsule_config_or_manifest(self):
        capsule_before = deepcopy(self.capsule)
        manifest_before = self.manifest.to_json()

        plan_oom_interventions(self.capsule, self.manifest)

        self.assertEqual(self.capsule, capsule_before)
        self.assertEqual(self.config, _full_config())
        self.assertEqual(self.manifest.to_json(), manifest_before)

    def test_odd_batch_uses_integer_accumulation_without_reducing_effective_batch(self):
        config = {
            "batch_size": 15,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": False,
        }
        capsule = _capsule(config, framework=_framework(allocator_config=None))
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)
        batch = plan.proposals[0]
        values = {
            change.capability_id: change.proposed_value
            for change in batch.changes
        }

        self.assertEqual(values["micro_batch_size"], 7)
        self.assertEqual(values["gradient_accumulation_steps"], 5)
        self.assertGreaterEqual(7 * 5, 15 * 2)

    def test_missing_accumulation_uses_explicit_micro_batch_only_rule(self):
        config = {
            "batch_size": 8,
            "gradient_checkpointing": False,
        }
        capsule = _capsule(config, framework=_framework(allocator_config=None))
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)

        self.assertEqual(plan.proposals[0].policy_rule, "halve_micro_batch")
        self.assertEqual(len(plan.proposals[0].changes), 1)
        self.assertEqual(
            plan.proposals[0].changes[0].capability_id,
            "micro_batch_size",
        )
        self.assertIn(
            "effective batch size may change",
            plan.proposals[0].expected_effect,
        )

    def test_project_can_tighten_an_automatic_capability_to_require_approval(self):
        declarations = [
            CapabilityDeclaration(
                "micro_batch_size",
                "trainer.per_device_train_batch_size",
                "approval_required",
            )
        ]
        manifest = _manifest_for(self.capsule, declarations=declarations)

        plan = plan_oom_interventions(self.capsule, manifest)
        batch = _proposal(plan, "halve_batch_preserve_effective_batch")
        combined = _proposal(plan, "halve_batch_and_checkpoint")

        self.assertIn(batch.proposal_id, plan.approval_required_proposal_ids)
        self.assertIn(combined.proposal_id, plan.approval_required_proposal_ids)
        self.assertNotIn(batch.proposal_id, plan.automatic_proposal_ids)
        self.assertNotIn(combined.proposal_id, plan.automatic_proposal_ids)
        self.assertEqual(OOMPolicyPlan.from_json(plan.to_json()), plan)

        filtered = plan_oom_interventions(
            self.capsule,
            manifest,
            include_approval_required=False,
        )
        self.assertNotIn(
            "halve_batch_preserve_effective_batch",
            {item.policy_rule for item in filtered.proposals},
        )
        self.assertEqual(
            _skip(filtered, "halve_batch_preserve_effective_batch").code,
            "approval_filtered",
        )


class PolicyFilteringAndSkipTests(unittest.TestCase):
    def setUp(self):
        self.capsule = _capsule(_full_config())
        self.manifest = _manifest_for(self.capsule)

    def test_approval_filter_retains_automatic_ladder_and_audits_every_gate(self):
        plan = plan_oom_interventions(
            self.capsule,
            self.manifest,
            include_approval_required=False,
        )

        self.assertEqual(
            tuple(item.policy_rule for item in plan.proposals),
            FULL_RULE_ORDER[:3],
        )
        self.assertEqual(plan.approval_required_proposal_ids, ())
        self.assertEqual(
            sum(item.code == "approval_filtered" for item in plan.skipped),
            10,
        )

    def test_proposal_limit_keeps_policy_prefix_and_records_later_skips(self):
        plan = plan_oom_interventions(
            self.capsule,
            self.manifest,
            max_proposals=2,
        )

        self.assertEqual(
            tuple(item.policy_rule for item in plan.proposals),
            FULL_RULE_ORDER[:2],
        )
        self.assertEqual(
            sum(item.code == "proposal_limit" for item in plan.skipped),
            len(FULL_RULE_ORDER) - 2,
        )

    def test_invalid_proposal_limits_and_filter_flag_are_rejected(self):
        invalid_limits = (0, -1, HARD_MAX_PROPOSALS + 1, True, 1.5)
        for value in invalid_limits:
            with self.subTest(value=value):
                with self.assertRaises(OOMPolicyError):
                    plan_oom_interventions(
                        self.capsule,
                        self.manifest,
                        max_proposals=value,
                    )

        with self.assertRaises(OOMPolicyError):
            plan_oom_interventions(
                self.capsule,
                self.manifest,
                include_approval_required="yes",
            )

    def test_sparse_manifest_produces_skips_instead_of_guessed_controls(self):
        config = {"unrelated": "value"}
        capsule = _capsule(config, framework=_framework(allocator_config=None))
        manifest = _manifest_for(capsule)

        plan = plan_oom_interventions(capsule, manifest)

        self.assertEqual(plan.proposals, ())
        self.assertEqual(plan.automatic_proposal_ids, ())
        self.assertEqual(plan.approval_required_proposal_ids, ())
        self.assertTrue(plan.skipped)
        self.assertTrue(
            all(item.code == "capability_unavailable" for item in plan.skipped)
        )

    def test_minimal_or_already_enabled_automatic_controls_are_skipped(self):
        config = {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": True,
        }
        capsule = _capsule(config, framework=_framework(allocator_config=None))
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)

        self.assertEqual(
            _skip(plan, "halve_batch_preserve_effective_batch").code,
            "already_minimal",
        )
        self.assertEqual(
            _skip(plan, "enable_gradient_checkpointing").code,
            "already_enabled",
        )
        self.assertEqual(
            _skip(plan, "halve_batch_and_checkpoint").code,
            "capability_unavailable",
        )

    def test_already_configured_broader_controls_are_audited_as_skips(self):
        config = {
            "use_cache": False,
            "memory_efficient_attention": True,
            "attn_implementation": "sdpa",
            "precision": "bf16",
            "max_seq_length": 1,
            "activation_offload": True,
            "offload_optimizer": True,
            "optimizer_bits": 8,
            "parameter_offload": True,
        }
        capsule = _capsule(config, framework=_framework(allocator_config=None))
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)

        expected = {
            "disable_training_model_cache": "already_disabled",
            "enable_memory_efficient_attention": "already_enabled",
            "use_sdpa_attention": "already_enabled",
            "use_lower_memory_precision": "already_enabled",
            "halve_sequence_length": "already_minimal",
            "enable_activation_offload": "already_enabled",
            "enable_optimizer_state_offload": "already_enabled",
            "use_8bit_optimizer_state": "already_minimal",
            "enable_parameter_offload": "already_enabled",
        }
        for rule, code in expected.items():
            with self.subTest(rule=rule):
                self.assertEqual(_skip(plan, rule).code, code)


class EvidenceGateTests(unittest.TestCase):
    def test_numeric_allocator_fragmentation_signal_appends_existing_config(self):
        config = {"batch_size": 32}
        framework = _framework(
            allocated_bytes=8 * 1024**3,
            reserved_bytes=12 * 1024**3,
        )
        capsule = _capsule(config, framework=framework)
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)
        proposal = _proposal(plan, "allocator_fragmentation_mitigation")

        self.assertEqual(
            proposal.changes[0].proposed_value,
            "max_split_size_mb:128,expandable_segments:True",
        )
        self.assertIn("reserved bytes", proposal.rationale)

    def test_allocator_message_signal_works_without_numeric_allocator_state(self):
        config = {"batch_size": 32}
        framework = _framework(
            allocated_bytes=None,
            reserved_bytes=None,
        )
        capsule = _capsule(
            config,
            framework=framework,
            message=(
                "CUDA out of memory; reserved but unallocated memory remains"
            ),
        )
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)

        self.assertIn(
            "allocator_fragmentation_mitigation",
            {item.policy_rule for item in plan.proposals},
        )

    def test_allocator_without_fragmentation_signal_is_skipped(self):
        config = {"batch_size": 32}
        framework = _framework(
            allocated_bytes=8 * 1024**3,
            reserved_bytes=8 * 1024**3 + 64 * 1024**2,
        )
        capsule = _capsule(config, framework=framework)
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)

        self.assertEqual(
            _skip(plan, "allocator_fragmentation_mitigation").code,
            "missing_evidence",
        )

    def test_allocator_already_using_expandable_segments_is_skipped(self):
        config = {"batch_size": 32}
        framework = _framework(
            allocator_config=(
                "max_split_size_mb:128,expandable_segments:True"
            ),
        )
        capsule = _capsule(config, framework=framework)
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)

        self.assertEqual(
            _skip(plan, "allocator_fragmentation_mitigation").code,
            "already_enabled",
        )

    def test_precision_prefers_bf16_with_explicit_support(self):
        config = {"precision": "fp32"}
        capsule = _capsule(
            config,
            framework=_framework(allocator_config=None),
        )
        manifest = _manifest_for(capsule)
        proposal = _proposal(
            plan_oom_interventions(capsule, manifest),
            "use_lower_memory_precision",
        )

        self.assertEqual(proposal.changes[0].proposed_value, "bf16")

    def test_precision_uses_fp16_with_cuda_but_without_bf16_evidence(self):
        config = {"precision": "fp32"}
        framework = _framework(
            allocator_config=None,
            bf16_supported=False,
            cuda_available=True,
        )
        capsule = _capsule(config, framework=framework)
        manifest = _manifest_for(capsule)
        proposal = _proposal(
            plan_oom_interventions(capsule, manifest),
            "use_lower_memory_precision",
        )

        self.assertEqual(proposal.changes[0].proposed_value, "fp16")

    def test_precision_without_cuda_or_bf16_evidence_is_skipped(self):
        config = {"precision": "fp32"}
        framework = _framework(
            allocator_config=None,
            bf16_supported=False,
            cuda_available=False,
        )
        capsule = _capsule(config, framework=framework)
        manifest = _manifest_for(capsule)
        plan = plan_oom_interventions(capsule, manifest)

        self.assertEqual(
            _skip(plan, "use_lower_memory_precision").code,
            "missing_evidence",
        )

    def test_attention_policy_only_changes_eager_to_sdpa(self):
        for current, expected in (
            ("eager", "proposal"),
            ("sdpa", "already_enabled"),
            ("flash_attention_2", "unsupported_baseline"),
        ):
            config = {"attn_implementation": current}
            capsule = _capsule(
                config,
                framework=_framework(allocator_config=None),
            )
            manifest = _manifest_for(capsule)
            plan = plan_oom_interventions(capsule, manifest)
            with self.subTest(current=current):
                if expected == "proposal":
                    proposal = _proposal(plan, "use_sdpa_attention")
                    self.assertEqual(
                        proposal.changes[0].proposed_value,
                        "sdpa",
                    )
                else:
                    self.assertEqual(
                        _skip(plan, "use_sdpa_attention").code,
                        expected,
                    )


class PolicyInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.capsule = _capsule(_full_config())
        self.manifest = _manifest_for(self.capsule)

    def test_non_oom_nondeterministic_and_nonrecoverable_capsules_are_rejected(self):
        cases = (
            _capsule(
                _full_config(),
                failure_class="nan_or_exploding_loss",
            ),
            _capsule(_full_config(), match_kind="model_generated"),
            _capsule(_full_config(), recoverable=False),
        )
        for capsule in cases:
            with self.subTest(classification=capsule["failure"]["classification"]):
                manifest = _manifest_for(capsule)
                with self.assertRaises(OOMPolicyError):
                    plan_oom_interventions(capsule, manifest)

    def test_invalid_capsule_schema_or_missing_config_is_rejected(self):
        damaged_schema = deepcopy(self.capsule)
        damaged_schema["schema"]["version"] = "99.0"
        with self.assertRaises(OOMPolicyError):
            plan_oom_interventions(damaged_schema, self.manifest)

        missing_config = deepcopy(self.capsule)
        del missing_config["evidence"]["config"]
        with self.assertRaises(OOMPolicyError):
            plan_oom_interventions(missing_config, self.manifest)

    def test_stale_or_missing_config_target_rejects_entire_plan(self):
        changed = deepcopy(self.capsule)
        changed["evidence"]["config"]["trainer"][
            "per_device_train_batch_size"
        ] = 24
        with self.assertRaisesRegex(OOMPolicyError, "stale"):
            plan_oom_interventions(changed, self.manifest)

        missing = deepcopy(self.capsule)
        del missing["evidence"]["config"]["trainer"][
            "per_device_train_batch_size"
        ]
        with self.assertRaises(OOMPolicyError):
            plan_oom_interventions(missing, self.manifest)

    def test_missing_or_changed_allocator_baseline_rejects_entire_plan(self):
        missing = deepcopy(self.capsule)
        missing["evidence"]["framework"]["allocator_config"] = None
        with self.assertRaisesRegex(OOMPolicyError, "environment target"):
            plan_oom_interventions(missing, self.manifest)

        changed = deepcopy(self.capsule)
        changed["evidence"]["framework"][
            "allocator_config"
        ] = "max_split_size_mb:64"
        with self.assertRaisesRegex(OOMPolicyError, "stale"):
            plan_oom_interventions(changed, self.manifest)

    def test_manifest_type_is_required(self):
        with self.assertRaises(OOMPolicyError):
            plan_oom_interventions(self.capsule, {})


class PolicySerializationSafetyTests(unittest.TestCase):
    def setUp(self):
        capsule = _capsule(_full_config())
        self.plan = plan_oom_interventions(
            capsule,
            _manifest_for(capsule),
        )

    def test_permission_relabel_reorder_and_id_detachment_are_rejected(self):
        permission = self.plan.to_dict()
        moved = permission["approval_required_proposal_ids"].pop(0)
        permission["automatic_proposal_ids"].append(moved)

        reordered = self.plan.to_dict()
        reordered["proposals"][0], reordered["proposals"][1] = (
            reordered["proposals"][1],
            reordered["proposals"][0],
        )

        detached = self.plan.to_dict()
        detached["proposals"][0]["proposal_id"] = "oom-forged"

        for payload in (permission, reordered, detached):
            with self.subTest(payload=payload):
                with self.assertRaises(OOMPolicyError):
                    OOMPolicyPlan.from_dict(payload)

    def test_unknown_rule_user_proposer_and_extra_plan_fields_are_rejected(self):
        unknown_rule = self.plan.to_dict()
        unknown_rule["proposals"][0]["policy_rule"] = "invented_rule"

        user_proposer = self.plan.to_dict()
        user_proposer["proposals"][0]["proposer"] = "user"

        extra = self.plan.to_dict()
        extra["verified_recovery"] = True

        fake_skip = self.plan.to_dict()
        fake_skip["skipped"].append(
            {
                "policy_rule": "invented_rule",
                "capability_ids": ["invented_capability"],
                "code": "capability_unavailable",
                "reason": "Invented audit record.",
                "evidence_refs": [],
            }
        )

        for payload in (unknown_rule, user_proposer, extra, fake_skip):
            with self.subTest(payload=payload):
                with self.assertRaises(OOMPolicyError):
                    OOMPolicyPlan.from_dict(payload)

    def test_schema_json_and_permission_partition_validation_fail_closed(self):
        wrong_name = self.plan.to_dict()
        wrong_name["schema"]["name"] = "other.schema"

        wrong_version = self.plan.to_dict()
        wrong_version["schema"]["version"] = "99.0"

        incomplete = self.plan.to_dict()
        incomplete["automatic_proposal_ids"].pop()

        overlapping = self.plan.to_dict()
        overlapping["approval_required_proposal_ids"].append(
            overlapping["automatic_proposal_ids"][0]
        )

        for payload in (wrong_name, wrong_version, incomplete, overlapping):
            with self.subTest(payload=payload):
                with self.assertRaises(OOMPolicyError):
                    OOMPolicyPlan.from_dict(payload)

        for encoded in ("not json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(OOMPolicyError):
                    OOMPolicyPlan.from_json(encoded)

    def test_policy_skip_round_trip_and_validation(self):
        skip = PolicySkip(
            policy_rule="use_lower_memory_precision",
            capability_ids=("precision",),
            code="missing_evidence",
            reason="No CUDA support evidence was captured.",
            evidence_refs=("EV-1", "EV-6"),
        )
        self.assertEqual(PolicySkip.from_dict(skip.to_dict()), skip)

        invalid_code = skip.to_dict()
        invalid_code["code"] = "silently_ignore"
        with self.assertRaises(OOMPolicyError):
            PolicySkip.from_dict(invalid_code)

        unknown = skip.to_dict()
        unknown["execute_anyway"] = True
        with self.assertRaises(OOMPolicyError):
            PolicySkip.from_dict(unknown)


if __name__ == "__main__":
    unittest.main()