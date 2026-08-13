"""Acceptance tests for WatcherML's bounded intervention artifacts."""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace

from watcherml.capabilities import (
    CapabilityDeclaration,
    discover_capabilities,
)
from watcherml.interventions import (
    INTERVENTION_AUTHORIZATION_SCHEMA_NAME,
    INTERVENTION_AUTHORIZATION_SCHEMA_VERSION,
    INTERVENTION_PROPOSAL_SCHEMA_NAME,
    INTERVENTION_PROPOSAL_SCHEMA_VERSION,
    INTERVENTION_RESOLUTION_SCHEMA_NAME,
    INTERVENTION_RESOLUTION_SCHEMA_VERSION,
    MAX_CHANGES_PER_INTERVENTION,
    InterventionAuthorization,
    InterventionAuthorizationError,
    InterventionChange,
    InterventionError,
    InterventionProposal,
    InterventionResolutionError,
    ResolvedIntervention,
    StaleInterventionError,
    materialize_intervention,
    proposal_digest,
    resolve_intervention,
)


def _source_config() -> dict:
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
            "name": "example/model",
        },
        "runtime": {
            "precision": "fp32",
            "attn_implementation": "eager",
        },
        "dataset": {
            "fingerprint": "dataset-sha256-example",
        },
    }


def _batch_proposal(proposal_id: str = "proposal-batch-1"):
    return InterventionProposal(
        proposal_id=proposal_id,
        policy_rule="halve_batch_preserve_effective_batch",
        changes=(
            InterventionChange(
                "micro_batch_size",
                "decrease",
                16,
            ),
            InterventionChange(
                "gradient_accumulation_steps",
                "increase",
                2,
            ),
        ),
        rationale=(
            "The deterministic OOM capsule captured a per-device batch size "
            "of 32. Reduce per-step activation memory while preserving the "
            "effective batch size."
        ),
        expected_effect=(
            "Lower activation memory while preserving effective batch size."
        ),
        evidence_refs=("EV-1", "EV-4", "EV-6"),
    )


def _sequence_proposal(proposal_id: str = "proposal-sequence-1"):
    return InterventionProposal(
        proposal_id=proposal_id,
        policy_rule="reduce_sequence_length",
        changes=(
            InterventionChange("sequence_length", "decrease", 1024),
        ),
        rationale=(
            "The captured configuration uses a 2048-token sequence length, "
            "which contributes to activation and attention memory."
        ),
        expected_effect="Reduce attention and activation memory.",
        evidence_refs=("EV-4", "EV-6"),
    )


class InterventionArtifactTests(unittest.TestCase):
    def test_change_round_trip_preserves_canonical_transition(self):
        change = InterventionChange(
            "micro_batch_size",
            "decrease",
            16,
        )

        self.assertEqual(
            InterventionChange.from_dict(change.to_dict()),
            change,
        )

    def test_change_rejects_invalid_operation_and_nonfinite_or_complex_values(self):
        invalid = (
            ("replace", 16),
            ("decrease", float("nan")),
            ("decrease", float("inf")),
            ("decrease", None),
            ("decrease", {"value": 16}),
            ("decrease", [16]),
        )
        for operation, value in invalid:
            with self.subTest(operation=operation, value=value):
                with self.assertRaises(InterventionError):
                    InterventionChange(
                        "micro_batch_size",
                        operation,
                        value,
                    )

    def test_change_deserialization_rejects_unknown_executable_fields(self):
        payload = InterventionChange(
            "micro_batch_size",
            "decrease",
            16,
        ).to_dict()
        payload["shell_command"] = "python arbitrary_script.py"

        with self.assertRaisesRegex(InterventionError, "unknown fields"):
            InterventionChange.from_dict(payload)

    def test_proposal_round_trip_and_digest_are_deterministic(self):
        proposal = _batch_proposal()
        encoded = proposal.to_json()
        restored = InterventionProposal.from_json(encoded)

        self.assertEqual(restored, proposal)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(proposal_digest(restored), proposal_digest(proposal))
        self.assertRegex(proposal_digest(proposal), r"^[a-f0-9]{64}$")

        payload = json.loads(encoded)
        self.assertEqual(
            payload["schema"],
            {
                "name": INTERVENTION_PROPOSAL_SCHEMA_NAME,
                "version": INTERVENTION_PROPOSAL_SCHEMA_VERSION,
            },
        )

    def test_proposal_digest_changes_when_any_audited_content_changes(self):
        proposal = _batch_proposal()
        mutations = (
            replace(proposal, policy_rule="another_bounded_rule"),
            replace(proposal, rationale="Different evidence-backed rationale."),
            replace(proposal, expected_effect="Different expected effect."),
            replace(proposal, evidence_refs=("EV-1",)),
            replace(proposal, proposer="user"),
            replace(
                proposal,
                changes=(
                    InterventionChange(
                        "micro_batch_size",
                        "decrease",
                        8,
                    ),
                    proposal.changes[1],
                ),
            ),
        )
        original_digest = proposal_digest(proposal)
        for mutated in mutations:
            with self.subTest(mutated=mutated):
                self.assertNotEqual(proposal_digest(mutated), original_digest)

    def test_proposal_requires_bounded_unique_changes_and_evidence(self):
        common = {
            "proposal_id": "proposal-invalid",
            "policy_rule": "bounded_rule",
            "rationale": "Evidence-backed rationale.",
            "expected_effect": "Bounded expected effect.",
            "evidence_refs": ("EV-1",),
        }

        with self.assertRaisesRegex(InterventionError, "at least one change"):
            InterventionProposal(changes=(), **common)

        duplicate = InterventionChange(
            "micro_batch_size",
            "decrease",
            16,
        )
        with self.assertRaisesRegex(InterventionError, "at most once"):
            InterventionProposal(
                changes=(duplicate, duplicate),
                **common,
            )

        too_many = tuple(
            InterventionChange(
                "future_capability_{}".format(index),
                "set",
                index,
            )
            for index in range(MAX_CHANGES_PER_INTERVENTION + 1)
        )
        with self.assertRaisesRegex(InterventionError, "at most"):
            InterventionProposal(changes=too_many, **common)

        with self.assertRaisesRegex(InterventionError, "evidence reference"):
            InterventionProposal(
                changes=(duplicate,),
                **{**common, "evidence_refs": ()},
            )

        with self.assertRaisesRegex(InterventionError, "must be unique"):
            InterventionProposal(
                changes=(duplicate,),
                **{**common, "evidence_refs": ("EV-1", "EV-1")},
            )

    def test_proposal_rejects_invalid_identity_rule_proposer_and_evidence(self):
        change = InterventionChange(
            "micro_batch_size",
            "decrease",
            16,
        )
        cases = (
            {"proposal_id": "proposal with spaces"},
            {"policy_rule": "Uppercase-Rule"},
            {"proposer": "llm"},
            {"evidence_refs": ("invalid evidence ref",)},
        )
        base = {
            "proposal_id": "proposal-valid",
            "policy_rule": "bounded_rule",
            "changes": (change,),
            "rationale": "Evidence-backed rationale.",
            "expected_effect": "Bounded expected effect.",
            "evidence_refs": ("EV-1",),
        }
        for mutation in cases:
            with self.subTest(mutation=mutation):
                with self.assertRaises(InterventionError):
                    InterventionProposal(**{**base, **mutation})

    def test_proposal_deserialization_rejects_schema_changes_and_field_smuggling(self):
        payload = _batch_proposal().to_dict()
        mutations = (
            ("schema_name", "other.schema"),
            ("schema_version", "99.0"),
            ("shell_command", "python arbitrary_script.py"),
            ("dependency_patch", {"torch": "new-version"}),
            ("dataset_patch", {"fingerprint": "different"}),
        )
        for field, value in mutations:
            damaged = deepcopy(payload)
            if field == "schema_name":
                damaged["schema"]["name"] = value
            elif field == "schema_version":
                damaged["schema"]["version"] = value
            else:
                damaged[field] = value
            with self.subTest(field=field):
                with self.assertRaises(InterventionError):
                    InterventionProposal.from_dict(damaged)

        for encoded in ("not json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(InterventionError):
                    InterventionProposal.from_json(encoded)


class InterventionResolutionTests(unittest.TestCase):
    def setUp(self):
        self.config = _source_config()
        self.manifest = discover_capabilities(self.config)

    def test_coupled_batch_intervention_resolves_to_real_nested_targets(self):
        proposal = _batch_proposal()

        resolved = resolve_intervention(
            proposal,
            self.manifest,
            self.config,
        )

        self.assertEqual(resolved.required_permission, "automatic")
        self.assertEqual(resolved.maximum_risk, "low")
        self.assertFalse(resolved.semantic_change)
        self.assertFalse(resolved.approval_required)
        self.assertEqual(
            resolved.config_patch,
            {
                "trainer.per_device_train_batch_size": 16,
                "trainer.gradient_accumulation_steps": 2,
            },
        )
        self.assertEqual(resolved.environment_patch, {})
        self.assertEqual(
            [change.before for change in resolved.changes],
            [32, 1],
        )
        self.assertEqual(
            [change.after for change in resolved.changes],
            [16, 2],
        )

    def test_resolution_manifest_is_versioned_and_contains_proposal_digest(self):
        proposal = _batch_proposal()
        resolved = resolve_intervention(
            proposal,
            self.manifest,
            self.config,
        )
        payload = json.loads(resolved.to_json())

        self.assertEqual(
            payload["schema"],
            {
                "name": INTERVENTION_RESOLUTION_SCHEMA_NAME,
                "version": INTERVENTION_RESOLUTION_SCHEMA_VERSION,
            },
        )
        self.assertEqual(payload["proposal_digest"], proposal_digest(proposal))
        self.assertEqual(payload["required_permission"], "automatic")

    def test_unsupported_code_dependency_dataset_and_unknown_controls_fail_closed(self):
        for capability_id in (
            "code_change",
            "dependency_change",
            "dataset_change",
            "learning_rate",
        ):
            proposal = InterventionProposal(
                proposal_id="proposal-{}".format(capability_id),
                policy_rule="unsupported_surface_test",
                changes=(
                    InterventionChange(capability_id, "set", "changed"),
                ),
                rationale="Attempt to exercise a non-capability surface.",
                expected_effect="This proposal must never resolve.",
                evidence_refs=("EV-1",),
            )
            with self.subTest(capability_id=capability_id):
                with self.assertRaises(InterventionResolutionError):
                    resolve_intervention(
                        proposal,
                        self.manifest,
                        self.config,
                    )

    def test_invalid_operation_direction_and_value_fail_during_resolution(self):
        invalid = (
            ("micro_batch_size", "increase", 64),
            ("micro_batch_size", "decrease", 32),
            ("micro_batch_size", "decrease", 0),
            ("gradient_checkpointing", "enable", False),
            ("precision", "set", "int4"),
        )
        for capability_id, operation, value in invalid:
            proposal = InterventionProposal(
                proposal_id="proposal-invalid-transition",
                policy_rule="invalid_transition_test",
                changes=(
                    InterventionChange(capability_id, operation, value),
                ),
                rationale="Exercise invalid transition handling.",
                expected_effect="This proposal must never resolve.",
                evidence_refs=("EV-1",),
            )
            with self.subTest(
                capability_id=capability_id,
                operation=operation,
                value=value,
            ):
                with self.assertRaises(InterventionResolutionError):
                    resolve_intervention(
                        proposal,
                        self.manifest,
                        self.config,
                    )

    def test_disabled_capability_never_resolves(self):
        manifest = discover_capabilities(
            self.config,
            declarations=[
                CapabilityDeclaration(
                    "micro_batch_size",
                    "trainer.per_device_train_batch_size",
                    "disabled",
                )
            ],
        )

        with self.assertRaisesRegex(
            InterventionResolutionError,
            "disabled",
        ):
            resolve_intervention(
                _batch_proposal(),
                manifest,
                self.config,
            )

    def test_changed_or_missing_config_baseline_is_stale(self):
        changed = deepcopy(self.config)
        changed["trainer"]["per_device_train_batch_size"] = 24
        with self.assertRaises(StaleInterventionError):
            resolve_intervention(
                _batch_proposal(),
                self.manifest,
                changed,
            )

        missing = deepcopy(self.config)
        del missing["trainer"]["per_device_train_batch_size"]
        with self.assertRaises(StaleInterventionError):
            resolve_intervention(
                _batch_proposal(),
                self.manifest,
                missing,
            )

    def test_resolving_never_mutates_config_manifest_or_proposal(self):
        config_before = deepcopy(self.config)
        manifest_before = self.manifest.to_json()
        proposal = _batch_proposal()
        proposal_before = proposal.to_json()

        resolve_intervention(proposal, self.manifest, self.config)

        self.assertEqual(self.config, config_before)
        self.assertEqual(self.manifest.to_json(), manifest_before)
        self.assertEqual(proposal.to_json(), proposal_before)


class InterventionMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.config = _source_config()
        self.manifest = discover_capabilities(self.config)

    def test_automatic_intervention_preserves_unrelated_source_values(self):
        proposal = _batch_proposal()
        resolved = resolve_intervention(
            proposal,
            self.manifest,
            self.config,
        )
        source_before = deepcopy(self.config)

        application = materialize_intervention(
            resolved,
            self.manifest,
            self.config,
        )

        self.assertEqual(
            application.config["trainer"]["per_device_train_batch_size"],
            16,
        )
        self.assertEqual(
            application.config["trainer"]["gradient_accumulation_steps"],
            2,
        )
        self.assertEqual(
            application.config["trainer"]["learning_rate"],
            0.0002,
        )
        self.assertEqual(
            application.config["dataset"]["fingerprint"],
            "dataset-sha256-example",
        )
        self.assertEqual(application.environment_patch, {})
        self.assertIsNone(application.authorization)
        self.assertEqual(self.config, source_before)
        self.assertIsNot(application.config, self.config)

    def test_semantic_change_requires_exact_digest_bound_authorization(self):
        proposal = _sequence_proposal()
        resolved = resolve_intervention(
            proposal,
            self.manifest,
            self.config,
        )
        self.assertTrue(resolved.approval_required)
        self.assertTrue(resolved.semantic_change)

        with self.assertRaises(InterventionAuthorizationError):
            materialize_intervention(
                resolved,
                self.manifest,
                self.config,
            )

        authorization = InterventionAuthorization.approve(
            proposal,
            approved_by="ml-team@example.com",
            reason="Approve this bounded sequence-length tradeoff.",
            approved_at=1_800_000_000.0,
        )
        application = materialize_intervention(
            resolved,
            self.manifest,
            self.config,
            authorization=authorization,
        )
        self.assertEqual(application.config["model"]["max_seq_length"], 1024)
        self.assertEqual(application.authorization, authorization)

    def test_authorization_for_another_id_or_changed_proposal_cannot_be_replayed(self):
        proposal = _sequence_proposal()
        resolved = resolve_intervention(
            proposal,
            self.manifest,
            self.config,
        )
        wrong_id_authorization = InterventionAuthorization.approve(
            _sequence_proposal("proposal-other"),
            approved_by="reviewer",
            reason="Approval for another proposal.",
            approved_at=1.0,
        )
        with self.assertRaisesRegex(
            InterventionAuthorizationError,
            "proposal_id",
        ):
            materialize_intervention(
                resolved,
                self.manifest,
                self.config,
                authorization=wrong_id_authorization,
            )

        authorization = InterventionAuthorization.approve(
            proposal,
            approved_by="reviewer",
            reason="Approval for the original proposal bytes.",
            approved_at=2.0,
        )
        changed_proposal = replace(
            proposal,
            expected_effect="Changed after approval.",
        )
        changed_resolved = resolve_intervention(
            changed_proposal,
            self.manifest,
            self.config,
        )
        with self.assertRaisesRegex(
            InterventionAuthorizationError,
            "digest",
        ):
            materialize_intervention(
                changed_resolved,
                self.manifest,
                self.config,
                authorization=authorization,
            )

    def test_forged_resolved_target_value_permission_or_risk_is_rejected(self):
        resolved = resolve_intervention(
            _batch_proposal(),
            self.manifest,
            self.config,
        )
        mutations = (
            {"target": "runtime.precision"},
            {"after": 8},
            {"permission": "approval_required"},
            {"risk": "high"},
            {"expected_effect": "Forged effect."},
        )
        for mutation in mutations:
            forged_change = replace(resolved.changes[0], **mutation)
            forged = ResolvedIntervention(
                resolved.proposal,
                (forged_change, resolved.changes[1]),
            )
            with self.subTest(mutation=mutation):
                with self.assertRaises(InterventionResolutionError):
                    materialize_intervention(
                        forged,
                        self.manifest,
                        self.config,
                    )

    def test_changed_baseline_after_resolution_cannot_materialize(self):
        resolved = resolve_intervention(
            _batch_proposal(),
            self.manifest,
            self.config,
        )
        changed = deepcopy(self.config)
        changed["trainer"]["per_device_train_batch_size"] = 24

        with self.assertRaises(StaleInterventionError):
            materialize_intervention(
                resolved,
                self.manifest,
                changed,
            )

    def test_environment_intervention_emits_only_changed_allowlisted_key(self):
        environment = {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            "OPENAI_API_KEY": "must-not-be-copied",
            "UNRELATED": "must-not-be-copied",
        }
        manifest = discover_capabilities(
            self.config,
            environment=environment,
        )
        proposal = InterventionProposal(
            proposal_id="proposal-allocator-1",
            policy_rule="allocator_fragmentation_mitigation",
            changes=(
                InterventionChange(
                    "allocator_configuration",
                    "set",
                    "expandable_segments:True",
                ),
            ),
            rationale=(
                "Captured allocator evidence supports one bounded allocator "
                "configuration trial."
            ),
            expected_effect="Mitigate allocator fragmentation before import.",
            evidence_refs=("EV-6",),
        )
        resolved = resolve_intervention(
            proposal,
            manifest,
            self.config,
            source_environment=environment,
        )
        authorization = InterventionAuthorization.approve(
            proposal,
            approved_by="gpu-owner",
            reason="Approve one allowlisted allocator trial.",
            approved_at=3.0,
        )
        environment_before = deepcopy(environment)

        application = materialize_intervention(
            resolved,
            manifest,
            self.config,
            source_environment=environment,
            authorization=authorization,
        )

        self.assertEqual(
            application.environment_patch,
            {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
        )
        self.assertNotIn("OPENAI_API_KEY", application.environment_patch)
        self.assertNotIn("UNRELATED", application.environment_patch)
        self.assertEqual(environment, environment_before)

    def test_environment_baseline_is_required_and_must_remain_unchanged(self):
        environment = {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        }
        manifest = discover_capabilities(
            self.config,
            environment=environment,
        )
        proposal = InterventionProposal(
            proposal_id="proposal-allocator-stale",
            policy_rule="allocator_fragmentation_mitigation",
            changes=(
                InterventionChange(
                    "allocator_configuration",
                    "set",
                    "expandable_segments:True",
                ),
            ),
            rationale="Use captured allocator evidence for one bounded trial.",
            expected_effect="Mitigate allocator fragmentation.",
            evidence_refs=("EV-6",),
        )

        with self.assertRaises(StaleInterventionError):
            resolve_intervention(proposal, manifest, self.config)

        changed = {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
        }
        with self.assertRaises(StaleInterventionError):
            resolve_intervention(
                proposal,
                manifest,
                self.config,
                source_environment=changed,
            )


class InterventionAuthorizationTests(unittest.TestCase):
    def test_authorization_round_trip_is_versioned_and_exact(self):
        proposal = _sequence_proposal()
        authorization = InterventionAuthorization.approve(
            proposal,
            approved_by="Rohan",
            reason="Approve this exact bounded trial.",
            approved_at=1_800_000_000.25,
        )
        payload = authorization.to_dict()

        self.assertEqual(
            payload["schema"],
            {
                "name": INTERVENTION_AUTHORIZATION_SCHEMA_NAME,
                "version": INTERVENTION_AUTHORIZATION_SCHEMA_VERSION,
            },
        )
        self.assertEqual(payload["proposal_digest"], proposal_digest(proposal))
        self.assertEqual(
            InterventionAuthorization.from_dict(payload),
            authorization,
        )

    def test_authorization_rejects_invalid_digest_timestamp_and_unknown_fields(self):
        proposal = _sequence_proposal()
        valid = InterventionAuthorization.approve(
            proposal,
            approved_by="reviewer",
            reason="Approve this exact proposal.",
            approved_at=1.0,
        ).to_dict()

        mutations = (
            ("proposal_digest", "not-a-digest"),
            ("proposal_digest", "A" * 64),
            ("approved_at", 0),
            ("approved_at", -1),
            ("approved_at", float("nan")),
            ("approved_at", True),
            ("shell_command", "forbidden"),
        )
        for field, value in mutations:
            damaged = deepcopy(valid)
            damaged[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(InterventionError):
                    InterventionAuthorization.from_dict(damaged)

    def test_authorization_rejects_schema_changes_and_empty_identity_text(self):
        proposal = _sequence_proposal()
        valid = InterventionAuthorization.approve(
            proposal,
            approved_by="reviewer",
            reason="Approve this exact proposal.",
            approved_at=1.0,
        ).to_dict()

        for field, value in (("name", "other"), ("version", "99.0")):
            damaged = deepcopy(valid)
            damaged["schema"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(InterventionError):
                    InterventionAuthorization.from_dict(damaged)

        with self.assertRaises(InterventionError):
            InterventionAuthorization.approve(
                proposal,
                approved_by=" ",
                reason="Valid reason.",
                approved_at=1.0,
            )
        with self.assertRaises(InterventionError):
            InterventionAuthorization.approve(
                proposal,
                approved_by="reviewer",
                reason=" ",
                approved_at=1.0,
            )


if __name__ == "__main__":
    unittest.main()