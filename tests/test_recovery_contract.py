"""Acceptance tests for WatcherML's versioned OOM recovery contract."""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

from watcherml.entrypoint import TrainingEntrypoint
from watcherml.interventions import (
    InterventionChange,
    InterventionProposal,
    ResolvedChange,
    ResolvedIntervention,
)
from watcherml.recovery_contract import (
    HARD_MAX_CONFIRMATION_RUNS,
    HARD_MAX_GPU_SECONDS,
    HARD_MAX_TIMEOUT_SECONDS,
    HARD_MAX_TRIALS,
    RECOVERY_CONTRACT_SCHEMA_NAME,
    RECOVERY_CONTRACT_SCHEMA_VERSION,
    ContractScopeError,
    InterventionPermissions,
    MetricGuard,
    RecoveryBudget,
    RecoveryContract,
    RecoveryContractError,
    VerificationRequirements,
    WorkloadIdentity,
    contract_digest,
    validate_intervention_scope,
)


def _source_config() -> dict:
    return {
        "trainer": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
        },
        "model": {
            "gradient_checkpointing": False,
            "max_seq_length": 2048,
            "name": "example/model",
        },
        "runtime": {"precision": "fp32"},
        "dataset": {"fingerprint": "dataset-example"},
    }


def _budget(**overrides) -> RecoveryBudget:
    values = {
        "max_trials": 9,
        "max_probe_trials": 5,
        "max_full_trials": 2,
        "probe_steps": 30,
        "trial_timeout_seconds": 1_800,
        "campaign_timeout_seconds": 7_200,
        "max_gpu_seconds": 5_400,
    }
    values.update(overrides)
    return RecoveryBudget(**values)


def _verification(**overrides) -> VerificationRequirements:
    values = {
        "minimum_progress_steps": 500,
        "metric_guards": (
            MetricGuard(
                name="validation_loss",
                direction="minimize",
                baseline_value=0.40,
                max_regression=0.05,
            ),
            MetricGuard(
                name="validation_accuracy",
                direction="maximize",
                baseline_value=0.82,
                max_regression=0.02,
                target_value=0.81,
            ),
        ),
        "confirmation_runs": 2,
        "max_peak_vram_bytes": 15 * 1024**3,
        "workload_identity": WorkloadIdentity(
            dataset_fingerprint="dataset-example",
            environment_fingerprint="environment-example",
            git_commit="abc123",
            model_identifier="example/model",
        ),
    }
    values.update(overrides)
    return VerificationRequirements(**values)


def _contract(
    *,
    source_config=None,
    budget=None,
    verification=None,
    permissions=None,
) -> RecoveryContract:
    return RecoveryContract(
        project="contract-tests",
        source_run_id="source-oom-run",
        entrypoint=TrainingEntrypoint(
            target="training.entrypoint:train",
            working_directory="training",
        ),
        source_config=(
            _source_config() if source_config is None else source_config
        ),
        budget=_budget() if budget is None else budget,
        verification=(
            _verification() if verification is None else verification
        ),
        permissions=permissions,
    )


def _resolved_intervention(
    *,
    permission: str = "automatic",
    risk: str = "low",
    semantic_change: bool = False,
) -> ResolvedIntervention:
    proposal = InterventionProposal(
        proposal_id="proposal-contract-scope",
        policy_rule="halve_batch_preserve_effective_batch",
        changes=(
            InterventionChange(
                capability_id="micro_batch_size",
                operation="decrease",
                proposed_value=16,
            ),
        ),
        rationale="The OOM capsule recorded a batch size of 32.",
        expected_effect="Reduce per-step activation memory.",
        evidence_refs=("EV-1", "EV-4"),
    )
    return ResolvedIntervention(
        proposal=proposal,
        changes=(
            ResolvedChange(
                capability_id="micro_batch_size",
                operation="decrease",
                location="config",
                target="trainer.per_device_train_batch_size",
                before=32,
                after=16,
                permission=permission,
                risk=risk,
                semantic_change=semantic_change,
                expected_effect="Reduce per-step activation memory.",
            ),
        ),
    )


class MetricGuardTests(unittest.TestCase):
    def test_maximize_regression_boundary_is_a_floor(self):
        guard = MetricGuard("accuracy", "maximize", 0.80, 0.03)

        self.assertAlmostEqual(guard.regression_boundary, 0.77)
        self.assertAlmostEqual(guard.acceptance_boundary, 0.77)

    def test_minimize_regression_boundary_is_a_ceiling(self):
        guard = MetricGuard("validation_loss", "minimize", 0.40, 0.05)

        self.assertAlmostEqual(guard.regression_boundary, 0.45)
        self.assertAlmostEqual(guard.acceptance_boundary, 0.45)

    def test_absolute_target_can_only_make_acceptance_stricter(self):
        maximize = MetricGuard("accuracy", "maximize", 0.80, 0.05, 0.78)
        minimize = MetricGuard("loss", "minimize", 0.40, 0.10, 0.45)

        self.assertAlmostEqual(maximize.acceptance_boundary, 0.78)
        self.assertAlmostEqual(minimize.acceptance_boundary, 0.45)

    def test_weaker_absolute_target_does_not_relax_regression_limit(self):
        maximize = MetricGuard("accuracy", "maximize", 0.80, 0.02, 0.70)
        minimize = MetricGuard("loss", "minimize", 0.40, 0.03, 0.80)

        self.assertAlmostEqual(maximize.acceptance_boundary, 0.78)
        self.assertAlmostEqual(minimize.acceptance_boundary, 0.43)

    def test_round_trip_preserves_guard(self):
        guard = MetricGuard("eval/loss", "minimize", 0.4, 0.05, 0.42)

        self.assertEqual(MetricGuard.from_dict(guard.to_dict()), guard)

    def test_invalid_metric_names_and_directions_are_rejected(self):
        for name, direction in (
            ("", "maximize"),
            ("accuracy with spaces", "maximize"),
            ("1accuracy", "maximize"),
            ("accuracy", "higher_is_better"),
        ):
            with self.subTest(name=name, direction=direction):
                with self.assertRaises(RecoveryContractError):
                    MetricGuard(name, direction, 0.8, 0.01)

    def test_nonfinite_boolean_and_negative_numbers_are_rejected(self):
        invalid = (
            {"baseline_value": float("nan")},
            {"baseline_value": float("inf")},
            {"baseline_value": True},
            {"max_regression": -0.01},
            {"max_regression": float("nan")},
            {"target_value": float("inf")},
        )
        defaults = {
            "name": "accuracy",
            "direction": "maximize",
            "baseline_value": 0.8,
            "max_regression": 0.01,
        }
        for override in invalid:
            with self.subTest(override=override):
                values = dict(defaults)
                values.update(override)
                with self.assertRaises(RecoveryContractError):
                    MetricGuard(**values)

    def test_deserialization_rejects_missing_and_unknown_fields(self):
        guard = MetricGuard("accuracy", "maximize", 0.8, 0.01)
        unknown = guard.to_dict()
        unknown["comparison_code"] = "arbitrary"

        with self.assertRaisesRegex(RecoveryContractError, "unknown fields"):
            MetricGuard.from_dict(unknown)
        with self.assertRaisesRegex(RecoveryContractError, "missing"):
            MetricGuard.from_dict({"name": "accuracy"})


class WorkloadIdentityTests(unittest.TestCase):
    def test_known_fields_are_explicit_and_ordered(self):
        identity = WorkloadIdentity(
            dataset_fingerprint="  dataset-sha  ",
            git_commit="abc123",
        )

        self.assertEqual(identity.dataset_fingerprint, "dataset-sha")
        self.assertEqual(
            identity.known_fields,
            ("dataset_fingerprint", "git_commit"),
        )

    def test_unknown_identity_values_remain_null(self):
        payload = WorkloadIdentity().to_dict()

        self.assertTrue(all(value is None for value in payload.values()))
        self.assertEqual(WorkloadIdentity().known_fields, ())

    def test_identity_round_trip_preserves_values(self):
        identity = _verification().workload_identity

        self.assertEqual(
            WorkloadIdentity.from_dict(identity.to_dict()),
            identity,
        )

    def test_identity_rejects_blank_non_string_and_unknown_fields(self):
        for value in ("", "   ", 123):
            with self.subTest(value=value):
                with self.assertRaises(RecoveryContractError):
                    WorkloadIdentity(dataset_fingerprint=value)

        with self.assertRaisesRegex(RecoveryContractError, "unknown fields"):
            WorkloadIdentity.from_dict({"dataset_fingerprint": "x", "seed": 7})


class RecoveryBudgetTests(unittest.TestCase):
    def test_budget_round_trip_and_numeric_normalization(self):
        budget = _budget()
        restored = RecoveryBudget.from_dict(budget.to_dict())

        self.assertEqual(restored, budget)
        self.assertIsInstance(restored.trial_timeout_seconds, float)
        self.assertIsInstance(restored.campaign_timeout_seconds, float)

    def test_global_trial_hard_cap_cannot_be_exceeded(self):
        with self.assertRaises(RecoveryContractError):
            _budget(max_trials=HARD_MAX_TRIALS + 1)

    def test_phase_trial_limits_cannot_exceed_total(self):
        with self.assertRaisesRegex(RecoveryContractError, "max_probe_trials"):
            _budget(max_trials=5, max_probe_trials=6, max_full_trials=1)
        with self.assertRaisesRegex(RecoveryContractError, "max_full_trials"):
            _budget(max_trials=5, max_probe_trials=1, max_full_trials=6)

    def test_boolean_zero_negative_and_float_counts_are_rejected(self):
        for field_name in (
            "max_trials",
            "max_probe_trials",
            "max_full_trials",
            "probe_steps",
        ):
            for value in (True, 0, -1, 1.5):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(RecoveryContractError):
                        _budget(**{field_name: value})

    def test_trial_timeout_cannot_exceed_campaign_timeout(self):
        with self.assertRaisesRegex(RecoveryContractError, "cannot exceed"):
            _budget(
                trial_timeout_seconds=601,
                campaign_timeout_seconds=600,
            )

    def test_timeout_and_gpu_limits_must_be_positive_finite_and_bounded(self):
        invalid = (
            ("trial_timeout_seconds", 0),
            ("campaign_timeout_seconds", float("nan")),
            ("trial_timeout_seconds", HARD_MAX_TIMEOUT_SECONDS + 1),
            ("max_gpu_seconds", -1),
            ("max_gpu_seconds", float("inf")),
            ("max_gpu_seconds", HARD_MAX_GPU_SECONDS + 1),
        )
        for field_name, value in invalid:
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(RecoveryContractError):
                    _budget(**{field_name: value})

    def test_deserialization_rejects_missing_unknown_and_non_object_payloads(self):
        payload = _budget().to_dict()
        del payload["probe_steps"]
        with self.assertRaisesRegex(RecoveryContractError, "missing"):
            RecoveryBudget.from_dict(payload)

        payload = _budget().to_dict()
        payload["money_budget"] = 100
        with self.assertRaisesRegex(RecoveryContractError, "unknown fields"):
            RecoveryBudget.from_dict(payload)

        with self.assertRaisesRegex(RecoveryContractError, "object"):
            RecoveryBudget.from_dict([])


class VerificationRequirementsTests(unittest.TestCase):
    def test_round_trip_preserves_guards_identity_and_limits(self):
        requirements = _verification()

        self.assertEqual(
            VerificationRequirements.from_dict(requirements.to_dict()),
            requirements,
        )

    def test_at_least_one_metric_guard_is_required(self):
        with self.assertRaisesRegex(RecoveryContractError, "at least one"):
            _verification(metric_guards=())

    def test_metric_guard_names_must_be_unique(self):
        duplicate = MetricGuard("loss", "minimize", 0.4, 0.05)
        with self.assertRaisesRegex(RecoveryContractError, "unique"):
            _verification(metric_guards=(duplicate, duplicate))

    def test_metric_guard_collection_must_be_typed_and_iterable(self):
        for value in (None, ("loss",)):
            with self.subTest(value=value):
                with self.assertRaises(RecoveryContractError):
                    _verification(metric_guards=value)

    def test_confirmation_runs_have_a_small_hard_cap(self):
        with self.assertRaises(RecoveryContractError):
            _verification(
                confirmation_runs=HARD_MAX_CONFIRMATION_RUNS + 1
            )

    def test_progress_vram_and_confirmation_counts_reject_booleans_and_zero(self):
        invalid = (
            {"minimum_progress_steps": 0},
            {"minimum_progress_steps": True},
            {"confirmation_runs": 0},
            {"confirmation_runs": True},
            {"max_peak_vram_bytes": 0},
            {"max_peak_vram_bytes": True},
        )
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(RecoveryContractError):
                    _verification(**override)

    def test_workload_identity_must_use_the_declared_type(self):
        with self.assertRaisesRegex(RecoveryContractError, "WorkloadIdentity"):
            _verification(workload_identity={"dataset_fingerprint": "x"})

    def test_deserialization_requires_arrays_and_rejects_unknown_fields(self):
        payload = _verification().to_dict()
        payload["metric_guards"] = {"loss": 0.4}
        with self.assertRaisesRegex(RecoveryContractError, "array"):
            VerificationRequirements.from_dict(payload)

        payload = _verification().to_dict()
        payload["accept_any_success"] = True
        with self.assertRaisesRegex(RecoveryContractError, "unknown fields"):
            VerificationRequirements.from_dict(payload)


class InterventionPermissionTests(unittest.TestCase):
    def test_default_permissions_allow_only_automatic_nonsemantic_changes(self):
        self.assertEqual(
            InterventionPermissions(),
            InterventionPermissions(False, False, False),
        )

    def test_semantic_or_high_risk_authority_requires_approval_authority(self):
        with self.assertRaisesRegex(RecoveryContractError, "semantic"):
            InterventionPermissions(allow_semantic_changes=True)
        with self.assertRaisesRegex(RecoveryContractError, "high-risk"):
            InterventionPermissions(allow_high_risk=True)

    def test_permissions_must_be_real_booleans(self):
        for field_name in (
            "allow_approval_required",
            "allow_semantic_changes",
            "allow_high_risk",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(RecoveryContractError):
                    InterventionPermissions(**{field_name: 1})

    def test_permissions_round_trip(self):
        permissions = InterventionPermissions(True, True, True)

        self.assertEqual(
            InterventionPermissions.from_dict(permissions.to_dict()),
            permissions,
        )

    def test_permissions_deserialization_is_strict(self):
        payload = InterventionPermissions().to_dict()
        del payload["allow_high_risk"]
        with self.assertRaisesRegex(RecoveryContractError, "missing"):
            InterventionPermissions.from_dict(payload)

        payload = InterventionPermissions().to_dict()
        payload["allow_shell_commands"] = True
        with self.assertRaisesRegex(RecoveryContractError, "unknown fields"):
            InterventionPermissions.from_dict(payload)


class RecoveryContractArtifactTests(unittest.TestCase):
    def test_contract_round_trip_preserves_canonical_json_and_digest(self):
        contract = _contract()
        encoded = contract.to_json()
        restored = RecoveryContract.from_json(encoded)

        self.assertEqual(restored, contract)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(contract_digest(restored), contract_digest(contract))
        self.assertRegex(contract_digest(contract), r"^[a-f0-9]{64}$")

        payload = json.loads(encoded)
        self.assertEqual(
            payload["schema"],
            {
                "name": RECOVERY_CONTRACT_SCHEMA_NAME,
                "version": RECOVERY_CONTRACT_SCHEMA_VERSION,
            },
        )

    def test_contract_records_fixed_recovery_safety_invariants(self):
        self.assertEqual(
            _contract().to_dict()["invariants"],
            {
                "failure_class": "cuda_out_of_memory",
                "fresh_process": True,
                "source_config_immutable": True,
                "no_oom_required": True,
                "confirmation_required": True,
            },
        )

    def test_source_config_is_deeply_sealed_from_original_and_returned_copies(self):
        original = _source_config()
        contract = _contract(source_config=original)
        original["trainer"]["per_device_train_batch_size"] = 1

        returned = contract.source_config
        returned["model"]["max_seq_length"] = 128
        serialized = contract.to_dict()
        serialized["source_config"]["runtime"]["precision"] = "bf16"

        self.assertEqual(
            contract.source_config["trainer"]["per_device_train_batch_size"],
            32,
        )
        self.assertEqual(contract.source_config["model"]["max_seq_length"], 2048)
        self.assertEqual(contract.source_config["runtime"]["precision"], "fp32")

    def test_contract_fields_are_frozen(self):
        contract = _contract()

        with self.assertRaises(FrozenInstanceError):
            contract.project = "changed"

    def test_reserved_trials_include_probe_full_and_confirmation_runs(self):
        contract = _contract()

        self.assertEqual(contract.reserved_trials, 9)

    def test_reservations_cannot_exceed_global_trial_budget(self):
        with self.assertRaisesRegex(RecoveryContractError, "reservations"):
            _contract(
                budget=_budget(
                    max_trials=8,
                    max_probe_trials=5,
                    max_full_trials=2,
                ),
                verification=_verification(confirmation_runs=2),
            )

    def test_probe_steps_cannot_exceed_required_full_progress(self):
        with self.assertRaisesRegex(RecoveryContractError, "probe_steps"):
            _contract(
                budget=_budget(probe_steps=501),
                verification=_verification(minimum_progress_steps=500),
            )

    def test_default_permissions_are_inserted_explicitly(self):
        contract = _contract(permissions=None)

        self.assertEqual(contract.permissions, InterventionPermissions())
        self.assertEqual(
            contract.to_dict()["permissions"],
            InterventionPermissions().to_dict(),
        )

    def test_digest_changes_when_contract_authority_or_baseline_changes(self):
        baseline = _contract()
        changed_config = _source_config()
        changed_config["trainer"]["per_device_train_batch_size"] = 16
        changed = _contract(source_config=changed_config)
        permissive = _contract(
            permissions=InterventionPermissions(True, True, True)
        )

        self.assertNotEqual(contract_digest(baseline), contract_digest(changed))
        self.assertNotEqual(contract_digest(baseline), contract_digest(permissive))

    def test_deserialization_rejects_modified_safety_invariants(self):
        invariant_names = tuple(_contract().to_dict()["invariants"])
        for name in invariant_names:
            with self.subTest(name=name):
                payload = _contract().to_dict()
                payload["invariants"][name] = False
                with self.assertRaisesRegex(
                    RecoveryContractError,
                    "invariants",
                ):
                    RecoveryContract.from_dict(payload)

    def test_deserialization_rejects_wrong_or_extended_schema(self):
        invalid_schemas = (
            {"name": "other", "version": RECOVERY_CONTRACT_SCHEMA_VERSION},
            {"name": RECOVERY_CONTRACT_SCHEMA_NAME, "version": "2.0"},
            {
                "name": RECOVERY_CONTRACT_SCHEMA_NAME,
                "version": RECOVERY_CONTRACT_SCHEMA_VERSION,
                "unsafe_compatibility": True,
            },
            "not-an-object",
        )
        for schema in invalid_schemas:
            with self.subTest(schema=schema):
                payload = _contract().to_dict()
                payload["schema"] = schema
                with self.assertRaises(RecoveryContractError):
                    RecoveryContract.from_dict(payload)

    def test_deserialization_rejects_unknown_top_level_fields(self):
        payload = _contract().to_dict()
        payload["skip_confirmation"] = True

        with self.assertRaisesRegex(RecoveryContractError, "unknown fields"):
            RecoveryContract.from_dict(payload)

    def test_invalid_json_and_non_object_json_are_rejected(self):
        for encoded in ("{not-json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(RecoveryContractError):
                    RecoveryContract.from_json(encoded)

    def test_contract_rejects_invalid_identity_and_component_types(self):
        valid = {
            "project": "contract-tests",
            "source_run_id": "source-oom-run",
            "entrypoint": TrainingEntrypoint("training:train"),
            "source_config": _source_config(),
            "budget": _budget(),
            "verification": _verification(),
        }
        invalid = (
            {"project": ""},
            {"source_run_id": "contains spaces"},
            {"entrypoint": "training:train"},
            {"source_config": {"value": float("nan")}},
            {"budget": {}},
            {"verification": {}},
            {"permissions": {}},
        )
        for override in invalid:
            with self.subTest(override=override):
                values = dict(valid)
                values.update(override)
                with self.assertRaises(RecoveryContractError):
                    RecoveryContract(**values)


class InterventionScopeTests(unittest.TestCase):
    def test_default_contract_allows_automatic_low_risk_nonsemantic_change(self):
        validate_intervention_scope(_contract(), _resolved_intervention())

    def test_default_contract_rejects_approval_required_change(self):
        intervention = _resolved_intervention(
            permission="approval_required",
            risk="medium",
        )

        with self.assertRaisesRegex(ContractScopeError, "approval-required"):
            validate_intervention_scope(_contract(), intervention)

    def test_approval_permission_does_not_implicitly_allow_semantic_change(self):
        contract = _contract(
            permissions=InterventionPermissions(
                allow_approval_required=True,
            )
        )
        intervention = _resolved_intervention(
            permission="approval_required",
            risk="medium",
            semantic_change=True,
        )

        with self.assertRaisesRegex(ContractScopeError, "semantic"):
            validate_intervention_scope(contract, intervention)

    def test_approval_permission_does_not_implicitly_allow_high_risk_change(self):
        contract = _contract(
            permissions=InterventionPermissions(
                allow_approval_required=True,
            )
        )
        intervention = _resolved_intervention(
            permission="approval_required",
            risk="high",
        )

        with self.assertRaisesRegex(ContractScopeError, "high-risk"):
            validate_intervention_scope(contract, intervention)

    def test_explicit_full_scope_allows_gated_semantic_high_risk_change(self):
        contract = _contract(
            permissions=InterventionPermissions(
                allow_approval_required=True,
                allow_semantic_changes=True,
                allow_high_risk=True,
            )
        )
        intervention = _resolved_intervention(
            permission="approval_required",
            risk="high",
            semantic_change=True,
        )

        validate_intervention_scope(contract, intervention)

    def test_scope_validation_rejects_wrong_argument_types(self):
        with self.assertRaisesRegex(ContractScopeError, "contract"):
            validate_intervention_scope({}, _resolved_intervention())
        with self.assertRaisesRegex(ContractScopeError, "intervention"):
            validate_intervention_scope(_contract(), {})


if __name__ == "__main__":
    unittest.main()