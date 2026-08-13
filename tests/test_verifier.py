"""Acceptance tests for deterministic WatcherML recovery verification."""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

from watcherml.entrypoint import TrainingEntrypoint
from watcherml.recovery_contract import (
    MetricGuard,
    RecoveryBudget,
    RecoveryContract,
    VerificationRequirements,
    WorkloadIdentity,
    contract_digest,
)
from watcherml.verifier import (
    CHECK_OUTCOMES,
    CONFIRMATION_EVIDENCE_SCHEMA_NAME,
    CONFIRMATION_EVIDENCE_SCHEMA_VERSION,
    MAX_CONFIRMATION_EVIDENCE,
    VERIFICATION_REPORT_SCHEMA_NAME,
    VERIFICATION_REPORT_SCHEMA_VERSION,
    VERIFICATION_VERDICTS,
    ConfirmationEvidence,
    RecoveryVerification,
    VerificationCheck,
    VerificationError,
    configuration_digest,
    evidence_digest,
    verification_digest,
    verify_recovery,
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
    }


def _candidate_config() -> dict:
    return {
        "trainer": {
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 2,
        },
        "model": {
            "gradient_checkpointing": False,
            "max_seq_length": 2048,
            "name": "example/model",
        },
        "runtime": {"precision": "fp32"},
    }


def _identity() -> WorkloadIdentity:
    return WorkloadIdentity(
        dataset_fingerprint="dataset-example",
        environment_fingerprint="environment-example",
        git_commit="abc123",
        model_identifier="example/model",
    )


def _contract(
    *,
    confirmation_runs: int = 2,
    max_peak_vram_bytes=15 * 1024**3,
    workload_identity=None,
    metric_guards=None,
) -> RecoveryContract:
    if workload_identity is None:
        workload_identity = _identity()
    if metric_guards is None:
        metric_guards = (
            MetricGuard(
                "validation_loss",
                "minimize",
                baseline_value=0.40,
                max_regression=0.05,
            ),
            MetricGuard(
                "validation_accuracy",
                "maximize",
                baseline_value=0.80,
                max_regression=0.02,
                target_value=0.79,
            ),
        )
    return RecoveryContract(
        project="verifier-tests",
        source_run_id="source-oom-run",
        entrypoint=TrainingEntrypoint("training.entrypoint:train"),
        source_config=_source_config(),
        budget=RecoveryBudget(
            max_trials=10,
            max_probe_trials=5,
            max_full_trials=2,
            probe_steps=30,
            trial_timeout_seconds=1_800,
            campaign_timeout_seconds=7_200,
            max_gpu_seconds=5_400,
        ),
        verification=VerificationRequirements(
            minimum_progress_steps=500,
            metric_guards=metric_guards,
            confirmation_runs=confirmation_runs,
            max_peak_vram_bytes=max_peak_vram_bytes,
            workload_identity=workload_identity,
        ),
    )


def _evidence(
    contract: RecoveryContract,
    index: int,
    *,
    candidate_config=None,
    **overrides
) -> ConfirmationEvidence:
    if candidate_config is None:
        candidate_config = _candidate_config()
    values = {
        "campaign_id": "campaign-1",
        "candidate_id": "candidate-1",
        "trial_id": "confirmation-trial-{}".format(index),
        "run_id": "confirmation-run-{}".format(index),
        "project": contract.project,
        "source_run_id": contract.source_run_id,
        "contract_digest": contract_digest(contract),
        "candidate_config_digest": configuration_digest(candidate_config),
        "trial_request_digest": "{:064x}".format(100 + index),
        "execution_manifest_digest": "{:064x}".format(200 + index),
        "phase": "confirmation",
        "status": "success",
        "metrics": {
            "validation_loss": 0.44,
            "validation_accuracy": 0.80,
            "samples_per_second": 12.0,
        },
        "progress_steps": 500,
        "peak_vram_bytes": 14 * 1024**3,
        "workload_identity": _identity(),
        "worker_pid": 10_000 + index,
        "failure_class": None,
    }
    values.update(overrides)
    return ConfirmationEvidence(**values)


def _verified_report(contract=None, confirmations=None) -> RecoveryVerification:
    contract = _contract() if contract is None else contract
    if confirmations is None:
        confirmations = (_evidence(contract, 1), _evidence(contract, 2))
    return verify_recovery(
        contract,
        campaign_id="campaign-1",
        candidate_id="candidate-1",
        candidate_config=_candidate_config(),
        confirmations=confirmations,
    )


def _find_check(report: RecoveryVerification, code: str, run_id=None):
    matches = [
        check
        for check in report.checks
        if check.code == code and (run_id is None or check.run_id == run_id)
    ]
    if len(matches) != 1:
        raise AssertionError(
            "expected exactly one check for {!r}/{!r}, found {}".format(
                code,
                run_id,
                len(matches),
            )
        )
    return matches[0]


class ConfirmationEvidenceArtifactTests(unittest.TestCase):
    def setUp(self):
        self.contract = _contract()
        self.evidence = _evidence(self.contract, 1)

    def test_round_trip_preserves_canonical_json_and_digest(self):
        encoded = self.evidence.to_json()
        restored = ConfirmationEvidence.from_json(encoded)

        self.assertEqual(restored, self.evidence)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(evidence_digest(restored), evidence_digest(self.evidence))
        self.assertRegex(evidence_digest(self.evidence), r"^[a-f0-9]{64}$")
        self.assertEqual(
            json.loads(encoded)["schema"],
            {
                "name": CONFIRMATION_EVIDENCE_SCHEMA_NAME,
                "version": CONFIRMATION_EVIDENCE_SCHEMA_VERSION,
            },
        )

    def test_digest_changes_when_observed_evidence_changes(self):
        changed = _evidence(
            self.contract,
            1,
            metrics={"validation_loss": 0.45, "validation_accuracy": 0.80},
        )

        self.assertNotEqual(evidence_digest(self.evidence), evidence_digest(changed))

    def test_metrics_are_normalized_and_deeply_immutable(self):
        original = {"validation_loss": 1, "validation_accuracy": 0.8}
        evidence = _evidence(self.contract, 1, metrics=original)
        original["validation_loss"] = 99

        self.assertEqual(evidence.metrics["validation_loss"], 1.0)
        with self.assertRaises(TypeError):
            evidence.metrics["validation_loss"] = 0.0
        exported = evidence.to_dict()
        exported["metrics"]["validation_loss"] = 7.0
        self.assertEqual(evidence.metrics["validation_loss"], 1.0)

    def test_dataclass_fields_are_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            self.evidence.status = "training_failed"

    def test_project_and_failure_class_are_trimmed(self):
        evidence = _evidence(
            self.contract,
            1,
            project="  verifier-tests  ",
            status="training_failed",
            failure_class="  cuda_out_of_memory  ",
        )

        self.assertEqual(evidence.project, "verifier-tests")
        self.assertEqual(evidence.failure_class, "cuda_out_of_memory")

    def test_optional_runtime_observations_may_remain_missing(self):
        evidence = _evidence(
            self.contract,
            1,
            progress_steps=None,
            peak_vram_bytes=None,
            worker_pid=None,
        )

        self.assertIsNone(evidence.progress_steps)
        self.assertIsNone(evidence.peak_vram_bytes)
        self.assertIsNone(evidence.worker_pid)

    def test_invalid_identifiers_are_rejected(self):
        for field_name in (
            "campaign_id",
            "candidate_id",
            "trial_id",
            "run_id",
            "source_run_id",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(VerificationError):
                    _evidence(self.contract, 1, **{field_name: "contains spaces"})

    def test_all_artifact_digests_must_be_lowercase_sha256(self):
        for field_name in (
            "contract_digest",
            "candidate_config_digest",
            "trial_request_digest",
            "execution_manifest_digest",
        ):
            for value in ("abc", "A" * 64, "g" * 64):
                with self.subTest(field_name=field_name, value=value[:4]):
                    with self.assertRaises(VerificationError):
                        _evidence(self.contract, 1, **{field_name: value})

    def test_invalid_phase_and_status_are_rejected(self):
        with self.assertRaisesRegex(VerificationError, "phase"):
            _evidence(self.contract, 1, phase="verification")
        with self.assertRaisesRegex(VerificationError, "status"):
            _evidence(self.contract, 1, status="completed")

    def test_metric_values_must_be_finite_numbers(self):
        invalid = (
            {"loss": True},
            {"loss": "0.4"},
            {"loss": float("nan")},
            {"loss": float("inf")},
            {"": 0.4},
        )
        for metrics in invalid:
            with self.subTest(metrics=metrics):
                with self.assertRaises(VerificationError):
                    _evidence(self.contract, 1, metrics=metrics)

    def test_progress_and_vram_must_be_nonnegative_integers(self):
        for field_name in ("progress_steps", "peak_vram_bytes"):
            for value in (True, -1, 1.5):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(VerificationError):
                        _evidence(self.contract, 1, **{field_name: value})

    def test_worker_pid_must_be_a_positive_integer(self):
        for value in (True, 0, -1, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(VerificationError):
                    _evidence(self.contract, 1, worker_pid=value)

    def test_workload_identity_must_use_declared_type(self):
        with self.assertRaisesRegex(VerificationError, "WorkloadIdentity"):
            _evidence(
                self.contract,
                1,
                workload_identity={"dataset_fingerprint": "dataset-example"},
            )

    def test_blank_or_oversized_failure_class_is_rejected(self):
        for value in ("", "   ", "x" * 4_001):
            with self.subTest(length=len(value)):
                with self.assertRaises(VerificationError):
                    _evidence(self.contract, 1, failure_class=value)

    def test_deserialization_rejects_missing_unknown_and_modified_schema(self):
        missing = self.evidence.to_dict()
        del missing["trial_request_digest"]
        with self.assertRaisesRegex(VerificationError, "missing"):
            ConfirmationEvidence.from_dict(missing)

        unknown = self.evidence.to_dict()
        unknown["ranking_score"] = 999
        with self.assertRaisesRegex(VerificationError, "unknown fields"):
            ConfirmationEvidence.from_dict(unknown)

        wrong_schema = self.evidence.to_dict()
        wrong_schema["schema"]["version"] = "2.0"
        with self.assertRaisesRegex(VerificationError, "schema.version"):
            ConfirmationEvidence.from_dict(wrong_schema)

    def test_invalid_json_and_wrong_digest_helper_type_are_rejected(self):
        for encoded in ("{not-json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(VerificationError):
                    ConfirmationEvidence.from_json(encoded)
        with self.assertRaises(VerificationError):
            evidence_digest({})


class VerificationCheckTests(unittest.TestCase):
    def test_round_trip_and_passed_property(self):
        check = VerificationCheck(
            code="metric.validation_loss",
            outcome="pass",
            message="Metric remained inside its declared boundary.",
            expected={"maximum": 0.45},
            observed=0.44,
            run_id="run-1",
        )

        self.assertTrue(check.passed)
        self.assertEqual(VerificationCheck.from_dict(check.to_dict()), check)

    def test_expected_and_observed_payloads_are_deeply_immutable(self):
        expected = {"limits": [{"maximum": 0.45}]}
        check = VerificationCheck(
            "metric.loss",
            "pass",
            "Metric passed.",
            expected,
            {"value": [0.44]},
        )
        expected["limits"][0]["maximum"] = 99

        self.assertEqual(check.expected["limits"][0]["maximum"], 0.45)
        with self.assertRaises(TypeError):
            check.expected["limits"][0]["maximum"] = 0.0
        exported = check.to_dict()
        exported["expected"]["limits"][0]["maximum"] = 8
        self.assertEqual(check.expected["limits"][0]["maximum"], 0.45)

    def test_invalid_code_outcome_message_and_run_id_are_rejected(self):
        invalid = (
            {"code": "Bad Code"},
            {"outcome": "unknown"},
            {"message": ""},
            {"run_id": "bad run id"},
        )
        defaults = {
            "code": "execution.status",
            "outcome": "pass",
            "message": "Execution passed.",
            "expected": "success",
            "observed": "success",
            "run_id": "run-1",
        }
        for override in invalid:
            with self.subTest(override=override):
                values = dict(defaults)
                values.update(override)
                with self.assertRaises(VerificationError):
                    VerificationCheck(**values)

    def test_non_json_or_nonfinite_payloads_are_rejected(self):
        for value in ({1, 2}, object(), float("nan"), float("inf")):
            with self.subTest(value=repr(value)):
                with self.assertRaises(VerificationError):
                    VerificationCheck(
                        "metric.loss",
                        "pass",
                        "Metric passed.",
                        value,
                        0.4,
                    )

    def test_deserialization_rejects_missing_unknown_and_non_object(self):
        check = VerificationCheck(
            "execution.status", "pass", "Execution passed.", "success", "success"
        )
        missing = check.to_dict()
        del missing["observed"]
        with self.assertRaisesRegex(VerificationError, "missing"):
            VerificationCheck.from_dict(missing)

        unknown = check.to_dict()
        unknown["override"] = True
        with self.assertRaisesRegex(VerificationError, "unknown fields"):
            VerificationCheck.from_dict(unknown)

        with self.assertRaisesRegex(VerificationError, "object"):
            VerificationCheck.from_dict([])


class DeterministicVerificationDecisionTests(unittest.TestCase):
    def setUp(self):
        self.contract = _contract()
        self.first = _evidence(self.contract, 1)
        self.second = _evidence(self.contract, 2)

    def verify(self, confirmations=None, contract=None, **kwargs):
        contract = self.contract if contract is None else contract
        if confirmations is None:
            confirmations = (self.first, self.second)
        values = {
            "campaign_id": "campaign-1",
            "candidate_id": "candidate-1",
            "candidate_config": _candidate_config(),
            "confirmations": confirmations,
        }
        values.update(kwargs)
        return verify_recovery(contract, **values)

    def test_complete_confirmation_set_is_verified(self):
        report = self.verify()

        self.assertTrue(report.verified)
        self.assertEqual(report.verdict, "verified")
        self.assertEqual(report.required_confirmation_runs, 2)
        self.assertEqual(report.observed_confirmation_runs, 2)
        self.assertFalse(report.failed_checks)
        self.assertFalse(report.missing_checks)
        self.assertTrue(all(check.outcome == "pass" for check in report.checks))

    def test_exact_metric_and_vram_boundaries_pass(self):
        exact_metrics = {
            "validation_loss": 0.45,
            "validation_accuracy": 0.79,
        }
        maximum = self.contract.verification.max_peak_vram_bytes
        confirmations = (
            _evidence(self.contract, 1, metrics=exact_metrics, peak_vram_bytes=maximum),
            _evidence(self.contract, 2, metrics=exact_metrics, peak_vram_bytes=maximum),
        )

        self.assertTrue(self.verify(confirmations).verified)

    def test_optional_vram_and_unknown_identities_create_no_checks(self):
        contract = _contract(
            max_peak_vram_bytes=None,
            workload_identity=WorkloadIdentity(),
        )
        confirmations = (
            _evidence(
                contract,
                1,
                peak_vram_bytes=None,
                workload_identity=WorkloadIdentity(),
            ),
            _evidence(
                contract,
                2,
                peak_vram_bytes=None,
                workload_identity=WorkloadIdentity(),
            ),
        )
        report = self.verify(confirmations, contract=contract)

        self.assertTrue(report.verified)
        self.assertFalse(any(c.code.startswith("resource.") for c in report.checks))
        self.assertFalse(any(c.code.startswith("identity.") for c in report.checks))

    def test_no_confirmations_is_insufficient_evidence(self):
        report = self.verify(())

        self.assertEqual(report.verdict, "insufficient_evidence")
        self.assertEqual(
            _find_check(report, "confirmation.count").outcome,
            "missing",
        )

    def test_fewer_than_required_confirmations_is_insufficient(self):
        report = self.verify((self.first,))

        self.assertEqual(report.verdict, "insufficient_evidence")
        self.assertEqual(report.observed_confirmation_runs, 1)

    def test_missing_progress_is_insufficient(self):
        report = self.verify(
            (self.first, _evidence(self.contract, 2, progress_steps=None))
        )

        self.assertEqual(report.verdict, "insufficient_evidence")
        self.assertEqual(
            _find_check(report, "execution.progress", "confirmation-run-2").outcome,
            "missing",
        )

    def test_missing_required_metric_is_insufficient(self):
        for metric_name in ("validation_loss", "validation_accuracy"):
            with self.subTest(metric_name=metric_name):
                metrics = dict(self.second.metrics)
                del metrics[metric_name]
                report = self.verify(
                    (self.first, _evidence(self.contract, 2, metrics=metrics))
                )
                self.assertEqual(report.verdict, "insufficient_evidence")
                self.assertEqual(
                    _find_check(
                        report,
                        "metric.{}".format(metric_name),
                        "confirmation-run-2",
                    ).outcome,
                    "missing",
                )

    def test_missing_required_vram_is_insufficient(self):
        report = self.verify(
            (self.first, _evidence(self.contract, 2, peak_vram_bytes=None))
        )

        self.assertEqual(report.verdict, "insufficient_evidence")
        self.assertEqual(
            _find_check(
                report,
                "resource.peak_vram_bytes",
                "confirmation-run-2",
            ).outcome,
            "missing",
        )

    def test_missing_worker_pid_is_insufficient(self):
        report = self.verify(
            (self.first, _evidence(self.contract, 2, worker_pid=None))
        )

        self.assertEqual(report.verdict, "insufficient_evidence")
        self.assertEqual(
            _find_check(
                report,
                "execution.worker_pid",
                "confirmation-run-2",
            ).outcome,
            "missing",
        )

    def test_missing_each_required_identity_is_insufficient(self):
        fields = self.contract.verification.workload_identity.known_fields
        for field_name in fields:
            with self.subTest(field_name=field_name):
                values = self.contract.verification.workload_identity.to_dict()
                values[field_name] = None
                evidence = _evidence(
                    self.contract,
                    2,
                    workload_identity=WorkloadIdentity.from_dict(values),
                )
                report = self.verify((self.first, evidence))
                self.assertEqual(report.verdict, "insufficient_evidence")
                self.assertEqual(
                    _find_check(
                        report,
                        "identity.{}".format(field_name),
                        "confirmation-run-2",
                    ).outcome,
                    "missing",
                )

    def test_observed_progress_below_contract_is_rejected(self):
        report = self.verify(
            (self.first, _evidence(self.contract, 2, progress_steps=499))
        )

        self.assertEqual(report.verdict, "rejected")
        self.assertEqual(
            _find_check(report, "execution.progress", "confirmation-run-2").outcome,
            "fail",
        )

    def test_minimized_metric_regression_is_rejected(self):
        report = self.verify(
            (
                self.first,
                _evidence(
                    self.contract,
                    2,
                    metrics={
                        "validation_loss": 0.451,
                        "validation_accuracy": 0.80,
                    },
                ),
            )
        )

        self.assertEqual(report.verdict, "rejected")

    def test_maximized_metric_regression_is_rejected(self):
        report = self.verify(
            (
                self.first,
                _evidence(
                    self.contract,
                    2,
                    metrics={
                        "validation_loss": 0.44,
                        "validation_accuracy": 0.789,
                    },
                ),
            )
        )

        self.assertEqual(report.verdict, "rejected")

    def test_stricter_absolute_metric_target_is_enforced(self):
        contract = _contract(
            metric_guards=(
                MetricGuard("validation_loss", "minimize", 0.4, 0.05),
                MetricGuard(
                    "validation_accuracy",
                    "maximize",
                    0.8,
                    0.05,
                    target_value=0.81,
                ),
            )
        )
        confirmations = (
            _evidence(contract, 1),
            _evidence(contract, 2),
        )

        self.assertEqual(
            self.verify(confirmations, contract=contract).verdict,
            "rejected",
        )

    def test_peak_vram_over_contract_is_rejected(self):
        report = self.verify(
            (
                self.first,
                _evidence(
                    self.contract,
                    2,
                    peak_vram_bytes=(15 * 1024**3) + 1,
                ),
            )
        )

        self.assertEqual(report.verdict, "rejected")

    def test_identity_mismatch_is_rejected(self):
        mismatched = WorkloadIdentity(
            dataset_fingerprint="different-dataset",
            environment_fingerprint="environment-example",
            git_commit="abc123",
            model_identifier="example/model",
        )
        report = self.verify(
            (
                self.first,
                _evidence(self.contract, 2, workload_identity=mismatched),
            )
        )

        self.assertEqual(report.verdict, "rejected")
        self.assertEqual(
            _find_check(
                report,
                "identity.dataset_fingerprint",
                "confirmation-run-2",
            ).outcome,
            "fail",
        )

    def test_every_non_success_execution_status_is_rejected(self):
        for status in (
            "training_failed",
            "contract_error",
            "worker_error",
            "timeout",
            "launch_error",
            "protocol_error",
        ):
            with self.subTest(status=status):
                report = self.verify(
                    (self.first, _evidence(self.contract, 2, status=status))
                )
                self.assertEqual(report.verdict, "rejected")

    def test_any_recorded_failure_class_is_rejected_even_with_success_status(self):
        report = self.verify(
            (
                self.first,
                _evidence(
                    self.contract,
                    2,
                    status="success",
                    failure_class="cuda_out_of_memory",
                ),
            )
        )

        self.assertEqual(report.verdict, "rejected")

    def test_probe_or_full_runs_cannot_prove_recovery(self):
        for phase in ("probe", "full"):
            with self.subTest(phase=phase):
                report = self.verify(
                    (self.first, _evidence(self.contract, 2, phase=phase))
                )
                self.assertEqual(report.verdict, "rejected")

    def test_every_contract_binding_is_enforced(self):
        mismatches = {
            "contract_digest": "f" * 64,
            "campaign_id": "different-campaign",
            "candidate_id": "different-candidate",
            "candidate_config_digest": "e" * 64,
            "project": "different-project",
            "source_run_id": "different-source-run",
        }
        for field_name, value in mismatches.items():
            with self.subTest(field_name=field_name):
                report = self.verify(
                    (self.first, _evidence(self.contract, 2, **{field_name: value}))
                )
                self.assertEqual(report.verdict, "rejected")

    def test_duplicate_trial_run_request_or_execution_is_rejected(self):
        duplicates = {
            "trial_id": self.first.trial_id,
            "run_id": self.first.run_id,
            "trial_request_digest": self.first.trial_request_digest,
            "execution_manifest_digest": self.first.execution_manifest_digest,
        }
        expected_codes = {
            "trial_id": "confirmation.unique_trial_ids",
            "run_id": "confirmation.unique_run_ids",
            "trial_request_digest": "confirmation.unique_requests",
            "execution_manifest_digest": "confirmation.unique_executions",
        }
        for field_name, value in duplicates.items():
            with self.subTest(field_name=field_name):
                report = self.verify(
                    (self.first, _evidence(self.contract, 2, **{field_name: value}))
                )
                self.assertEqual(report.verdict, "rejected")
                self.assertEqual(
                    _find_check(report, expected_codes[field_name]).outcome,
                    "fail",
                )

    def test_extra_supplied_confirmation_is_evaluated_not_cherry_picked(self):
        failing_third = _evidence(
            self.contract,
            3,
            metrics={
                "validation_loss": 9.0,
                "validation_accuracy": 0.80,
            },
        )
        report = self.verify((self.first, self.second, failing_third))

        self.assertEqual(report.observed_confirmation_runs, 3)
        self.assertEqual(report.verdict, "rejected")

    def test_observed_failure_takes_precedence_over_missing_evidence(self):
        second = _evidence(
            self.contract,
            2,
            status="training_failed",
            progress_steps=None,
            metrics={},
            peak_vram_bytes=None,
            worker_pid=None,
            workload_identity=WorkloadIdentity(),
            failure_class="cuda_out_of_memory",
        )
        report = self.verify((self.first, second))

        self.assertEqual(report.verdict, "rejected")
        self.assertTrue(report.failed_checks)
        self.assertTrue(report.missing_checks)

    def test_output_is_deterministic_for_identical_inputs(self):
        first = self.verify()
        second = self.verify()

        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(verification_digest(first), verification_digest(second))

    def test_configuration_digest_is_order_independent_but_value_sensitive(self):
        first = {"batch_size": 16, "nested": {"a": 1, "b": 2}}
        reordered = {"nested": {"b": 2, "a": 1}, "batch_size": 16}
        changed = {"batch_size": 8, "nested": {"a": 1, "b": 2}}

        self.assertEqual(configuration_digest(first), configuration_digest(reordered))
        self.assertNotEqual(configuration_digest(first), configuration_digest(changed))

    def test_invalid_verification_arguments_are_rejected(self):
        valid = {
            "contract": self.contract,
            "campaign_id": "campaign-1",
            "candidate_id": "candidate-1",
            "candidate_config": _candidate_config(),
            "confirmations": (self.first, self.second),
        }
        invalid = (
            {"contract": {}},
            {"campaign_id": "bad campaign"},
            {"candidate_id": "bad candidate"},
            {"candidate_config": {"bad": float("nan")}},
            {"confirmations": None},
            {"confirmations": ({},)},
        )
        for override in invalid:
            with self.subTest(override=tuple(override)):
                values = dict(valid)
                values.update(override)
                with self.assertRaises(VerificationError):
                    verify_recovery(**values)

    def test_confirmation_evidence_collection_has_a_hard_limit(self):
        evidence = tuple(
            _evidence(self.contract, index)
            for index in range(1, MAX_CONFIRMATION_EVIDENCE + 2)
        )

        with self.assertRaisesRegex(VerificationError, "limit"):
            self.verify(evidence)


class RecoveryVerificationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.report = _verified_report()

    def test_round_trip_preserves_canonical_json_schema_and_digest(self):
        encoded = self.report.to_json()
        restored = RecoveryVerification.from_json(encoded)

        self.assertEqual(restored, self.report)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(verification_digest(restored), verification_digest(self.report))
        self.assertRegex(verification_digest(self.report), r"^[a-f0-9]{64}$")
        self.assertEqual(
            json.loads(encoded)["schema"],
            {
                "name": VERIFICATION_REPORT_SCHEMA_NAME,
                "version": VERIFICATION_REPORT_SCHEMA_VERSION,
            },
        )

    def test_report_properties_expose_failed_and_missing_checks(self):
        contract = _contract()
        second = _evidence(
            contract,
            2,
            status="training_failed",
            progress_steps=None,
        )
        report = _verified_report(
            contract,
            (_evidence(contract, 1), second),
        )

        self.assertFalse(report.verified)
        self.assertTrue(report.failed_checks)
        self.assertTrue(report.missing_checks)

    def test_report_fields_and_check_collection_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.report.verdict = "rejected"
        with self.assertRaises(AttributeError):
            self.report.checks.append(self.report.checks[0])

    def test_deserialization_rejects_inconsistent_verified_boolean(self):
        payload = self.report.to_dict()
        payload["verified"] = False

        with self.assertRaisesRegex(VerificationError, "verified"):
            RecoveryVerification.from_dict(payload)

    def test_deserialization_rejects_verdict_inconsistent_with_checks(self):
        payload = self.report.to_dict()
        payload["verdict"] = "rejected"

        with self.assertRaisesRegex(VerificationError, "verdict"):
            RecoveryVerification.from_dict(payload)

    def test_deserialization_rejects_count_inconsistent_with_run_ids(self):
        payload = self.report.to_dict()
        payload["observed_confirmation_runs"] = 99

        with self.assertRaisesRegex(VerificationError, "run_ids"):
            RecoveryVerification.from_dict(payload)

    def test_verified_report_cannot_claim_fewer_runs_than_required(self):
        with self.assertRaisesRegex(VerificationError, "every declared"):
            RecoveryVerification(
                campaign_id="campaign-1",
                candidate_id="candidate-1",
                contract_digest="a" * 64,
                candidate_config_digest="b" * 64,
                verdict="verified",
                required_confirmation_runs=2,
                observed_confirmation_runs=1,
                confirmation_run_ids=("run-1",),
                checks=(
                    VerificationCheck(
                        "confirmation.count",
                        "pass",
                        "Count passed.",
                        2,
                        1,
                    ),
                ),
            )

    def test_verified_report_cannot_claim_duplicate_confirmation_runs(self):
        with self.assertRaisesRegex(VerificationError, "unique"):
            RecoveryVerification(
                campaign_id="campaign-1",
                candidate_id="candidate-1",
                contract_digest="a" * 64,
                candidate_config_digest="b" * 64,
                verdict="verified",
                required_confirmation_runs=2,
                observed_confirmation_runs=2,
                confirmation_run_ids=("run-1", "run-1"),
                checks=(
                    VerificationCheck(
                        "confirmation.count",
                        "pass",
                        "Count passed.",
                        2,
                        2,
                    ),
                ),
            )

    def test_deserialization_rejects_missing_unknown_or_wrong_collection_types(self):
        missing = self.report.to_dict()
        del missing["verified"]
        with self.assertRaisesRegex(VerificationError, "missing"):
            RecoveryVerification.from_dict(missing)

        unknown = self.report.to_dict()
        unknown["ranking_score"] = 1.0
        with self.assertRaisesRegex(VerificationError, "unknown fields"):
            RecoveryVerification.from_dict(unknown)

        bad_run_ids = self.report.to_dict()
        bad_run_ids["confirmation_run_ids"] = {"run-1": True}
        with self.assertRaisesRegex(VerificationError, "array"):
            RecoveryVerification.from_dict(bad_run_ids)

        bad_checks = self.report.to_dict()
        bad_checks["checks"] = {"all": "pass"}
        with self.assertRaisesRegex(VerificationError, "array"):
            RecoveryVerification.from_dict(bad_checks)

    def test_deserialization_rejects_modified_schema_and_invalid_json(self):
        payload = self.report.to_dict()
        payload["schema"]["version"] = "2.0"
        with self.assertRaisesRegex(VerificationError, "schema.version"):
            RecoveryVerification.from_dict(payload)

        for encoded in ("{not-json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(VerificationError):
                    RecoveryVerification.from_json(encoded)

    def test_digest_helpers_reject_wrong_artifact_types(self):
        with self.assertRaises(VerificationError):
            verification_digest({})
        with self.assertRaises(VerificationError):
            configuration_digest({"bad": object()})

    def test_public_status_sets_document_the_closed_vocabulary(self):
        self.assertEqual(
            VERIFICATION_VERDICTS,
            frozenset({"verified", "rejected", "insufficient_evidence"}),
        )
        self.assertEqual(CHECK_OUTCOMES, frozenset({"pass", "fail", "missing"}))


if __name__ == "__main__":
    unittest.main()