"""Acceptance tests for deterministic WatcherML candidate ranking."""
from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

from watcherml.entrypoint import TrainingEntrypoint
from watcherml.ranking import (
    CANDIDATE_ELIGIBILITY,
    DEFAULT_PREFERENCE_ORDER,
    INTERVENTION_RISKS,
    MAX_RANKING_CANDIDATES,
    RANKING_CANDIDATE_SCHEMA_NAME,
    RANKING_CANDIDATE_SCHEMA_VERSION,
    RANKING_FACTORS,
    RANKING_POLICY_SCHEMA_NAME,
    RANKING_POLICY_SCHEMA_VERSION,
    RANKING_REASON_OUTCOMES,
    RANKING_REPORT_SCHEMA_NAME,
    RANKING_REPORT_SCHEMA_VERSION,
    CandidateAssessment,
    CandidateRanking,
    RankingCandidate,
    RankingError,
    RankingPolicy,
    RankingReason,
    rank_candidates,
    ranking_digest,
)
from watcherml.recovery_contract import (
    InterventionPermissions,
    MetricGuard,
    RecoveryBudget,
    RecoveryContract,
    VerificationRequirements,
    WorkloadIdentity,
    contract_digest,
)


def _identity() -> WorkloadIdentity:
    return WorkloadIdentity(
        dataset_fingerprint="dataset-example",
        environment_fingerprint="environment-example",
        git_commit="abc123",
        model_identifier="example/model",
    )


def _contract(
    *,
    metric_guards=None,
    max_peak_vram_bytes=15 * 1024**3,
    workload_identity=None,
    permissions=None,
) -> RecoveryContract:
    if metric_guards is None:
        metric_guards = (
            MetricGuard("validation_loss", "minimize", 0.40, 0.05),
            MetricGuard("validation_accuracy", "maximize", 0.80, 0.02),
        )
    if workload_identity is None:
        workload_identity = _identity()
    return RecoveryContract(
        project="ranking-tests",
        source_run_id="source-oom-run",
        entrypoint=TrainingEntrypoint("training.entrypoint:train"),
        source_config={
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "model": "example/model",
        },
        budget=RecoveryBudget(
            max_trials=10,
            max_probe_trials=5,
            max_full_trials=2,
            probe_steps=30,
            trial_timeout_seconds=1_800,
            campaign_timeout_seconds=7_200,
        ),
        verification=VerificationRequirements(
            minimum_progress_steps=500,
            metric_guards=metric_guards,
            confirmation_runs=2,
            max_peak_vram_bytes=max_peak_vram_bytes,
            workload_identity=workload_identity,
        ),
        permissions=permissions,
    )


def _candidate(
    contract: RecoveryContract,
    index: int,
    **overrides
) -> RankingCandidate:
    values = {
        "campaign_id": "campaign-1",
        "candidate_id": "candidate-{:02d}".format(index),
        "trial_id": "full-trial-{:02d}".format(index),
        "run_id": "full-run-{:02d}".format(index),
        "project": contract.project,
        "source_run_id": contract.source_run_id,
        "contract_digest": contract_digest(contract),
        "candidate_config_digest": "{:064x}".format(index),
        "trial_request_digest": "{:064x}".format(100 + index),
        "execution_manifest_digest": "{:064x}".format(200 + index),
        "phase": "full",
        "status": "success",
        "metrics": {
            "validation_loss": 0.44,
            "validation_accuracy": 0.81,
            "samples_per_second": 12.0,
        },
        "progress_steps": 500,
        "peak_vram_bytes": 14 * 1024**3,
        "workload_identity": _identity(),
        "worker_pid": 10_000 + index,
        "failure_class": None,
        "intervention_risk": "low",
        "approval_required": False,
        "semantic_change": False,
        "change_count": 2,
    }
    values.update(overrides)
    return RankingCandidate(**values)


def _policy(**overrides) -> RankingPolicy:
    values = {
        "primary_metric": "validation_loss",
        "throughput_metric": "samples_per_second",
        "preference_order": DEFAULT_PREFERENCE_ORDER,
    }
    values.update(overrides)
    return RankingPolicy(**values)


def _rank(contract=None, candidates=None, policy=None) -> CandidateRanking:
    contract = _contract() if contract is None else contract
    if candidates is None:
        candidates = (
            _candidate(
                contract,
                1,
                metrics={
                    "validation_loss": 0.42,
                    "validation_accuracy": 0.81,
                    "samples_per_second": 10.0,
                },
                peak_vram_bytes=14 * 1024**3,
            ),
            _candidate(
                contract,
                2,
                metrics={
                    "validation_loss": 0.44,
                    "validation_accuracy": 0.82,
                    "samples_per_second": 15.0,
                },
                peak_vram_bytes=12 * 1024**3,
            ),
        )
    return rank_candidates(
        contract,
        campaign_id="campaign-1",
        policy=_policy() if policy is None else policy,
        candidates=candidates,
    )


def _assessment(report: CandidateRanking, candidate_id: str):
    matches = [
        item for item in report.assessments if item.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            "expected one assessment for {!r}, found {}".format(
                candidate_id,
                len(matches),
            )
        )
    return matches[0]


def _reason(assessment: CandidateAssessment, code: str):
    matches = [item for item in assessment.reasons if item.code == code]
    if len(matches) != 1:
        raise AssertionError(
            "expected one reason for {!r}, found {}".format(code, len(matches))
        )
    return matches[0]


class RankingPolicyTests(unittest.TestCase):
    def test_default_without_throughput_metric_omits_inactive_factor(self):
        policy = RankingPolicy(primary_metric="validation_loss")

        self.assertNotIn("throughput", policy.preference_order)
        self.assertEqual(
            policy.preference_order,
            tuple(
                factor
                for factor in DEFAULT_PREFERENCE_ORDER
                if factor != "throughput"
            ),
        )

    def test_default_with_throughput_metric_preserves_full_order(self):
        policy = _policy()

        self.assertEqual(policy.preference_order, DEFAULT_PREFERENCE_ORDER)

    def test_custom_explicit_preference_order_is_preserved(self):
        order = (
            "peak_vram_bytes",
            "primary_metric",
            "intervention_risk",
        )
        policy = RankingPolicy("validation_loss", preference_order=order)

        self.assertEqual(policy.preference_order, order)

    def test_policy_round_trip_preserves_algorithm_invariants(self):
        policy = _policy()
        encoded = policy.to_json()
        restored = RankingPolicy.from_json(encoded)

        self.assertEqual(restored, policy)
        self.assertEqual(restored.to_json(), encoded)
        payload = json.loads(encoded)
        self.assertEqual(
            payload["schema"],
            {
                "name": RANKING_POLICY_SCHEMA_NAME,
                "version": RANKING_POLICY_SCHEMA_VERSION,
            },
        )
        self.assertEqual(payload["algorithm"], "constraint_first_lexicographic")
        self.assertIs(payload["weighted_score"], False)

    def test_primary_metric_must_be_a_contract_guard(self):
        with self.assertRaisesRegex(RankingError, "metric guard"):
            RankingPolicy("untracked_metric").validate_against(_contract())

    def test_primary_and_throughput_metric_names_are_strict(self):
        for field_name in ("primary_metric", "throughput_metric"):
            for value in ("", "metric with spaces", "1metric"):
                with self.subTest(field_name=field_name, value=value):
                    kwargs = {"primary_metric": "validation_loss"}
                    kwargs[field_name] = value
                    with self.assertRaises(RankingError):
                        RankingPolicy(**kwargs)

    def test_throughput_metric_must_differ_from_primary(self):
        with self.assertRaisesRegex(RankingError, "differ"):
            RankingPolicy("validation_loss", "validation_loss")

    def test_preference_order_must_be_nonempty_unique_and_known(self):
        invalid = (
            (),
            ("primary_metric", "primary_metric"),
            ("primary_metric", "opaque_score"),
            ("peak_vram_bytes",),
        )
        for order in invalid:
            with self.subTest(order=order):
                with self.assertRaises(RankingError):
                    RankingPolicy("validation_loss", preference_order=order)

    def test_custom_throughput_factor_requires_metric(self):
        with self.assertRaisesRegex(RankingError, "requires"):
            RankingPolicy(
                "validation_loss",
                preference_order=("primary_metric", "throughput"),
            )

    def test_throughput_metric_requires_factor(self):
        with self.assertRaisesRegex(RankingError, "requires"):
            RankingPolicy(
                "validation_loss",
                "samples_per_second",
                preference_order=("primary_metric", "peak_vram_bytes"),
            )

    def test_deserialization_rejects_algorithm_or_weighted_score_tampering(self):
        payload = _policy().to_dict()
        payload["algorithm"] = "weighted_sum"
        with self.assertRaisesRegex(RankingError, "algorithm"):
            RankingPolicy.from_dict(payload)

        payload = _policy().to_dict()
        payload["weighted_score"] = True
        with self.assertRaisesRegex(RankingError, "weighted_score"):
            RankingPolicy.from_dict(payload)

    def test_deserialization_rejects_missing_unknown_schema_and_invalid_json(self):
        missing = _policy().to_dict()
        del missing["primary_metric"]
        with self.assertRaisesRegex(RankingError, "missing"):
            RankingPolicy.from_dict(missing)

        unknown = _policy().to_dict()
        unknown["weights"] = {"loss": 0.7}
        with self.assertRaisesRegex(RankingError, "unknown fields"):
            RankingPolicy.from_dict(unknown)

        schema = _policy().to_dict()
        schema["schema"]["version"] = "2.0"
        with self.assertRaisesRegex(RankingError, "schema.version"):
            RankingPolicy.from_dict(schema)

        for encoded in ("{bad-json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(RankingError):
                    RankingPolicy.from_json(encoded)


class RankingCandidateArtifactTests(unittest.TestCase):
    def setUp(self):
        self.contract = _contract()
        self.candidate = _candidate(self.contract, 1)

    def test_round_trip_preserves_canonical_json_and_schema(self):
        encoded = self.candidate.to_json()
        restored = RankingCandidate.from_json(encoded)

        self.assertEqual(restored, self.candidate)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(
            json.loads(encoded)["schema"],
            {
                "name": RANKING_CANDIDATE_SCHEMA_NAME,
                "version": RANKING_CANDIDATE_SCHEMA_VERSION,
            },
        )

    def test_metrics_are_normalized_and_deeply_immutable(self):
        metrics = {"validation_loss": 1, "validation_accuracy": 0.81}
        candidate = _candidate(self.contract, 1, metrics=metrics)
        metrics["validation_loss"] = 99

        self.assertEqual(candidate.metrics["validation_loss"], 1.0)
        with self.assertRaises(TypeError):
            candidate.metrics["validation_loss"] = 0.4
        exported = candidate.to_dict()
        exported["metrics"]["validation_loss"] = 7
        self.assertEqual(candidate.metrics["validation_loss"], 1.0)

    def test_dataclass_fields_are_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            self.candidate.status = "training_failed"

    def test_project_and_failure_class_are_trimmed(self):
        candidate = _candidate(
            self.contract,
            1,
            project="  ranking-tests  ",
            status="training_failed",
            failure_class="  cuda_out_of_memory  ",
        )

        self.assertEqual(candidate.project, "ranking-tests")
        self.assertEqual(candidate.failure_class, "cuda_out_of_memory")

    def test_missing_runtime_observations_can_be_represented(self):
        candidate = _candidate(
            self.contract,
            1,
            progress_steps=None,
            peak_vram_bytes=None,
            worker_pid=None,
        )

        self.assertIsNone(candidate.progress_steps)
        self.assertIsNone(candidate.peak_vram_bytes)
        self.assertIsNone(candidate.worker_pid)

    def test_invalid_identifiers_are_rejected(self):
        for field_name in (
            "campaign_id",
            "candidate_id",
            "trial_id",
            "run_id",
            "source_run_id",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(RankingError):
                    _candidate(self.contract, 1, **{field_name: "contains spaces"})

    def test_artifact_digests_must_be_lowercase_sha256(self):
        for field_name in (
            "contract_digest",
            "candidate_config_digest",
            "trial_request_digest",
            "execution_manifest_digest",
        ):
            for value in ("short", "A" * 64, "g" * 64):
                with self.subTest(field_name=field_name, value=value[:4]):
                    with self.assertRaises(RankingError):
                        _candidate(self.contract, 1, **{field_name: value})

    def test_invalid_phase_and_status_are_rejected(self):
        with self.assertRaisesRegex(RankingError, "phase"):
            _candidate(self.contract, 1, phase="ranking")
        with self.assertRaisesRegex(RankingError, "status"):
            _candidate(self.contract, 1, status="completed")

    def test_metric_names_and_values_are_strict(self):
        invalid = (
            {"": 0.4},
            {"metric with spaces": 0.4},
            {"loss": True},
            {"loss": "0.4"},
            {"loss": float("nan")},
            {"loss": float("inf")},
        )
        for metrics in invalid:
            with self.subTest(metrics=metrics):
                with self.assertRaises(RankingError):
                    _candidate(self.contract, 1, metrics=metrics)

    def test_progress_and_vram_must_be_nonnegative_integers(self):
        for field_name in ("progress_steps", "peak_vram_bytes"):
            for value in (True, -1, 1.5):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(RankingError):
                        _candidate(self.contract, 1, **{field_name: value})

    def test_worker_pid_and_change_count_must_be_positive_integers(self):
        for field_name in ("worker_pid", "change_count"):
            for value in (True, 0, -1, 1.5):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(RankingError):
                        _candidate(self.contract, 1, **{field_name: value})

    def test_workload_identity_must_use_declared_type(self):
        with self.assertRaisesRegex(RankingError, "WorkloadIdentity"):
            _candidate(
                self.contract,
                1,
                workload_identity={"dataset_fingerprint": "dataset-example"},
            )

    def test_intervention_risk_and_flags_are_strict(self):
        with self.assertRaisesRegex(RankingError, "intervention_risk"):
            _candidate(self.contract, 1, intervention_risk="extreme")
        for field_name in ("approval_required", "semantic_change"):
            with self.subTest(field_name=field_name):
                with self.assertRaises(RankingError):
                    _candidate(self.contract, 1, **{field_name: 1})

    def test_blank_or_oversized_failure_class_is_rejected(self):
        for value in ("", "   ", "x" * 4_001):
            with self.subTest(length=len(value)):
                with self.assertRaises(RankingError):
                    _candidate(self.contract, 1, failure_class=value)

    def test_deserialization_rejects_missing_unknown_and_nested_unknown_fields(self):
        missing = self.candidate.to_dict()
        del missing["workload_identity"]
        with self.assertRaisesRegex(RankingError, "missing"):
            RankingCandidate.from_dict(missing)

        unknown = self.candidate.to_dict()
        unknown["weighted_score"] = 0.8
        with self.assertRaisesRegex(RankingError, "unknown fields"):
            RankingCandidate.from_dict(unknown)

        nested = self.candidate.to_dict()
        nested["intervention"]["shell_command"] = "python train.py"
        with self.assertRaisesRegex(RankingError, "unknown fields"):
            RankingCandidate.from_dict(nested)

    def test_deserialization_rejects_schema_invalid_json_and_identity_tampering(self):
        schema = self.candidate.to_dict()
        schema["schema"]["version"] = "2.0"
        with self.assertRaisesRegex(RankingError, "schema.version"):
            RankingCandidate.from_dict(schema)

        identity = self.candidate.to_dict()
        identity["workload_identity"]["random_seed"] = "hidden"
        with self.assertRaisesRegex(RankingError, "unknown fields"):
            RankingCandidate.from_dict(identity)

        for encoded in ("{bad-json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(RankingError):
                    RankingCandidate.from_json(encoded)


class RankingReasonAndAssessmentTests(unittest.TestCase):
    def test_reason_round_trip_and_deep_immutability(self):
        expected = {"limits": [{"maximum": 0.45}]}
        reason = RankingReason(
            "metric.validation_loss",
            "pass",
            "Metric passed its contract gate.",
            expected,
            {"value": [0.44]},
        )
        expected["limits"][0]["maximum"] = 99

        self.assertEqual(RankingReason.from_dict(reason.to_dict()), reason)
        self.assertEqual(reason.expected["limits"][0]["maximum"], 0.45)
        with self.assertRaises(TypeError):
            reason.expected["limits"][0]["maximum"] = 0.0

    def test_reason_rejects_invalid_code_outcome_text_and_json(self):
        invalid = (
            {"code": "Bad Code"},
            {"outcome": "unknown"},
            {"message": ""},
            {"expected": float("nan")},
            {"observed": object()},
        )
        defaults = {
            "code": "execution.status",
            "outcome": "pass",
            "message": "Execution passed.",
            "expected": "success",
            "observed": "success",
        }
        for override in invalid:
            with self.subTest(override=tuple(override)):
                values = dict(defaults)
                values.update(override)
                with self.assertRaises(RankingError):
                    RankingReason(**values)

    def test_assessment_round_trip_and_preference_immutability(self):
        report = _rank()
        assessment = report.assessments[0]
        restored = CandidateAssessment.from_dict(assessment.to_dict())

        self.assertEqual(restored, assessment)
        with self.assertRaises(TypeError):
            assessment.preference_values["primary_metric"] = 9

    def test_eligible_assessment_requires_rank(self):
        reason = RankingReason(
            "execution.status", "pass", "Execution passed.", "success", "success"
        )
        with self.assertRaisesRegex(RankingError, "require"):
            CandidateAssessment(
                "candidate-1",
                "run-1",
                "eligible",
                None,
                {"primary_metric": 0.4},
                "primary_metric",
                (reason,),
            )

    def test_ineligible_assessment_cannot_have_rank(self):
        reason = RankingReason(
            "execution.status", "fail", "Execution failed.", "success", "timeout"
        )
        with self.assertRaisesRegex(RankingError, "cannot"):
            CandidateAssessment(
                "candidate-1",
                "run-1",
                "rejected",
                1,
                {"primary_metric": None},
                None,
                (reason,),
            )

    def test_assessment_eligibility_must_match_reason_outcomes(self):
        failing = RankingReason(
            "execution.status", "fail", "Execution failed.", "success", "timeout"
        )
        with self.assertRaisesRegex(RankingError, "inconsistent"):
            CandidateAssessment(
                "candidate-1",
                "run-1",
                "eligible",
                1,
                {"primary_metric": 0.4},
                "primary_metric",
                (failing,),
            )

    def test_assessment_rejects_invalid_deciding_factor_and_reasons(self):
        passing = RankingReason(
            "execution.status", "pass", "Execution passed.", "success", "success"
        )
        with self.assertRaisesRegex(RankingError, "deciding_factor"):
            CandidateAssessment(
                "candidate-1",
                "run-1",
                "eligible",
                1,
                {"primary_metric": 0.4},
                "opaque_score",
                (passing,),
            )
        with self.assertRaisesRegex(RankingError, "reasons"):
            CandidateAssessment(
                "candidate-1",
                "run-1",
                "eligible",
                1,
                {"primary_metric": 0.4},
                "primary_metric",
                (),
            )

    def test_reason_and_assessment_deserialization_are_strict(self):
        reason = RankingReason(
            "execution.status", "pass", "Execution passed.", "success", "success"
        ).to_dict()
        reason["override"] = True
        with self.assertRaisesRegex(RankingError, "unknown fields"):
            RankingReason.from_dict(reason)

        assessment = _rank().assessments[0].to_dict()
        assessment["score"] = 0.9
        with self.assertRaisesRegex(RankingError, "unknown fields"):
            CandidateAssessment.from_dict(assessment)


class ConstraintFirstRankingDecisionTests(unittest.TestCase):
    def setUp(self):
        self.contract = _contract()

    def test_primary_minimized_metric_orders_candidates_independent_of_input(self):
        better = _candidate(
            self.contract,
            1,
            metrics={
                "validation_loss": 0.41,
                "validation_accuracy": 0.81,
                "samples_per_second": 8.0,
            },
            peak_vram_bytes=14 * 1024**3,
        )
        worse = _candidate(
            self.contract,
            2,
            metrics={
                "validation_loss": 0.44,
                "validation_accuracy": 0.83,
                "samples_per_second": 20.0,
            },
            peak_vram_bytes=10 * 1024**3,
        )

        first = _rank(self.contract, (worse, better))
        second = _rank(self.contract, (better, worse))

        self.assertEqual(first.confirmation_order, ("candidate-01", "candidate-02"))
        self.assertEqual(first, second)

    def test_maximized_primary_metric_uses_higher_value(self):
        policy = RankingPolicy(
            "validation_accuracy",
            preference_order=("primary_metric", "change_count"),
        )
        lower = _candidate(
            self.contract,
            1,
            metrics={"validation_loss": 0.43, "validation_accuracy": 0.81},
        )
        higher = _candidate(
            self.contract,
            2,
            metrics={"validation_loss": 0.44, "validation_accuracy": 0.84},
        )

        report = _rank(self.contract, (lower, higher), policy)

        self.assertEqual(report.confirmation_order[0], "candidate-02")

    def test_custom_vram_first_policy_can_prioritize_memory_headroom(self):
        policy = RankingPolicy(
            "validation_loss",
            preference_order=("peak_vram_bytes", "primary_metric"),
        )
        better_loss = _candidate(
            self.contract,
            1,
            metrics={"validation_loss": 0.41, "validation_accuracy": 0.81},
            peak_vram_bytes=14 * 1024**3,
        )
        lower_vram = _candidate(
            self.contract,
            2,
            metrics={"validation_loss": 0.44, "validation_accuracy": 0.81},
            peak_vram_bytes=10 * 1024**3,
        )

        report = _rank(self.contract, (better_loss, lower_vram), policy)

        self.assertEqual(report.confirmation_order[0], "candidate-02")

    def test_throughput_breaks_tie_in_favor_of_higher_value(self):
        slower = _candidate(
            self.contract,
            1,
            metrics={
                "validation_loss": 0.43,
                "validation_accuracy": 0.81,
                "samples_per_second": 8.0,
            },
        )
        faster = _candidate(
            self.contract,
            2,
            metrics={
                "validation_loss": 0.43,
                "validation_accuracy": 0.81,
                "samples_per_second": 16.0,
            },
        )

        report = _rank(self.contract, (slower, faster))

        self.assertEqual(report.confirmation_order[0], "candidate-02")
        self.assertEqual(
            _assessment(report, "candidate-01").deciding_factor,
            "throughput",
        )

    def test_lower_intervention_risk_breaks_tie(self):
        permissions = InterventionPermissions(True, True, True)
        contract = _contract(permissions=permissions)
        low = _candidate(contract, 1, intervention_risk="low")
        high = _candidate(
            contract,
            2,
            intervention_risk="high",
            approval_required=True,
        )
        policy = RankingPolicy(
            "validation_loss",
            preference_order=("primary_metric", "intervention_risk"),
        )

        self.assertEqual(
            _rank(contract, (high, low), policy).confirmation_order[0],
            "candidate-01",
        )

    def test_nonsemantic_change_breaks_tie(self):
        contract = _contract(permissions=InterventionPermissions(True, True, True))
        ordinary = _candidate(contract, 1)
        semantic = _candidate(
            contract,
            2,
            semantic_change=True,
            approval_required=True,
        )
        policy = RankingPolicy(
            "validation_loss",
            preference_order=("primary_metric", "semantic_change"),
        )

        self.assertEqual(
            _rank(contract, (semantic, ordinary), policy).confirmation_order[0],
            "candidate-01",
        )

    def test_automatic_change_breaks_tie(self):
        contract = _contract(permissions=InterventionPermissions(True, False, False))
        automatic = _candidate(contract, 1)
        gated = _candidate(contract, 2, approval_required=True)
        policy = RankingPolicy(
            "validation_loss",
            preference_order=("primary_metric", "approval_required"),
        )

        self.assertEqual(
            _rank(contract, (gated, automatic), policy).confirmation_order[0],
            "candidate-01",
        )

    def test_fewer_changes_breaks_tie(self):
        simpler = _candidate(self.contract, 1, change_count=1)
        complex_candidate = _candidate(self.contract, 2, change_count=3)
        policy = RankingPolicy(
            "validation_loss",
            preference_order=("primary_metric", "change_count"),
        )

        self.assertEqual(
            _rank(self.contract, (complex_candidate, simpler), policy).confirmation_order[0],
            "candidate-01",
        )

    def test_candidate_id_is_final_stable_tie_breaker(self):
        first = _candidate(self.contract, 1)
        second = _candidate(self.contract, 2)
        policy = RankingPolicy(
            "validation_loss",
            preference_order=("primary_metric",),
        )

        report = _rank(self.contract, (second, first), policy)

        self.assertEqual(report.confirmation_order, ("candidate-01", "candidate-02"))
        self.assertIsNone(_assessment(report, "candidate-02").deciding_factor)

    def test_missing_optional_throughput_sorts_last_but_remains_eligible(self):
        measured = _candidate(self.contract, 1)
        missing = _candidate(
            self.contract,
            2,
            metrics={"validation_loss": 0.44, "validation_accuracy": 0.81},
        )

        report = _rank(self.contract, (missing, measured))

        self.assertEqual(report.confirmation_order[0], "candidate-01")
        self.assertEqual(_assessment(report, "candidate-02").eligibility, "eligible")

    def test_missing_optional_vram_sorts_last_when_contract_does_not_require_it(self):
        contract = _contract(max_peak_vram_bytes=None)
        measured = _candidate(contract, 1)
        missing = _candidate(contract, 2, peak_vram_bytes=None)

        report = _rank(contract, (missing, measured))

        self.assertEqual(report.confirmation_order[0], "candidate-01")
        self.assertEqual(_assessment(report, "candidate-02").eligibility, "eligible")

    def test_empty_candidate_set_produces_empty_provisional_schedule(self):
        report = _rank(self.contract, ())

        self.assertEqual(report.confirmation_order, ())
        self.assertEqual(report.assessments, ())
        self.assertIsNone(report.next_confirmation_candidate_id)

    def test_non_success_phase_or_failure_class_is_rejected(self):
        cases = (
            {"status": "training_failed"},
            {"status": "timeout"},
            {"phase": "probe"},
            {"phase": "confirmation"},
            {"failure_class": "cuda_out_of_memory"},
        )
        for index, override in enumerate(cases, 1):
            with self.subTest(override=override):
                candidate = _candidate(self.contract, index, **override)
                assessment = _rank(self.contract, (candidate,)).assessments[0]
                self.assertEqual(assessment.eligibility, "rejected")

    def test_missing_worker_progress_metric_vram_or_identity_is_insufficient(self):
        missing_identity = WorkloadIdentity(
            dataset_fingerprint=None,
            environment_fingerprint="environment-example",
            git_commit="abc123",
            model_identifier="example/model",
        )
        cases = (
            {"worker_pid": None},
            {"progress_steps": None},
            {"metrics": {"validation_accuracy": 0.81}},
            {"peak_vram_bytes": None},
            {"workload_identity": missing_identity},
        )
        for index, override in enumerate(cases, 1):
            with self.subTest(override=tuple(override)):
                assessment = _rank(
                    self.contract,
                    (_candidate(self.contract, index, **override),),
                ).assessments[0]
                self.assertEqual(assessment.eligibility, "insufficient_evidence")

    def test_below_progress_metric_regression_vram_or_identity_mismatch_is_rejected(self):
        wrong_identity = WorkloadIdentity(
            dataset_fingerprint="different",
            environment_fingerprint="environment-example",
            git_commit="abc123",
            model_identifier="example/model",
        )
        cases = (
            {"progress_steps": 499},
            {"metrics": {"validation_loss": 0.451, "validation_accuracy": 0.81}},
            {"metrics": {"validation_loss": 0.44, "validation_accuracy": 0.779}},
            {"peak_vram_bytes": (15 * 1024**3) + 1},
            {"workload_identity": wrong_identity},
        )
        for index, override in enumerate(cases, 1):
            with self.subTest(override=tuple(override)):
                assessment = _rank(
                    self.contract,
                    (_candidate(self.contract, index, **override),),
                ).assessments[0]
                self.assertEqual(assessment.eligibility, "rejected")

    def test_contract_campaign_project_and_source_bindings_are_enforced(self):
        cases = (
            {"contract_digest": "f" * 64},
            {"campaign_id": "different-campaign"},
            {"project": "different-project"},
            {"source_run_id": "different-source"},
        )
        for index, override in enumerate(cases, 1):
            with self.subTest(override=override):
                assessment = _rank(
                    self.contract,
                    (_candidate(self.contract, index, **override),),
                ).assessments[0]
                self.assertEqual(assessment.eligibility, "rejected")

    def test_contract_intervention_permissions_are_enforced(self):
        cases = (
            {"approval_required": True},
            {"approval_required": True, "semantic_change": True},
            {"approval_required": True, "intervention_risk": "high"},
        )
        expected_reason = (
            "intervention.approval_scope",
            "intervention.semantic_scope",
            "intervention.high_risk_scope",
        )
        for index, (override, reason_code) in enumerate(
            zip(cases, expected_reason),
            1,
        ):
            with self.subTest(override=override):
                assessment = _rank(
                    self.contract,
                    (_candidate(self.contract, index, **override),),
                ).assessments[0]
                self.assertEqual(assessment.eligibility, "rejected")
                self.assertEqual(_reason(assessment, reason_code).outcome, "fail")

    def test_fully_permissive_contract_can_rank_gated_high_risk_semantic_candidate(self):
        contract = _contract(permissions=InterventionPermissions(True, True, True))
        candidate = _candidate(
            contract,
            1,
            approval_required=True,
            semantic_change=True,
            intervention_risk="high",
        )

        assessment = _rank(contract, (candidate,)).assessments[0]

        self.assertEqual(assessment.eligibility, "eligible")

    def test_observed_failure_overrides_simultaneously_missing_evidence(self):
        candidate = _candidate(
            self.contract,
            1,
            status="training_failed",
            metrics={},
            progress_steps=None,
            peak_vram_bytes=None,
            worker_pid=None,
            workload_identity=WorkloadIdentity(),
            failure_class="cuda_out_of_memory",
        )
        assessment = _rank(self.contract, (candidate,)).assessments[0]

        self.assertEqual(assessment.eligibility, "rejected")
        self.assertTrue(any(reason.outcome == "fail" for reason in assessment.reasons))
        self.assertTrue(any(reason.outcome == "missing" for reason in assessment.reasons))

    def test_duplicate_candidate_artifacts_are_rejected_before_ranking(self):
        first = _candidate(self.contract, 1)
        fields = (
            "candidate_id",
            "trial_id",
            "run_id",
            "candidate_config_digest",
            "trial_request_digest",
            "execution_manifest_digest",
        )
        for field_name in fields:
            with self.subTest(field_name=field_name):
                second = _candidate(
                    self.contract,
                    2,
                    **{field_name: getattr(first, field_name)}
                )
                with self.assertRaisesRegex(RankingError, "duplicate"):
                    _rank(self.contract, (first, second))

    def test_invalid_rank_arguments_and_hard_collection_limit_are_rejected(self):
        candidate = _candidate(self.contract, 1)
        invalid = (
            {"contract": {}},
            {"campaign_id": "bad campaign"},
            {"policy": {}},
            {"candidates": None},
            {"candidates": ({},)},
        )
        valid = {
            "contract": self.contract,
            "campaign_id": "campaign-1",
            "policy": _policy(),
            "candidates": (candidate,),
        }
        for override in invalid:
            with self.subTest(override=tuple(override)):
                values = dict(valid)
                values.update(override)
                with self.assertRaises(RankingError):
                    rank_candidates(**values)

        too_many = tuple(
            _candidate(self.contract, index)
            for index in range(1, MAX_RANKING_CANDIDATES + 2)
        )
        with self.assertRaisesRegex(RankingError, "limit"):
            _rank(self.contract, too_many)

    def test_report_contains_no_weighted_score_or_recovery_claim(self):
        payload = _rank().to_dict()

        self.assertTrue(payload["provisional"])
        self.assertIsNone(payload["recovery_verdict"])
        self.assertNotIn("best_run_id", payload)
        self.assertNotIn("ranking_score", payload)
        self.assertNotIn("verified", payload)


class CandidateRankingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.report = _rank()

    def test_round_trip_preserves_canonical_json_schema_and_digest(self):
        encoded = self.report.to_json()
        restored = CandidateRanking.from_json(encoded)

        self.assertEqual(restored, self.report)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(ranking_digest(restored), ranking_digest(self.report))
        self.assertRegex(ranking_digest(self.report), r"^[a-f0-9]{64}$")
        self.assertEqual(
            json.loads(encoded)["schema"],
            {
                "name": RANKING_REPORT_SCHEMA_NAME,
                "version": RANKING_REPORT_SCHEMA_VERSION,
            },
        )

    def test_next_confirmation_candidate_is_first_provisional_candidate(self):
        self.assertEqual(
            self.report.next_confirmation_candidate_id,
            self.report.confirmation_order[0],
        )

    def test_report_collections_and_fields_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.report.campaign_id = "changed"
        with self.assertRaises(AttributeError):
            self.report.confirmation_order.append("candidate-99")

    def test_candidate_assessments_are_ordered_by_rank_then_ineligible_id(self):
        contract = _contract()
        eligible = _candidate(contract, 3)
        rejected = _candidate(contract, 2, status="training_failed")
        missing = _candidate(contract, 1, progress_steps=None)
        report = _rank(contract, (rejected, eligible, missing))

        self.assertEqual(
            tuple(item.candidate_id for item in report.assessments),
            ("candidate-03", "candidate-01", "candidate-02"),
        )

    def test_deserialization_rejects_provisional_or_verdict_tampering(self):
        payload = self.report.to_dict()
        payload["provisional"] = False
        with self.assertRaisesRegex(RankingError, "provisional"):
            CandidateRanking.from_dict(payload)

        payload = self.report.to_dict()
        payload["recovery_verdict"] = "verified"
        with self.assertRaisesRegex(RankingError, "verdict"):
            CandidateRanking.from_dict(payload)

    def test_deserialization_rejects_inconsistent_next_candidate(self):
        payload = self.report.to_dict()
        payload["next_confirmation_candidate_id"] = "candidate-99"

        with self.assertRaisesRegex(RankingError, "inconsistent"):
            CandidateRanking.from_dict(payload)

    def test_report_requires_contiguous_ranks_and_matching_order(self):
        assessments = list(self.report.assessments)
        first = assessments[0]
        invalid_rank = CandidateAssessment(
            first.candidate_id,
            first.run_id,
            first.eligibility,
            3,
            dict(first.preference_values),
            first.deciding_factor,
            first.reasons,
        )
        with self.assertRaisesRegex(RankingError, "contiguous"):
            CandidateRanking(
                self.report.campaign_id,
                self.report.contract_digest,
                self.report.policy,
                self.report.confirmation_order,
                (invalid_rank,) + tuple(assessments[1:]),
            )

        with self.assertRaisesRegex(RankingError, "match"):
            CandidateRanking(
                self.report.campaign_id,
                self.report.contract_digest,
                self.report.policy,
                tuple(reversed(self.report.confirmation_order)),
                self.report.assessments,
            )

    def test_report_rejects_duplicate_order_or_assessment_ids(self):
        with self.assertRaisesRegex(RankingError, "unique"):
            CandidateRanking(
                self.report.campaign_id,
                self.report.contract_digest,
                self.report.policy,
                (self.report.confirmation_order[0],) * 2,
                self.report.assessments,
            )

        duplicate = self.report.assessments[0]
        with self.assertRaisesRegex(RankingError, "unique"):
            CandidateRanking(
                self.report.campaign_id,
                self.report.contract_digest,
                self.report.policy,
                self.report.confirmation_order,
                (duplicate, duplicate),
            )

    def test_deserialization_rejects_missing_unknown_wrong_arrays_and_schema(self):
        missing = self.report.to_dict()
        del missing["next_confirmation_candidate_id"]
        with self.assertRaisesRegex(RankingError, "missing"):
            CandidateRanking.from_dict(missing)

        unknown = self.report.to_dict()
        unknown["score"] = 0.9
        with self.assertRaisesRegex(RankingError, "unknown fields"):
            CandidateRanking.from_dict(unknown)

        bad_order = self.report.to_dict()
        bad_order["confirmation_order"] = {"candidate-01": 1}
        with self.assertRaisesRegex(RankingError, "array"):
            CandidateRanking.from_dict(bad_order)

        bad_assessments = self.report.to_dict()
        bad_assessments["assessments"] = {"eligible": []}
        with self.assertRaisesRegex(RankingError, "array"):
            CandidateRanking.from_dict(bad_assessments)

        schema = self.report.to_dict()
        schema["schema"]["version"] = "2.0"
        with self.assertRaisesRegex(RankingError, "schema.version"):
            CandidateRanking.from_dict(schema)

    def test_invalid_json_and_digest_helper_type_are_rejected(self):
        for encoded in ("{bad-json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(RankingError):
                    CandidateRanking.from_json(encoded)
        with self.assertRaises(RankingError):
            ranking_digest({})

    def test_public_vocabulary_is_closed_and_documented(self):
        self.assertEqual(
            CANDIDATE_ELIGIBILITY,
            frozenset({"eligible", "rejected", "insufficient_evidence"}),
        )
        self.assertEqual(
            RANKING_REASON_OUTCOMES,
            frozenset({"pass", "fail", "missing"}),
        )
        self.assertEqual(INTERVENTION_RISKS, frozenset({"low", "medium", "high"}))
        self.assertEqual(set(DEFAULT_PREFERENCE_ORDER), RANKING_FACTORS)


if __name__ == "__main__":
    unittest.main()