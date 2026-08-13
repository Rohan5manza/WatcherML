"""Acceptance tests for bounded deterministic WatcherML campaigns.

These tests deliberately use an in-memory executor adapter.  Process creation,
deadline enforcement, and worker protocol behavior are covered by
``test_trial_runner.py``; this file tests the orchestration boundary that
consumes those authenticated execution manifests.
"""
from __future__ import annotations

import json
import math
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

from watcherml.campaign import (
    CAMPAIGN_OBSERVATION_SCHEMA_NAME,
    CAMPAIGN_OBSERVATION_SCHEMA_VERSION,
    CAMPAIGN_RESULT_SCHEMA_NAME,
    CAMPAIGN_RESULT_SCHEMA_VERSION,
    CAMPAIGN_STATUSES,
    CAMPAIGN_TRIAL_SCHEMA_NAME,
    CAMPAIGN_TRIAL_SCHEMA_VERSION,
    CampaignBudgetUsage,
    CampaignCandidate,
    CampaignError,
    CampaignResult,
    CampaignTrial,
    ExecutedTrial,
    RunObservation,
    campaign_result_digest,
    run_campaign,
)
from watcherml.capabilities import discover_capabilities
from watcherml.entrypoint import TrainingEntrypoint
from watcherml.interventions import (
    InterventionApplication,
    InterventionAuthorization,
    InterventionChange,
    InterventionProposal,
    materialize_intervention,
    resolve_intervention,
)
from watcherml.ranking import RankingPolicy
from watcherml.recovery_contract import (
    ContractScopeError,
    InterventionPermissions,
    MetricGuard,
    RecoveryBudget,
    RecoveryContract,
    VerificationRequirements,
    WorkloadIdentity,
    contract_digest,
)
from watcherml.trial_protocol import (
    EXIT_CONTRACT_ERROR,
    EXIT_SUCCESS,
    EXIT_TRAINING_FAILED,
    EXIT_WORKER_ERROR,
    TrialResult,
)
from watcherml.trial_runner import TrialExecution


def _source_config() -> dict:
    return {
        "trainer": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
        },
        "model": {
            "gradient_checkpointing": False,
            "max_seq_length": 2048,
            "use_cache": True,
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
    max_trials=4,
    max_probe_trials=1,
    max_full_trials=1,
    confirmation_runs=2,
    probe_steps=5,
    trial_timeout_seconds=60,
    campaign_timeout_seconds=600,
    max_gpu_seconds=None,
    minimum_progress_steps=100,
    metric_guards=None,
    max_peak_vram_bytes=None,
    workload_identity=None,
    permissions=None,
) -> RecoveryContract:
    if metric_guards is None:
        metric_guards = (
            MetricGuard("validation_loss", "minimize", 0.40, 0.05),
        )
    if workload_identity is None:
        workload_identity = WorkloadIdentity()
    return RecoveryContract(
        project="campaign-tests",
        source_run_id="source-oom-run",
        entrypoint=TrainingEntrypoint("training.entrypoint:train"),
        source_config=_source_config(),
        budget=RecoveryBudget(
            max_trials=max_trials,
            max_probe_trials=max_probe_trials,
            max_full_trials=max_full_trials,
            probe_steps=probe_steps,
            trial_timeout_seconds=trial_timeout_seconds,
            campaign_timeout_seconds=campaign_timeout_seconds,
            max_gpu_seconds=max_gpu_seconds,
        ),
        verification=VerificationRequirements(
            minimum_progress_steps=minimum_progress_steps,
            metric_guards=metric_guards,
            confirmation_runs=confirmation_runs,
            max_peak_vram_bytes=max_peak_vram_bytes,
            workload_identity=workload_identity,
        ),
        permissions=permissions,
    )


def _batch_candidate(
    proposal_id="candidate-batch-16",
    batch_size=16,
    accumulation_steps=2,
) -> CampaignCandidate:
    source = _source_config()
    manifest = discover_capabilities(source)
    proposal = InterventionProposal(
        proposal_id=proposal_id,
        policy_rule="halve_batch_preserve_effective_batch",
        changes=(
            InterventionChange("micro_batch_size", "decrease", batch_size),
            InterventionChange(
                "gradient_accumulation_steps",
                "increase",
                accumulation_steps,
            ),
        ),
        rationale="Reduce per-step activation memory using recorded OOM evidence.",
        expected_effect="Reduce activation memory while preserving batch intent.",
        evidence_refs=("EV-1", "EV-4"),
    )
    resolved = resolve_intervention(proposal, manifest, source)
    application = materialize_intervention(resolved, manifest, source)
    return CampaignCandidate(resolved, application)


def _checkpoint_candidate(
    proposal_id="candidate-checkpoint",
    *,
    authorize=False,
) -> CampaignCandidate:
    source = _source_config()
    manifest = discover_capabilities(source)
    proposal = InterventionProposal(
        proposal_id=proposal_id,
        policy_rule="enable_gradient_checkpointing",
        changes=(
            InterventionChange("gradient_checkpointing", "enable", True),
        ),
        rationale="Trade recomputation for lower activation storage.",
        expected_effect="Reduce stored activation memory.",
        evidence_refs=("EV-1",),
    )
    resolved = resolve_intervention(proposal, manifest, source)
    authorization = None
    if authorize and resolved.approval_required:
        authorization = InterventionAuthorization.approve(
            proposal,
            approved_by="test-engineer",
            reason="Explicitly approved for this campaign test.",
            approved_at=1_700_000_000,
        )
    application = materialize_intervention(
        resolved,
        manifest,
        source,
        authorization=authorization,
    )
    return CampaignCandidate(resolved, application)


def _sequence_candidate(
    proposal_id="candidate-sequence",
) -> CampaignCandidate:
    """Create an explicitly authorized semantic/medium-risk candidate."""
    source = _source_config()
    manifest = discover_capabilities(source)
    proposal = InterventionProposal(
        proposal_id=proposal_id,
        policy_rule="reduce_sequence_length",
        changes=(
            InterventionChange("sequence_length", "decrease", 1024),
        ),
        rationale="Sequence length contributes to attention memory.",
        expected_effect="Reduce attention and activation memory.",
        evidence_refs=("EV-1",),
    )
    resolved = resolve_intervention(proposal, manifest, source)
    authorization = InterventionAuthorization.approve(
        proposal,
        approved_by="test-engineer",
        reason="Explicit semantic-change authorization for this test.",
        approved_at=1_700_000_000,
    )
    application = materialize_intervention(
        resolved,
        manifest,
        source,
        authorization=authorization,
    )
    return CampaignCandidate(resolved, application)


class _Clock:
    def __init__(self, now=100.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Executor:
    """Build realistic TrialExecution objects from per-request specifications."""

    def __init__(self, behavior=None, clock=None):
        self.behavior = behavior
        self.clock = clock
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout_seconds):
        self.requests.append(request)
        self.timeouts.append(timeout_seconds)
        call_index = len(self.requests)
        spec = {}
        if self.behavior is not None:
            returned = self.behavior(request, call_index)
            if returned is not None:
                spec = dict(returned)
        if spec.get("raise") is not None:
            raise spec["raise"]
        if "raw_return" in spec:
            return spec["raw_return"]

        status = spec.get("status", "success")
        duration = float(spec.get("duration", 1.0))
        if self.clock is not None:
            self.clock.advance(spec.get("clock_advance", 0.0))
        worker_pid = spec.get("worker_pid", 10_000 + call_index)
        child_pid = spec.get("child_pid", worker_pid)
        result = None
        if status in {
            "success",
            "training_failed",
            "contract_error",
            "worker_error",
        } and not spec.get("result_none", False):
            exit_codes = {
                "success": EXIT_SUCCESS,
                "training_failed": EXIT_TRAINING_FAILED,
                "contract_error": EXIT_CONTRACT_ERROR,
                "worker_error": EXIT_WORKER_ERROR,
            }
            failure_class = spec.get("failure_class")
            if "failure_class" not in spec and status == "training_failed":
                failure_class = "cuda_out_of_memory"
            metrics = spec.get("metrics", {"validation_loss": 0.40})
            result_values = {
                "trial_id": request.trial_id,
                "project": request.project,
                "phase": request.phase,
                "status": status,
                "worker_exit_code": exit_codes[status],
                "run_id": request.run_id,
                "metrics": metrics,
                "failure_class": failure_class,
                "capsule_schema_version": (
                    "1.0" if status == "training_failed" else None
                ),
                "started_at": 1.0,
                "ended_at": 1.0 + duration,
                "duration_seconds": duration,
                "worker_pid": spec.get("result_worker_pid", worker_pid),
                "error": None if status == "success" else {"message": status},
                "campaign_id": request.campaign_id,
                "source_run_id": request.source_run_id,
            }
            result_values.update(spec.get("result_overrides", {}))
            result = TrialResult(**result_values)

        timed_out = status == "timeout"
        execution_values = {
            "trial_id": request.trial_id,
            "run_id": request.run_id,
            "project": request.project,
            "phase": request.phase,
            "status": status,
            "child_exit_code": (
                None if status in {"timeout", "launch_error"} else 0
            ),
            "child_pid": (
                None if status == "launch_error" else child_pid
            ),
            "timed_out": timed_out,
            "termination": "SIGTERM" if timed_out else None,
            "started_at": 1.0,
            "ended_at": 1.0 + duration,
            "duration_seconds": duration,
            "trial_directory": "/tmp/{}".format(request.trial_id),
            "request_path": "/tmp/{}/request.json".format(request.trial_id),
            "result_path": "/tmp/{}/result.json".format(request.trial_id),
            "stdout_path": "/tmp/{}/stdout.log".format(request.trial_id),
            "stderr_path": "/tmp/{}/stderr.log".format(request.trial_id),
            "execution_path": "/tmp/{}/execution.json".format(request.trial_id),
            "result": result,
            "error": (
                {"message": status}
                if status in {"timeout", "launch_error", "protocol_error"}
                else None
            ),
        }
        execution_values.update(spec.get("execution_overrides", {}))
        execution = TrialExecution(**execution_values)
        progress = spec.get(
            "progress",
            request.max_steps if request.phase == "probe" else 100,
        )
        observation = RunObservation(
            progress_steps=progress,
            peak_vram_bytes=spec.get("peak_vram_bytes"),
            workload_identity=spec.get("identity", WorkloadIdentity()),
            gpu_seconds=spec.get("gpu_seconds"),
        )
        return ExecutedTrial(execution, observation)


def _run(
    *,
    contract=None,
    candidates=None,
    executor=None,
    policy=None,
    clock=None,
    campaign_id="campaign-1",
) -> CampaignResult:
    contract = _contract() if contract is None else contract
    candidates = (_batch_candidate(),) if candidates is None else candidates
    executor = _Executor() if executor is None else executor
    policy = RankingPolicy("validation_loss") if policy is None else policy
    clock = _Clock() if clock is None else clock
    return run_campaign(
        contract,
        campaign_id=campaign_id,
        candidates=candidates,
        ranking_policy=policy,
        executor=executor,
        clock=clock,
    )


class RunObservationTests(unittest.TestCase):
    def test_round_trip_preserves_explicit_missing_values(self):
        observation = RunObservation(None, None)
        restored = RunObservation.from_dict(observation.to_dict())

        self.assertEqual(restored, observation)
        self.assertEqual(
            observation.to_dict()["schema"],
            {
                "name": CAMPAIGN_OBSERVATION_SCHEMA_NAME,
                "version": CAMPAIGN_OBSERVATION_SCHEMA_VERSION,
            },
        )

    def test_complete_observation_round_trips(self):
        observation = RunObservation(100, 12 * 1024**3, _identity(), 4.5)
        self.assertEqual(
            RunObservation.from_dict(observation.to_dict()),
            observation,
        )

    def test_negative_boolean_and_nonfinite_numbers_are_rejected(self):
        invalid = (
            {"progress_steps": -1},
            {"progress_steps": True},
            {"peak_vram_bytes": -1},
            {"peak_vram_bytes": False},
            {"gpu_seconds": -0.1},
            {"gpu_seconds": float("nan")},
            {"gpu_seconds": float("inf")},
        )
        defaults = {
            "progress_steps": None,
            "peak_vram_bytes": None,
            "gpu_seconds": None,
        }
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(CampaignError):
                    RunObservation(**{**defaults, **changes})

    def test_identity_type_is_required(self):
        with self.assertRaisesRegex(CampaignError, "WorkloadIdentity"):
            RunObservation(1, None, workload_identity={})

    def test_deserialization_rejects_unknown_fields_and_schema_changes(self):
        payload = RunObservation(None, None).to_dict()
        payload["inferred_metrics"] = {"validation_loss": 0.1}
        with self.assertRaisesRegex(CampaignError, "unknown fields"):
            RunObservation.from_dict(payload)

        payload = RunObservation(None, None).to_dict()
        payload["schema"]["version"] = "99.0"
        with self.assertRaisesRegex(CampaignError, "unsupported"):
            RunObservation.from_dict(payload)


class CampaignCandidateTests(unittest.TestCase):
    def test_candidate_seals_config_and_environment_inputs(self):
        source = _source_config()
        manifest = discover_capabilities(source)
        proposal = InterventionProposal(
            "candidate-sealed",
            "bounded_batch",
            (
                InterventionChange("micro_batch_size", "decrease", 16),
                InterventionChange(
                    "gradient_accumulation_steps", "increase", 2
                ),
            ),
            "Use OOM evidence to reduce activation memory.",
            "Reduce per-step memory.",
            ("EV-1",),
        )
        resolved = resolve_intervention(proposal, manifest, source)
        application = materialize_intervention(resolved, manifest, source)
        candidate = CampaignCandidate(resolved, application)

        application.config["trainer"]["per_device_train_batch_size"] = 1
        self.assertEqual(
            candidate.config["trainer"]["per_device_train_batch_size"],
            16,
        )
        returned = candidate.config
        returned["trainer"]["per_device_train_batch_size"] = 2
        self.assertEqual(
            candidate.config["trainer"]["per_device_train_batch_size"],
            16,
        )

    def test_candidate_id_and_digest_come_from_sealed_inputs(self):
        candidate = _batch_candidate()
        self.assertEqual(candidate.candidate_id, "candidate-batch-16")
        self.assertRegex(candidate.config_digest, r"^[a-f0-9]{64}$")
        self.assertEqual(
            candidate.to_dict()["candidate_config_digest"],
            candidate.config_digest,
        )

    def test_candidate_rejects_wrong_public_types(self):
        with self.assertRaisesRegex(CampaignError, "ResolvedIntervention"):
            CampaignCandidate({}, InterventionApplication("candidate-x", {}))
        with self.assertRaisesRegex(CampaignError, "InterventionApplication"):
            CampaignCandidate(_batch_candidate().resolved, {})

    def test_application_must_match_proposal_id(self):
        candidate = _batch_candidate()
        forged = InterventionApplication(
            "another-proposal",
            candidate.config,
            candidate.environment_patch,
        )
        with self.assertRaisesRegex(CampaignError, "proposal_id"):
            CampaignCandidate(candidate.resolved, forged)

    def test_required_authorization_cannot_be_omitted(self):
        # Sequence length is approval-required in the v1 capability catalog.
        source = _source_config()
        manifest = discover_capabilities(source)
        proposal = InterventionProposal(
            "candidate-sequence",
            "reduce_sequence_length",
            (InterventionChange("sequence_length", "decrease", 1024),),
            "Sequence length contributes to attention memory.",
            "Reduce attention memory.",
            ("EV-1",),
        )
        resolved = resolve_intervention(proposal, manifest, source)
        forged = InterventionApplication(
            proposal.proposal_id,
            {
                **source,
                "model": {**source["model"], "max_seq_length": 1024},
            },
        )
        with self.assertRaisesRegex(CampaignError, "lacks authorization"):
            CampaignCandidate(resolved, forged)

    def test_authorization_is_bound_to_exact_proposal_digest(self):
        first = _batch_candidate("candidate-first", 16, 2)
        second = _batch_candidate("candidate-second", 8, 4)
        authorization = InterventionAuthorization.approve(
            first.resolved.proposal,
            approved_by="test-engineer",
            reason="Test proposal binding.",
            approved_at=1_700_000_000,
        )
        forged = InterventionApplication(
            second.candidate_id,
            second.config,
            authorization=authorization,
        )
        with self.assertRaisesRegex(CampaignError, "another proposal"):
            CampaignCandidate(second.resolved, forged)


class CampaignTrialArtifactTests(unittest.TestCase):
    def _trial(self, **overrides):
        values = {
            "candidate_id": "candidate-1",
            "trial_id": "trial-1",
            "run_id": "run-1",
            "phase": "full",
            "status": "success",
            "request_digest": "1" * 64,
            "execution_manifest_digest": "2" * 64,
            "duration_seconds": 3.0,
            "gpu_seconds": 2.5,
            "progress_steps": 100,
            "peak_vram_bytes": 1024,
            "workload_identity": _identity(),
            "worker_pid": 1234,
            "failure_class": None,
            "metrics": {"validation_loss": 0.4},
        }
        values.update(overrides)
        return CampaignTrial(**values)

    def test_round_trip_and_schema(self):
        trial = self._trial()
        restored = CampaignTrial.from_dict(trial.to_dict())
        self.assertEqual(restored, trial)
        self.assertEqual(
            trial.to_dict()["schema"],
            {
                "name": CAMPAIGN_TRIAL_SCHEMA_NAME,
                "version": CAMPAIGN_TRIAL_SCHEMA_VERSION,
            },
        )

    def test_metrics_are_finite_and_immutable(self):
        source = {"validation_loss": 0.4}
        trial = self._trial(metrics=source)
        source["validation_loss"] = 9.0
        self.assertEqual(trial.metrics["validation_loss"], 0.4)
        with self.assertRaises(TypeError):
            trial.metrics["validation_loss"] = 0.1
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(CampaignError):
                    self._trial(metrics={"validation_loss": value})

    def test_status_phase_ids_and_digests_are_strict(self):
        invalid = (
            {"candidate_id": "bad id"},
            {"phase": "ranking"},
            {"status": "almost_success"},
            {"request_digest": "short"},
            {"execution_manifest_digest": "A" * 64},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(CampaignError):
                    self._trial(**changes)

    def test_succeeded_requires_success_without_failure_class(self):
        self.assertTrue(self._trial().succeeded)
        self.assertFalse(
            self._trial(failure_class="cuda_out_of_memory").succeeded
        )
        self.assertFalse(self._trial(status="training_failed").succeeded)

    def test_unknown_serialized_fields_are_rejected(self):
        payload = self._trial().to_dict()
        payload["recovery_score"] = 1.0
        with self.assertRaisesRegex(CampaignError, "unknown fields"):
            CampaignTrial.from_dict(payload)


class CampaignBudgetUsageTests(unittest.TestCase):
    def test_phase_counts_must_equal_attempted_count(self):
        with self.assertRaisesRegex(CampaignError, "equal"):
            CampaignBudgetUsage(4, 1, 1, 1, 1.0, 0.0, True)

    def test_values_are_nonnegative_finite_and_boolean(self):
        invalid = (
            {"attempted_trials": -1},
            {"probe_trials": True},
            {"elapsed_seconds": float("nan")},
            {"observed_gpu_seconds": -1},
            {"gpu_measurement_complete": 1},
        )
        defaults = {
            "attempted_trials": 0,
            "probe_trials": 0,
            "full_trials": 0,
            "confirmation_trials": 0,
            "elapsed_seconds": 0.0,
            "observed_gpu_seconds": 0.0,
            "gpu_measurement_complete": True,
        }
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(CampaignError):
                    CampaignBudgetUsage(**{**defaults, **changes})

    def test_round_trip(self):
        usage = CampaignBudgetUsage(4, 1, 1, 2, 5.0, 4.0, True)
        self.assertEqual(CampaignBudgetUsage.from_dict(usage.to_dict()), usage)


class CampaignExecutionTests(unittest.TestCase):
    def test_happy_path_runs_probe_full_and_declared_confirmations(self):
        executor = _Executor()
        result = _run(executor=executor)

        self.assertTrue(result.verified)
        self.assertEqual(result.status, "verified")
        self.assertEqual(result.stopped_reason, "verified_recovery")
        self.assertEqual(
            [request.phase for request in executor.requests],
            ["probe", "full", "confirmation", "confirmation"],
        )
        self.assertEqual(executor.requests[0].max_steps, 5)
        self.assertTrue(
            all(request.max_steps is None for request in executor.requests[1:])
        )
        self.assertEqual(result.usage.attempted_trials, 4)
        self.assertEqual(result.usage.probe_trials, 1)
        self.assertEqual(result.usage.full_trials, 1)
        self.assertEqual(result.usage.confirmation_trials, 2)

    def test_every_request_is_bound_to_contract_and_campaign(self):
        executor = _Executor()
        contract = _contract()
        _run(contract=contract, executor=executor, campaign_id="campaign-bound")

        for request in executor.requests:
            self.assertEqual(request.project, contract.project)
            self.assertEqual(request.campaign_id, "campaign-bound")
            self.assertEqual(request.source_run_id, contract.source_run_id)
            self.assertEqual(request.entrypoint, contract.entrypoint)

    def test_trial_and_run_ids_are_unique_across_phases(self):
        result = _run()
        self.assertEqual(len({item.trial_id for item in result.trials}), 4)
        self.assertEqual(len({item.run_id for item in result.trials}), 4)
        self.assertEqual(len({item.request_digest for item in result.trials}), 4)
        self.assertEqual(
            len({item.execution_manifest_digest for item in result.trials}),
            4,
        )

    def test_ids_are_deterministic_for_reconstructible_campaign_inputs(self):
        first = _run()
        second = _run()
        self.assertEqual(
            [item.trial_id for item in first.trials],
            [item.trial_id for item in second.trials],
        )
        self.assertEqual(
            [item.run_id for item in first.trials],
            [item.run_id for item in second.trials],
        )

    def test_probe_failure_prevents_full_and_confirmation_trials(self):
        executor = _Executor(
            lambda request, _: (
                {"status": "training_failed"}
                if request.phase == "probe"
                else {}
            )
        )
        result = _run(executor=executor)
        self.assertEqual(result.status, "not_recovered")
        self.assertEqual(result.stopped_reason, "no_probe_survivors")
        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(result.probe_survivor_ids, ())

    def test_timeout_probe_does_not_survive(self):
        executor = _Executor(
            lambda request, _: {"status": "timeout"}
        )
        result = _run(executor=executor)
        self.assertEqual(result.stopped_reason, "no_probe_survivors")
        self.assertEqual(result.trials[0].status, "timeout")

    def test_success_without_worker_result_fails_closed(self):
        executor = _Executor(lambda request, _: {"result_none": True})
        result = _run(executor=executor)
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stopped_reason, "trial_evidence_mismatch")
        self.assertEqual(result.usage.attempted_trials, 1)
        self.assertEqual(result.trials, ())

    def test_full_failure_is_ranked_as_ineligible_not_recovered(self):
        executor = _Executor(
            lambda request, _: (
                {"status": "training_failed"}
                if request.phase == "full"
                else {}
            )
        )
        result = _run(executor=executor)
        self.assertEqual(result.status, "not_recovered")
        self.assertEqual(result.stopped_reason, "no_eligible_full_trials")
        self.assertIsNotNone(result.ranking)
        self.assertEqual(result.ranking.confirmation_order, ())
        self.assertEqual(len(executor.requests), 2)

    def test_full_metric_regression_blocks_confirmation(self):
        executor = _Executor(
            lambda request, _: (
                {"metrics": {"validation_loss": 0.9}}
                if request.phase == "full"
                else {}
            )
        )
        result = _run(executor=executor)
        self.assertEqual(result.stopped_reason, "no_eligible_full_trials")
        self.assertEqual(len(executor.requests), 2)

    def test_missing_full_progress_blocks_confirmation(self):
        executor = _Executor(
            lambda request, _: (
                {"progress": None} if request.phase == "full" else {}
            )
        )
        result = _run(executor=executor)
        self.assertEqual(result.stopped_reason, "no_eligible_full_trials")

    def test_peak_vram_contract_is_applied_before_confirmation(self):
        contract = _contract(max_peak_vram_bytes=100)
        executor = _Executor(
            lambda request, _: (
                {"peak_vram_bytes": 101}
                if request.phase == "full"
                else {"peak_vram_bytes": 90}
            )
        )
        result = _run(contract=contract, executor=executor)
        self.assertEqual(result.stopped_reason, "no_eligible_full_trials")

    def test_workload_identity_mismatch_blocks_confirmation(self):
        contract = _contract(workload_identity=_identity())
        wrong = replace(_identity(), dataset_fingerprint="different")
        executor = _Executor(
            lambda request, _: (
                {"identity": wrong}
                if request.phase == "full"
                else {"identity": _identity()}
            )
        )
        result = _run(contract=contract, executor=executor)
        self.assertEqual(result.stopped_reason, "no_eligible_full_trials")

    def test_confirmation_failure_is_rejected_by_verifier(self):
        executor = _Executor(
            lambda request, _: (
                {"status": "training_failed"}
                if request.phase == "confirmation"
                else {}
            )
        )
        result = _run(executor=executor)
        self.assertEqual(result.status, "not_recovered")
        self.assertEqual(
            result.stopped_reason,
            "all_ranked_candidates_rejected",
        )
        self.assertEqual(result.verifications[0].verdict, "rejected")

    def test_confirmation_metric_regression_is_not_a_recovery(self):
        executor = _Executor(
            lambda request, _: (
                {"metrics": {"validation_loss": 0.8}}
                if request.phase == "confirmation"
                else {}
            )
        )
        result = _run(executor=executor)
        self.assertFalse(result.verified)
        self.assertEqual(result.verifications[0].verdict, "rejected")

    def test_confirmation_missing_required_evidence_is_not_verified(self):
        contract = _contract(max_peak_vram_bytes=1_000)
        executor = _Executor(
            lambda request, _: (
                {"peak_vram_bytes": None}
                if request.phase == "confirmation"
                else {"peak_vram_bytes": 900}
            )
        )
        result = _run(contract=contract, executor=executor)
        self.assertFalse(result.verified)
        self.assertEqual(
            result.verifications[0].verdict,
            "insufficient_evidence",
        )

    def test_two_candidates_are_ranked_then_confirmed_in_rank_order(self):
        first = _batch_candidate("candidate-batch-16", 16, 2)
        second = _batch_candidate("candidate-batch-8", 8, 4)

        def behavior(request, _):
            batch = request.config["trainer"]["per_device_train_batch_size"]
            if request.phase == "full":
                return {
                    "metrics": {
                        "validation_loss": 0.41 if batch == 16 else 0.43
                    }
                }
            return {}

        contract = _contract(
            max_trials=6,
            max_probe_trials=2,
            max_full_trials=2,
        )
        executor = _Executor(behavior)
        result = _run(
            contract=contract,
            candidates=(first, second),
            executor=executor,
        )
        self.assertTrue(result.verified)
        self.assertEqual(
            result.ranking.confirmation_order,
            ("candidate-batch-16", "candidate-batch-8"),
        )
        confirmed_batches = [
            request.config["trainer"]["per_device_train_batch_size"]
            for request in executor.requests
            if request.phase == "confirmation"
        ]
        self.assertEqual(confirmed_batches, [16, 16])

    def test_rejected_first_candidate_allows_next_ranked_candidate(self):
        first = _batch_candidate("candidate-batch-16", 16, 2)
        second = _batch_candidate("candidate-batch-8", 8, 4)

        def behavior(request, _):
            batch = request.config["trainer"]["per_device_train_batch_size"]
            if request.phase == "full":
                return {
                    "metrics": {
                        "validation_loss": 0.40 if batch == 16 else 0.42
                    }
                }
            if request.phase == "confirmation":
                return {
                    "metrics": {
                        "validation_loss": 0.90 if batch == 16 else 0.40
                    }
                }
            return {}

        contract = _contract(
            max_trials=8,
            max_probe_trials=2,
            max_full_trials=2,
        )
        executor = _Executor(behavior)
        result = _run(
            contract=contract,
            candidates=(first, second),
            executor=executor,
        )
        self.assertTrue(result.verified)
        self.assertEqual(result.verified_candidate_id, "candidate-batch-8")
        self.assertEqual(
            [item.verdict for item in result.verifications],
            ["rejected", "verified"],
        )
        self.assertEqual(result.usage.confirmation_trials, 4)

    def test_confirmation_set_is_not_partially_started_without_budget(self):
        first = _batch_candidate("candidate-batch-16", 16, 2)
        second = _batch_candidate("candidate-batch-8", 8, 4)

        def behavior(request, _):
            batch = request.config["trainer"]["per_device_train_batch_size"]
            if request.phase == "full":
                return {
                    "metrics": {
                        "validation_loss": 0.40 if batch == 16 else 0.42
                    }
                }
            if request.phase == "confirmation" and batch == 16:
                return {"metrics": {"validation_loss": 0.9}}
            return {}

        contract = _contract(
            max_trials=7,
            max_probe_trials=2,
            max_full_trials=2,
        )
        executor = _Executor(behavior)
        result = _run(
            contract=contract,
            candidates=(first, second),
            executor=executor,
        )
        self.assertEqual(result.status, "stopped")
        self.assertEqual(
            result.stopped_reason,
            "confirmation_budget_exhausted",
        )
        confirmed_batches = [
            request.config["trainer"]["per_device_train_batch_size"]
            for request in executor.requests
            if request.phase == "confirmation"
        ]
        self.assertEqual(confirmed_batches, [16, 16])

    def test_probe_cap_limits_candidates_in_policy_order(self):
        first = _batch_candidate("candidate-batch-16", 16, 2)
        second = _batch_candidate("candidate-batch-8", 8, 4)
        executor = _Executor()
        result = _run(candidates=(first, second), executor=executor)
        self.assertTrue(result.verified)
        probed = [
            request.config["trainer"]["per_device_train_batch_size"]
            for request in executor.requests
            if request.phase == "probe"
        ]
        self.assertEqual(probed, [16])
        self.assertEqual(result.planned_candidate_ids, (
            "candidate-batch-16",
            "candidate-batch-8",
        ))

    def test_empty_candidate_set_uses_no_compute(self):
        executor = _Executor()
        result = _run(candidates=(), executor=executor)
        self.assertEqual(result.status, "not_recovered")
        self.assertEqual(result.stopped_reason, "no_candidates")
        self.assertEqual(executor.requests, [])
        self.assertEqual(result.usage.attempted_trials, 0)

    def test_executor_exception_is_a_visible_stop_and_consumes_attempt(self):
        executor = _Executor(
            lambda request, _: {"raise": RuntimeError("executor failed")}
        )
        result = _run(executor=executor)
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stopped_reason, "trial_executor_error")
        self.assertEqual(result.usage.attempted_trials, 1)
        self.assertEqual(result.trials, ())

    def test_non_execution_return_is_rejected(self):
        executor = _Executor(lambda request, _: {"raw_return": {}})
        result = _run(executor=executor)
        self.assertEqual(result.stopped_reason, "trial_evidence_mismatch")

    def test_execution_request_binding_mismatch_is_rejected(self):
        mismatches = (
            {"trial_id": "other-trial"},
            {"run_id": "other-run"},
            {"project": "other-project"},
            {"phase": "full"},
        )
        for override in mismatches:
            with self.subTest(override=override):
                executor = _Executor(
                    lambda request, _, override=override: {
                        "execution_overrides": override
                    }
                )
                result = _run(executor=executor)
                self.assertEqual(
                    result.stopped_reason,
                    "trial_evidence_mismatch",
                )

    def test_worker_result_binding_mismatch_is_rejected(self):
        mismatches = (
            {"trial_id": "other-trial"},
            {"run_id": "other-run"},
            {"project": "other-project"},
            {"phase": "full"},
            {"campaign_id": "other-campaign"},
            {"source_run_id": "other-source"},
        )
        for override in mismatches:
            with self.subTest(override=override):
                executor = _Executor(
                    lambda request, _, override=override: {
                        "result_overrides": override
                    }
                )
                result = _run(executor=executor)
                self.assertEqual(
                    result.stopped_reason,
                    "trial_evidence_mismatch",
                )

    def test_worker_pid_must_match_launched_child(self):
        executor = _Executor(
            lambda request, _: {"result_worker_pid": 999_999}
        )
        result = _run(executor=executor)
        self.assertEqual(result.stopped_reason, "trial_evidence_mismatch")

    def test_trial_timeout_is_passed_to_every_executor_call(self):
        executor = _Executor()
        _run(executor=executor)
        self.assertEqual(executor.timeouts, [60, 60, 60, 60])

    def test_remaining_gpu_budget_conservatively_caps_timeout(self):
        contract = _contract(max_gpu_seconds=7)
        executor = _Executor(
            lambda request, _: {"gpu_seconds": 1.0}
        )
        _run(contract=contract, executor=executor)
        self.assertEqual(executor.timeouts, [7.0, 6.0, 5.0, 4.0])

    def test_missing_gpu_usage_stops_when_gpu_budget_is_declared(self):
        contract = _contract(max_gpu_seconds=100)
        executor = _Executor()
        result = _run(contract=contract, executor=executor)
        self.assertEqual(result.status, "stopped")
        self.assertEqual(
            result.stopped_reason,
            "gpu_budget_evidence_missing",
        )
        self.assertEqual(len(executor.requests), 1)

    def test_exhausted_gpu_budget_stops_after_recorded_trial(self):
        contract = _contract(max_gpu_seconds=5)
        executor = _Executor(
            lambda request, _: {"gpu_seconds": 5.0}
        )
        result = _run(contract=contract, executor=executor)
        self.assertEqual(result.stopped_reason, "gpu_budget_exhausted")
        self.assertEqual(result.usage.observed_gpu_seconds, 5.0)
        self.assertTrue(result.usage.gpu_measurement_complete)

    def test_campaign_wall_deadline_stops_after_current_trial(self):
        clock = _Clock()
        contract = _contract(
            trial_timeout_seconds=5,
            campaign_timeout_seconds=5,
        )
        executor = _Executor(
            lambda request, _: {"clock_advance": 5.0},
            clock=clock,
        )
        result = _run(contract=contract, executor=executor, clock=clock)
        self.assertEqual(result.stopped_reason, "campaign_timeout")
        self.assertEqual(len(executor.requests), 1)

    def test_accounted_trial_duration_cannot_hide_elapsed_budget(self):
        # The injected clock does not move, but authenticated execution
        # durations still consume campaign time.
        contract = _contract(
            trial_timeout_seconds=5,
            campaign_timeout_seconds=5,
        )
        executor = _Executor(lambda request, _: {"duration": 5.0})
        result = _run(contract=contract, executor=executor)
        self.assertEqual(result.stopped_reason, "campaign_timeout")


class CampaignInputBoundaryTests(unittest.TestCase):
    def test_duplicate_candidate_ids_are_rejected_before_compute(self):
        first = _batch_candidate("candidate-duplicate", 16, 2)
        second = _batch_candidate("candidate-duplicate", 8, 4)
        executor = _Executor()
        with self.assertRaisesRegex(CampaignError, "ids must be unique"):
            _run(candidates=(first, second), executor=executor)
        self.assertEqual(executor.requests, [])

    def test_duplicate_candidate_configurations_are_rejected(self):
        first = _batch_candidate("candidate-first", 16, 2)
        second = _batch_candidate("candidate-second", 16, 2)
        with self.assertRaisesRegex(CampaignError, "unique configurations"):
            _run(candidates=(first, second))

    def test_application_cannot_smuggle_undeclared_config_changes(self):
        candidate = _batch_candidate()
        forged_config = candidate.config
        forged_config["runtime"]["precision"] = "bf16"
        forged_application = InterventionApplication(
            candidate.candidate_id,
            forged_config,
            candidate.environment_patch,
        )
        forged = CampaignCandidate(candidate.resolved, forged_application)
        with self.assertRaisesRegex(CampaignError, "outside"):
            _run(candidates=(forged,))

    def test_application_environment_must_match_resolved_changes(self):
        candidate = _batch_candidate()
        forged_application = InterventionApplication(
            candidate.candidate_id,
            candidate.config,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        forged = CampaignCandidate(candidate.resolved, forged_application)
        with self.assertRaisesRegex(CampaignError, "environment patch"):
            _run(candidates=(forged,))

    def test_contract_scope_is_rechecked_before_compute(self):
        candidate = _sequence_candidate()
        executor = _Executor()
        with self.assertRaises(ContractScopeError):
            _run(candidates=(candidate,), executor=executor)
        self.assertEqual(executor.requests, [])

    def test_wrong_public_input_types_fail_before_compute(self):
        cases = (
            {"contract": {}},
            {"candidates": ({},)},
            {"policy": {}},
            {"executor": "not-callable"},
            {"clock": "not-callable"},
            {"campaign_id": "bad id"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                kwargs = {
                    "contract": _contract(),
                    "candidates": (_batch_candidate(),),
                    "policy": RankingPolicy("validation_loss"),
                    "executor": _Executor(),
                    "clock": _Clock(),
                    "campaign_id": "campaign-1",
                }
                kwargs.update(changes)
                with self.assertRaises((CampaignError, ValueError)):
                    _run(**kwargs)

    def test_policy_primary_metric_must_exist_in_contract(self):
        with self.assertRaisesRegex(ValueError, "metric guard"):
            _run(policy=RankingPolicy("untracked_metric"))

    def test_nonfinite_initial_clock_is_rejected(self):
        with self.assertRaisesRegex(CampaignError, "non-finite"):
            _run(clock=lambda: float("nan"))


class CampaignResultArtifactTests(unittest.TestCase):
    def test_verified_result_round_trips_with_stable_digest(self):
        result = _run()
        encoded = result.to_json()
        restored = CampaignResult.from_json(encoded)

        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(
            campaign_result_digest(restored),
            campaign_result_digest(result),
        )
        payload = json.loads(encoded)
        self.assertEqual(
            payload["schema"],
            {
                "name": CAMPAIGN_RESULT_SCHEMA_NAME,
                "version": CAMPAIGN_RESULT_SCHEMA_VERSION,
            },
        )
        self.assertTrue(payload["invariants"]["ranking_is_provisional"])
        self.assertTrue(
            payload["invariants"]["verifier_is_only_recovery_authority"]
        )

    def test_result_and_nested_collections_are_frozen(self):
        result = _run()
        with self.assertRaises(FrozenInstanceError):
            result.status = "not_recovered"
        with self.assertRaises(TypeError):
            result.trials[-1].metrics["validation_loss"] = 0.0

    def test_campaign_status_catalog_is_closed(self):
        self.assertEqual(
            CAMPAIGN_STATUSES,
            frozenset({"verified", "not_recovered", "stopped"}),
        )

    def test_verified_flag_cannot_disagree_with_status(self):
        payload = json.loads(_run().to_json())
        payload["verified"] = False
        with self.assertRaisesRegex(CampaignError, "verified flag"):
            CampaignResult.from_dict(payload)

    def test_verified_status_requires_exactly_one_verified_report(self):
        payload = json.loads(_run().to_json())
        payload["verifications"] = []
        with self.assertRaisesRegex(CampaignError, "verified report"):
            CampaignResult.from_dict(payload)

    def test_nonverified_status_cannot_hide_a_verified_report(self):
        payload = json.loads(_run().to_json())
        payload["status"] = "not_recovered"
        payload["stopped_reason"] = "all_ranked_candidates_rejected"
        payload["verified"] = False
        payload["verified_candidate_id"] = None
        payload["verified_run_ids"] = []
        with self.assertRaisesRegex(CampaignError, "cannot be hidden"):
            CampaignResult.from_dict(payload)

    def test_nonverified_result_cannot_claim_verified_ids(self):
        result = _run(
            executor=_Executor(
                lambda request, _: (
                    {"status": "training_failed"}
                    if request.phase == "probe"
                    else {}
                )
            )
        )
        payload = json.loads(result.to_json())
        payload["verified_candidate_id"] = "candidate-batch-16"
        payload["verified_run_ids"] = ["run-forged"]
        with self.assertRaisesRegex(CampaignError, "cannot expose"):
            CampaignResult.from_dict(payload)

    def test_verified_candidate_and_run_ids_must_match_verifier(self):
        original = json.loads(_run().to_json())
        mutations = (
            {"verified_candidate_id": "another-candidate"},
            {"verified_run_ids": ["run-forged-a", "run-forged-b"]},
        )
        for changes in mutations:
            with self.subTest(changes=changes):
                payload = deepcopy(original)
                payload.update(changes)
                with self.assertRaises(CampaignError):
                    CampaignResult.from_dict(payload)

    def test_verification_must_reference_recorded_confirmation_trials(self):
        payload = json.loads(_run().to_json())
        report = payload["verifications"][0]
        old_run_id = report["confirmation_run_ids"][0]
        forged_run_id = "run-unrecorded-confirmation"
        report["confirmation_run_ids"][0] = forged_run_id
        report["checks"] = [
            {
                **check,
                "run_id": (
                    forged_run_id if check.get("run_id") == old_run_id else check.get("run_id")
                ),
            }
            for check in report["checks"]
        ]
        payload["verified_run_ids"][0] = forged_run_id
        with self.assertRaisesRegex(CampaignError, "unrecorded"):
            CampaignResult.from_dict(payload)

    def test_verified_recovery_reason_requires_verified_status(self):
        result = _run(
            executor=_Executor(
                lambda request, _: (
                    {"status": "training_failed"}
                    if request.phase == "probe"
                    else {}
                )
            )
        )
        payload = json.loads(result.to_json())
        payload["stopped_reason"] = "verified_recovery"
        with self.assertRaisesRegex(CampaignError, "requires verified"):
            CampaignResult.from_dict(payload)

    def test_stopped_reason_must_be_machine_readable(self):
        payload = json.loads(_run().to_json())
        payload["stopped_reason"] = "Looks good!"
        with self.assertRaisesRegex(CampaignError, "machine-readable"):
            CampaignResult.from_dict(payload)

    def test_duplicate_trial_run_request_and_execution_ids_are_rejected(self):
        original = json.loads(_run().to_json())
        fields = (
            "trial_id",
            "run_id",
            "request_digest",
            "execution_manifest_digest",
        )
        for field_name in fields:
            with self.subTest(field_name=field_name):
                payload = deepcopy(original)
                payload["trials"][1][field_name] = payload["trials"][0][field_name]
                with self.assertRaisesRegex(CampaignError, "unique"):
                    CampaignResult.from_dict(payload)

    def test_trial_and_report_candidates_must_be_planned(self):
        original = json.loads(_run().to_json())
        payload = deepcopy(original)
        payload["trials"][0]["candidate_id"] = "unknown-candidate"
        with self.assertRaisesRegex(CampaignError, "unknown candidate"):
            CampaignResult.from_dict(payload)

        payload = deepcopy(original)
        payload["verifications"][0]["candidate_id"] = "unknown-candidate"
        with self.assertRaises(CampaignError):
            CampaignResult.from_dict(payload)

    def test_invariants_schema_and_unknown_fields_cannot_be_changed(self):
        original = json.loads(_run().to_json())
        mutations = []
        payload = deepcopy(original)
        payload["invariants"]["ranking_is_provisional"] = False
        mutations.append(payload)
        payload = deepcopy(original)
        payload["schema"]["version"] = "99.0"
        mutations.append(payload)
        payload = deepcopy(original)
        payload["ai_recommendation"] = "promote candidate"
        mutations.append(payload)
        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaises(CampaignError):
                    CampaignResult.from_dict(payload)

    def test_invalid_json_and_wrong_digest_input_are_rejected(self):
        with self.assertRaisesRegex(CampaignError, "invalid campaign result JSON"):
            CampaignResult.from_json("not-json")
        with self.assertRaisesRegex(CampaignError, "CampaignResult"):
            campaign_result_digest({})


if __name__ == "__main__":
    unittest.main()