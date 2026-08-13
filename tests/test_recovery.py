"""Integration acceptance tests for WatcherML's public OOM recovery API.

These tests cover the boundary assembled by ``watcherml.recovery``.  The
individual policy, contract, runner, campaign, ranking, and verifier modules
have their own unit suites.  Here we prove that the public workflow connects
those pieces without restoring any legacy in-process ``train_fn`` retry,
heuristic recovery score, LLM planner, or unverified "best run" claim.

The preparation-only tests use a valid persisted v1 OOM capsule and perform no
training.  Campaign tests invoke the real ``watcherml._trial_worker`` in fresh
Python processes and inspect the resulting SQLite rows and immutable artifact.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import sys
import tempfile
import textwrap
import unittest
from copy import deepcopy
from pathlib import Path

from watcherml.capsule import build_evidence_index
from watcherml.entrypoint import EntrypointSignatureError, TrainingEntrypoint
from watcherml.interventions import InterventionAuthorization
from watcherml.recovery import (
    RECOVERY_PREPARATION_SCHEMA_NAME,
    RECOVERY_PREPARATION_SCHEMA_VERSION,
    RECOVERY_RESULT_SCHEMA_NAME,
    RECOVERY_RESULT_SCHEMA_VERSION,
    RecoveryIntegrationError,
    RecoveryPreparation,
    RecoveryResult,
    prepare_oom_recovery,
    preparation_digest,
    print_recovery_summary,
    recover_from_oom,
    recovery_result_digest,
    run_prepared_recovery,
)
from watcherml.recovery_contract import (
    InterventionPermissions,
    MetricGuard,
    RecoveryBudget,
    VerificationRequirements,
)
from watcherml.storage import Storage


PROJECT = "recovery-integration"
SOURCE_RUN_ID = "source-oom-run"


TRAINING_MODULE = """
def train(config, max_steps=None):
    batch_size = config.get("batch_size")
    sequence_length = config.get("max_seq_length")
    if batch_size is not None and batch_size > config.get("oom_batch_limit", 16):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    if sequence_length is not None and sequence_length > config.get("oom_sequence_limit", 1024):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    steps = max_steps if max_steps is not None else config.get("full_steps", 20)
    return {
        "validation_loss": config.get("trial_validation_loss", 0.4),
        "steps_completed": steps,
        "throughput": 100.0,
    }
"""


ALWAYS_OOM_MODULE = """
def train(config, max_steps=None):
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
"""


BAD_PROGRESS_MODULE = """
def train(config, max_steps=None):
    if max_steps is not None:
        return {"validation_loss": 0.4, "steps_completed": max_steps}
    return {"validation_loss": 0.4, "steps_completed": 1.5}
"""


CUSTOM_PROGRESS_MODULE = """
def train(config, max_steps=None):
    steps = max_steps if max_steps is not None else config.get("full_steps", 20)
    return {"validation_loss": 0.4, "examples_seen": steps}
"""


UNBOUNDED_MODULE = """
def train(config):
    return {"validation_loss": 0.4, "steps_completed": 20}
"""


def _capsule(config: dict, *, run_id: str = SOURCE_RUN_ID,
             project: str = PROJECT, failure_class: str = "cuda_out_of_memory") -> dict:
    """Return a valid deterministic failure-capsule v1 fixture."""
    evidence = {
        "config": deepcopy(config),
        "training_state": {"last_logged_step": 41},
        "runtime": {"pid": 1234, "working_directory": "/project"},
        "resource_state_at_failure": {"vram_used_mib_peak": 15_500},
        "gpu": {
            "available": True,
            "gpus": [{"name": "test-gpu", "memory_total_mib": 16_384}],
        },
        "framework": {
            "python_version": "3.12.6",
            "torch_available": True,
            "cuda_available": True,
            "bf16_supported": True,
            "allocated_bytes": 8 * 1024**3,
            "reserved_bytes": 12 * 1024**3,
            "allocator_config": "max_split_size_mb:128",
        },
        "environment": {"fingerprint": "source-environment"},
        "git": {"available": True, "commit": "abc123"},
        "dataset": {"fingerprint": "dataset-example"},
        "recent_metrics": [{"name": "loss", "value": 0.8, "step": 41}],
        "notebook_cells_executed": None,
    }
    classification = {
        "rule": failure_class,
        "rule_version": "1.0",
        "match_kind": "deterministic",
        "recoverable_by_bounded_trial": failure_class == "cuda_out_of_memory",
        "evidence_ids": ["EV-1", "EV-2", "EV-4", "EV-5", "EV-6"],
    }
    failure = {
        "class": failure_class,
        "exception_type": "RuntimeError",
        "message": "CUDA out of memory. Tried to allocate 2.00 GiB",
        "traceback": "Traceback: CUDA out of memory",
        "classification": classification,
    }
    return {
        "schema": {"name": "watcherml.failure-capsule", "version": "1.0"},
        "run_id": run_id,
        "project": project,
        "captured_at": 1_800_000_000.0,
        "failure": failure,
        "failure_class": failure_class,
        "evidence": evidence,
        "evidence_index": build_evidence_index(evidence),
        "capture": {"score": 10, "maximum": 10, "present": [], "missing": []},
    }


def _seed_source(storage: Storage, config: dict, *, run_id: str = SOURCE_RUN_ID,
                 project: str = PROJECT, failure_class: str = "cuda_out_of_memory") -> dict:
    capsule = _capsule(
        config,
        run_id=run_id,
        project=project,
        failure_class=failure_class,
    )
    storage.upsert_run(
        run_id,
        project=project,
        config_json=config,
        started_at=1_800_000_000.0,
        ended_at=1_800_000_001.0,
        duration_seconds=1.0,
        exit_status="failed",
        git_json={"available": True, "commit": "abc123"},
        env_json={"fingerprint": "source-environment"},
        gpu_json={"available": True},
        resource_json={"vram_used_mib_peak": 15_500},
    )
    storage.save_failure(
        run_id,
        "RuntimeError",
        capsule["failure"]["message"],
        capsule["failure"]["traceback"],
        capsule["failure"]["classification"],
        capsule["evidence"],
        capsule=capsule,
    )
    return capsule


def _verification(*, confirmations: int = 2) -> VerificationRequirements:
    return VerificationRequirements(
        minimum_progress_steps=20,
        metric_guards=(
            MetricGuard(
                name="validation_loss",
                direction="minimize",
                baseline_value=0.5,
                max_regression=0.1,
            ),
        ),
        confirmation_runs=confirmations,
    )


def _budget(*, confirmations: int = 2) -> RecoveryBudget:
    return RecoveryBudget(
        max_trials=2 + confirmations,
        max_probe_trials=1,
        max_full_trials=1,
        probe_steps=3,
        trial_timeout_seconds=20,
        campaign_timeout_seconds=60,
    )


class _IntegrationFixture:
    def __init__(self, *, config: dict | None = None, module_source: str = TRAINING_MODULE):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.project_root = self.root / "project"
        self.project_root.mkdir()
        self.storage = Storage(str(self.root / "storage"))
        self.write_module("recovery_training", module_source)
        self.config = config or {
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "oom_batch_limit": 16,
            "full_steps": 20,
            "trial_validation_loss": 0.4,
            "model_name": "example/model",
            "dataset": {"fingerprint": "dataset-example"},
        }
        self.capsule = _seed_source(self.storage, self.config)

    def write_module(self, name: str, source: str) -> None:
        (self.project_root / (name + ".py")).write_text(
            textwrap.dedent(source).lstrip(),
            encoding="utf-8",
        )
        sys.modules.pop(name, None)

    @property
    def entrypoint(self) -> str:
        return "recovery_training:train"

    def prepare(self, **overrides) -> RecoveryPreparation:
        arguments = {
            "failed_run_id": SOURCE_RUN_ID,
            "entrypoint": self.entrypoint,
            "verification": _verification(),
            "budget": _budget(),
            "storage": self.storage,
            "project_root": self.project_root,
        }
        arguments.update(overrides)
        return prepare_oom_recovery(**arguments)

    def run(self, preparation: RecoveryPreparation | None = None, **overrides) -> RecoveryResult:
        arguments = {
            "preparation": preparation or self.prepare(),
            "storage": self.storage,
            "project_root": self.project_root,
            "trials_root": self.root / "trials",
            "print_summary": False,
        }
        arguments.update(overrides)
        return run_prepared_recovery(**arguments)

    def close(self) -> None:
        self.storage._conn.close()
        self._temporary.cleanup()


class RecoveryIntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _IntegrationFixture()

    def tearDown(self) -> None:
        self.fixture.close()


class PreparationTests(RecoveryIntegrationTestCase):
    def test_preparation_is_zero_compute_and_does_not_create_campaign_rows(self):
        preparation = self.fixture.prepare()

        self.assertFalse(preparation.to_dict()["compute_started"])
        self.assertEqual(self.fixture.storage.list_recovery_campaigns(), [])
        self.assertFalse((self.fixture.root / "trials").exists())

    def test_preparation_seals_capsule_contract_capabilities_and_policy(self):
        preparation = self.fixture.prepare()

        self.assertEqual(preparation.contract.source_run_id, SOURCE_RUN_ID)
        self.assertEqual(preparation.contract.project, PROJECT)
        self.assertEqual(preparation.contract.entrypoint.target, self.fixture.entrypoint)
        self.assertTrue(preparation.entrypoint_validation["supports_max_steps"])
        self.assertEqual(len(preparation.capsule_digest), 64)
        self.assertGreaterEqual(len(preparation.policy_plan.proposals), 1)
        self.assertIn(
            "halve_batch_preserve_effective_batch",
            [item.policy_rule for item in preparation.policy_plan.proposals],
        )

    def test_string_and_training_entrypoint_forms_are_equivalent(self):
        as_string = self.fixture.prepare()
        as_object = self.fixture.prepare(
            entrypoint=TrainingEntrypoint(self.fixture.entrypoint)
        )

        self.assertEqual(as_string.contract.entrypoint, as_object.contract.entrypoint)
        self.assertEqual(as_string.policy_plan, as_object.policy_plan)

    def test_preparation_round_trip_and_digest_are_stable(self):
        preparation = self.fixture.prepare()
        restored = RecoveryPreparation.from_json(preparation.to_json())

        self.assertEqual(restored, preparation)
        self.assertEqual(preparation_digest(restored), preparation_digest(preparation))
        self.assertEqual(
            preparation.to_dict()["schema"],
            {
                "name": RECOVERY_PREPARATION_SCHEMA_NAME,
                "version": RECOVERY_PREPARATION_SCHEMA_VERSION,
            },
        )

    def test_preparation_rejects_unknown_fields_and_tampered_invariants(self):
        original = self.fixture.prepare().to_dict()
        cases = []
        unknown = deepcopy(original)
        unknown["surprise"] = True
        cases.append(unknown)
        compute = deepcopy(original)
        compute["compute_started"] = True
        cases.append(compute)
        invariant = deepcopy(original)
        invariant["invariants"]["deterministic_policy"] = False
        cases.append(invariant)
        digest = deepcopy(original)
        digest["contract_digest"] = "0" * 64
        cases.append(digest)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(RecoveryIntegrationError):
                    RecoveryPreparation.from_dict(payload)

    def test_preparation_rejects_unknown_missing_and_non_oom_runs(self):
        with self.assertRaisesRegex(RecoveryIntegrationError, "not found"):
            prepare_oom_recovery(
                "missing-run",
                self.fixture.entrypoint,
                _verification(),
                storage=self.fixture.storage,
                project_root=self.fixture.project_root,
            )

        self.fixture.storage.upsert_run(
            "successful-run",
            project=PROJECT,
            config_json={"batch_size": 32},
            exit_status="success",
        )
        with self.assertRaisesRegex(RecoveryIntegrationError, "no failure capsule"):
            prepare_oom_recovery(
                "successful-run",
                self.fixture.entrypoint,
                _verification(),
                storage=self.fixture.storage,
                project_root=self.fixture.project_root,
            )

        _seed_source(
            self.fixture.storage,
            {"batch_size": 32},
            run_id="ordinary-failure",
            failure_class="python_exception",
        )
        with self.assertRaisesRegex(RecoveryIntegrationError, "not cuda_out_of_memory"):
            prepare_oom_recovery(
                "ordinary-failure",
                self.fixture.entrypoint,
                _verification(),
                storage=self.fixture.storage,
                project_root=self.fixture.project_root,
            )

    def test_preparation_rejects_capsule_without_source_config(self):
        capsule = _capsule({})
        self.fixture.storage.save_failure(
            SOURCE_RUN_ID,
            "RuntimeError",
            capsule["failure"]["message"],
            capsule["failure"]["traceback"],
            capsule["failure"]["classification"],
            capsule["evidence"],
            capsule=capsule,
        )

        with self.assertRaisesRegex(RecoveryIntegrationError, "source configuration"):
            self.fixture.prepare()

    def test_preparation_requires_a_bounded_entrypoint(self):
        self.fixture.write_module("unbounded_training", UNBOUNDED_MODULE)

        with self.assertRaises(EntrypointSignatureError):
            self.fixture.prepare(entrypoint="unbounded_training:train")

    def test_max_proposals_and_approval_filter_are_public_controls(self):
        limited = self.fixture.prepare(max_proposals=1)
        automatic_only = self.fixture.prepare(include_approval_required=False)

        self.assertEqual(len(limited.policy_plan.proposals), 1)
        self.assertEqual(automatic_only.approval_required_proposal_ids, ())
        self.assertTrue(automatic_only.automatic_proposal_ids)

    def test_source_environment_is_captured_from_capsule(self):
        preparation = self.fixture.prepare()

        self.assertEqual(
            preparation.source_environment,
            {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
        )

    def test_proposal_lookup_and_authorization_fail_closed(self):
        preparation = self.fixture.prepare()
        automatic_id = preparation.automatic_proposal_ids[0]

        with self.assertRaisesRegex(RecoveryIntegrationError, "not in this preparation"):
            preparation.proposal("unknown-proposal")
        with self.assertRaisesRegex(RecoveryIntegrationError, "only approval-required"):
            preparation.authorize(
                automatic_id,
                approved_by="engineer@example.com",
                reason="reviewed",
            )


class PublicAPISurfaceTests(unittest.TestCase):
    def test_public_wrapper_has_no_legacy_train_fn_or_llm_parameters(self):
        parameters = inspect.signature(recover_from_oom).parameters

        self.assertIn("entrypoint", parameters)
        self.assertIn("verification", parameters)
        self.assertNotIn("train_fn", parameters)
        self.assertNotIn("model", parameters)
        self.assertNotIn("api_base", parameters)

    def test_preparation_and_execution_are_separate_public_operations(self):
        self.assertIn("failed_run_id", inspect.signature(prepare_oom_recovery).parameters)
        self.assertIn("preparation", inspect.signature(run_prepared_recovery).parameters)


class AuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _IntegrationFixture(
            config={
                "max_seq_length": 2048,
                "oom_sequence_limit": 1024,
                "full_steps": 20,
                "trial_validation_loss": 0.4,
            }
        )
        self.permissions = InterventionPermissions(
            allow_approval_required=True,
            allow_semantic_changes=True,
        )

    def tearDown(self) -> None:
        self.fixture.close()

    @staticmethod
    def _sequence_proposal_id(preparation: RecoveryPreparation) -> str:
        return next(
            proposal.proposal_id
            for proposal in preparation.policy_plan.proposals
            if proposal.policy_rule == "halve_sequence_length"
        )

    def test_unapproved_broader_intervention_is_recorded_as_skipped(self):
        preparation = self.fixture.prepare(permissions=self.permissions)
        result = self.fixture.run(preparation)

        self.assertFalse(result.verified)
        self.assertEqual(result.executed_proposal_ids, ())
        self.assertEqual(
            len(result.skipped_proposals),
            len(preparation.approval_required_proposal_ids),
        )
        self.assertTrue(
            all(item.code == "authorization_missing" for item in result.skipped_proposals)
        )
        self.assertEqual(result.campaign.usage.attempted_trials, 0)

    def test_authorized_sequence_intervention_runs_and_verifies(self):
        preparation = self.fixture.prepare(permissions=self.permissions)
        proposal_id = self._sequence_proposal_id(preparation)
        authorization = preparation.authorize(
            proposal_id,
            approved_by="ml-platform@example.com",
            reason="The sequence-length change was reviewed for this campaign.",
            approved_at=1_800_000_010.0,
        )
        result = self.fixture.run(
            preparation,
            authorizations={proposal_id: authorization},
        )

        self.assertTrue(result.verified)
        self.assertEqual(result.executed_proposal_ids, (proposal_id,))
        self.assertEqual(
            {item.proposal_id for item in result.skipped_proposals},
            set(preparation.approval_required_proposal_ids) - {proposal_id},
        )
        self.assertTrue(
            all(item.code == "authorization_missing" for item in result.skipped_proposals)
        )
        self.assertEqual(result.verified_candidate_id, proposal_id)

    def test_unknown_and_automatic_authorization_ids_are_rejected(self):
        preparation = self.fixture.prepare(permissions=self.permissions)
        proposal_id = preparation.approval_required_proposal_ids[0]
        authorization = preparation.authorize(
            proposal_id,
            approved_by="reviewer",
            reason="reviewed",
        )

        with self.assertRaisesRegex(RecoveryIntegrationError, "mapping key"):
            self.fixture.run(
                preparation,
                authorizations={"unknown-proposal": authorization},
            )

        automatic_fixture = _IntegrationFixture()
        try:
            automatic = automatic_fixture.prepare()
            automatic_id = automatic.automatic_proposal_ids[0]
            fake = InterventionAuthorization.approve(
                automatic.proposal(automatic_id),
                approved_by="reviewer",
                reason="unnecessary authorization used to test rejection",
            )
            with self.assertRaisesRegex(RecoveryIntegrationError, "automatic proposals"):
                automatic_fixture.run(
                    automatic,
                    authorizations={automatic_id: fake},
                )
        finally:
            automatic_fixture.close()

    def test_authorized_proposal_still_cannot_exceed_contract_permissions(self):
        preparation = self.fixture.prepare(
            permissions=InterventionPermissions(
                allow_approval_required=True,
                allow_semantic_changes=False,
            )
        )
        proposal_id = self._sequence_proposal_id(preparation)
        authorization = preparation.authorize(
            proposal_id,
            approved_by="reviewer",
            reason="reviewed",
        )

        with self.assertRaisesRegex(RecoveryIntegrationError, "exceeds contract scope"):
            self.fixture.run(
                preparation,
                authorizations={proposal_id: authorization},
            )


class RealCampaignIntegrationTests(RecoveryIntegrationTestCase):
    def test_real_isolated_campaign_reaches_verified_recovery(self):
        result = self.fixture.run(campaign_id="verified-campaign")

        self.assertTrue(result.verified)
        self.assertEqual(result.campaign.status, "verified")
        self.assertEqual(result.campaign.stopped_reason, "verified_recovery")
        self.assertEqual(result.campaign.usage.probe_trials, 1)
        self.assertEqual(result.campaign.usage.full_trials, 1)
        self.assertEqual(result.campaign.usage.confirmation_trials, 2)
        self.assertEqual(len(result.verified_run_ids), 2)
        self.assertEqual(len(set(result.verified_run_ids)), 2)
        self.assertTrue(all(trial.worker_pid for trial in result.campaign.trials))
        self.assertEqual(
            len({trial.worker_pid for trial in result.campaign.trials}),
            len(result.campaign.trials),
        )

    def test_verified_campaign_is_persisted_with_trial_rows_and_artifact(self):
        result = self.fixture.run(campaign_id="persistence-campaign")
        campaign_row = self.fixture.storage.get_recovery_campaign(result.campaign_id)
        trial_rows = self.fixture.storage.list_recovery_trials(result.campaign_id)
        artifacts = self.fixture.storage.get_artifacts(result.campaign_id)

        self.assertIsNotNone(campaign_row)
        self.assertEqual(campaign_row["stopped_reason"], "verified_recovery")
        self.assertIsNone(campaign_row["best_run_id"])
        self.assertEqual(json.loads(campaign_row["report_json"]), result.to_dict())
        self.assertEqual(len(trial_rows), 4)
        self.assertEqual([row["phase"] for row in trial_rows], [
            "probe", "full", "confirmation", "confirmation"
        ])
        self.assertEqual(sum(int(row["verified"]) for row in trial_rows), 2)
        self.assertEqual(len(artifacts), 1)
        artifact_path = Path(artifacts[0]["path"])
        payload = artifact_path.read_bytes()
        self.assertEqual(json.loads(payload), result.to_dict())
        self.assertEqual(hashlib.sha256(payload).hexdigest(), artifacts[0]["checksum"])
        self.assertEqual(len(payload), artifacts[0]["size_bytes"])

    def test_only_verified_campaign_marks_source_resolved(self):
        verified = self.fixture.run(campaign_id="resolution-campaign")
        row = self.fixture.storage.get_run(SOURCE_RUN_ID)

        self.assertTrue(verified.verified)
        self.assertEqual(row["resolved"], 1)
        self.assertIn(verified.campaign_id, row["resolved_note"])
        self.assertIn(verified.verified_candidate_id, row["resolved_note"])

    def test_always_oom_campaign_is_not_recovered_and_source_stays_unresolved(self):
        self.fixture.write_module("recovery_training", ALWAYS_OOM_MODULE)
        result = self.fixture.run(campaign_id="not-recovered-campaign")
        source = self.fixture.storage.get_run(SOURCE_RUN_ID)

        self.assertFalse(result.verified)
        self.assertEqual(result.campaign.status, "not_recovered")
        self.assertNotEqual(result.campaign.stopped_reason, "verified_recovery")
        self.assertEqual(result.verified_run_ids, ())
        self.assertFalse(bool(source["resolved"]))

    def test_non_integral_full_progress_fails_closed(self):
        self.fixture.write_module("recovery_training", BAD_PROGRESS_MODULE)
        result = self.fixture.run(campaign_id="bad-progress-campaign")

        self.assertFalse(result.verified)
        self.assertEqual(result.campaign.status, "stopped")
        self.assertEqual(result.campaign.stopped_reason, "trial_executor_error")

    def test_custom_progress_metric_is_honored(self):
        self.fixture.write_module("recovery_training", CUSTOM_PROGRESS_MODULE)
        result = self.fixture.run(
            campaign_id="custom-progress-campaign",
            progress_metric="examples_seen",
        )

        self.assertTrue(result.verified)
        self.assertTrue(
            all(
                trial.progress_steps == (3 if trial.phase == "probe" else 20)
                for trial in result.campaign.trials
            )
        )

    def test_source_capsule_change_after_preparation_prevents_compute(self):
        preparation = self.fixture.prepare()
        changed = deepcopy(self.fixture.capsule)
        changed["captured_at"] += 1
        self.fixture.storage.save_failure(
            SOURCE_RUN_ID,
            "RuntimeError",
            changed["failure"]["message"],
            changed["failure"]["traceback"],
            changed["failure"]["classification"],
            changed["evidence"],
            capsule=changed,
        )

        with self.assertRaisesRegex(RecoveryIntegrationError, "changed after"):
            self.fixture.run(preparation, campaign_id="must-not-start")
        self.assertIsNone(self.fixture.storage.get_recovery_campaign("must-not-start"))

    def test_entrypoint_is_revalidated_before_campaign_creation(self):
        preparation = self.fixture.prepare()
        self.fixture.write_module("recovery_training", UNBOUNDED_MODULE)

        with self.assertRaises(EntrypointSignatureError):
            self.fixture.run(preparation, campaign_id="entrypoint-changed")
        self.assertIsNone(self.fixture.storage.get_recovery_campaign("entrypoint-changed"))

    def test_one_call_wrapper_uses_same_public_workflow(self):
        result = recover_from_oom(
            SOURCE_RUN_ID,
            self.fixture.entrypoint,
            _verification(),
            budget=_budget(),
            storage=self.fixture.storage,
            project_root=self.fixture.project_root,
            trials_root=self.fixture.root / "wrapper-trials",
            campaign_id="wrapper-campaign",
            print_summary=False,
        )

        self.assertIsInstance(result, RecoveryResult)
        self.assertTrue(result.verified)
        self.assertEqual(result.campaign_id, "wrapper-campaign")


class RecoveryResultArtifactTests(RecoveryIntegrationTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.result = self.fixture.run(campaign_id="artifact-campaign")
        self.payload = self.result.to_dict()

    def test_result_round_trip_schema_and_digest(self):
        restored = RecoveryResult.from_json(self.result.to_json())

        self.assertEqual(restored, self.result)
        self.assertEqual(recovery_result_digest(restored), recovery_result_digest(self.result))
        self.assertEqual(
            self.payload["schema"],
            {"name": RECOVERY_RESULT_SCHEMA_NAME, "version": RECOVERY_RESULT_SCHEMA_VERSION},
        )
        self.assertTrue(self.payload["verified"])

    def test_result_rejects_tampered_preparation_digest_and_verified_flag(self):
        bad_digest = deepcopy(self.payload)
        bad_digest["preparation_digest"] = "0" * 64
        bad_verified = deepcopy(self.payload)
        bad_verified["verified"] = False

        for payload in (bad_digest, bad_verified):
            with self.subTest(payload=payload):
                with self.assertRaises(RecoveryIntegrationError):
                    RecoveryResult.from_dict(payload)

    def test_result_rejects_unaccounted_or_duplicate_proposals(self):
        unaccounted = deepcopy(self.payload)
        unaccounted["executed_proposal_ids"] = unaccounted["executed_proposal_ids"][:-1]
        duplicate = deepcopy(self.payload)
        duplicate["executed_proposal_ids"] = [
            duplicate["executed_proposal_ids"][0],
            duplicate["executed_proposal_ids"][0],
        ]

        for payload in (unaccounted, duplicate):
            with self.subTest(payload=payload):
                with self.assertRaises(RecoveryIntegrationError):
                    RecoveryResult.from_dict(payload)

    def test_summary_reports_verifier_backed_claims_and_never_best_run(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_recovery_summary(self.result)
        rendered = output.getvalue()

        self.assertIn("Verified recovery:", rendered)
        self.assertIn("Independent confirmation runs:", rendered)
        self.assertNotIn("best run", rendered.lower())
        self.assertNotIn("score=", rendered.lower())

    def test_result_invariants_are_not_optional_marketing_flags(self):
        payload = deepcopy(self.payload)
        payload["invariants"]["verifier_is_only_recovery_authority"] = False

        with self.assertRaisesRegex(RecoveryIntegrationError, "invariants"):
            RecoveryResult.from_dict(payload)


if __name__ == "__main__":
    unittest.main()