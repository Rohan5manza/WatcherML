"""Acceptance tests for WatcherML v1 recovery persistence.

This suite treats SQLite as an audit boundary.  A completed full trial is not
stored as a verified recovery, immutable campaign identities cannot be reused,
and the normalized query columns must agree with the complete JSON artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from watcherml.storage import (
    RECOVERY_RESULT_FILENAME,
    STORAGE_SCHEMA_VERSION,
    Storage,
    StorageConflictError,
    StorageError,
)


PROJECT = "storage-integration"
SOURCE_RUN = "source-oom"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract() -> dict:
    return {
        "schema": {"name": "watcherml.recovery-contract", "version": "1.0"},
        "project": PROJECT,
        "source_run_id": SOURCE_RUN,
        "entrypoint": {
            "schema": {"name": "watcherml.training-entrypoint", "version": "1.0"},
            "kind": "python_callable",
            "target": "train:main",
            "working_directory": ".",
        },
        "source_config": {"batch_size": 32, "gradient_accumulation_steps": 1},
        "budget": {"max_trials": 3},
        "verification": {"confirmation_runs": 1},
    }


def _proposal(proposal_id: str, rule: str) -> dict:
    return {
        "schema": {"name": "watcherml.intervention-proposal", "version": "1.0"},
        "proposal_id": proposal_id,
        "policy_rule": rule,
        "proposer": "deterministic_policy",
        "changes": [],
        "rationale": "Evidence-backed bounded change.",
        "expected_effect": "Reduce activation memory.",
        "evidence_refs": ["EV-1"],
    }


def _trial(
    phase: str,
    ordinal: int,
    *,
    candidate_id: str = "proposal-auto",
    status: str = "success",
) -> dict:
    return {
        "schema": {"name": "watcherml.campaign-trial", "version": "1.0"},
        "candidate_id": candidate_id,
        "trial_id": "trial-{}-{}".format(phase, ordinal),
        "run_id": "run-{}-{}".format(phase, ordinal),
        "phase": phase,
        "status": status,
        "request_digest": _digest("request-{}-{}".format(phase, ordinal)),
        "execution_manifest_digest": _digest(
            "execution-{}-{}".format(phase, ordinal)
        ),
        "duration_seconds": 1.25,
        "gpu_seconds": 1.0,
        "progress_steps": 3 if phase == "probe" else 20,
        "peak_vram_bytes": 4 * 1024**3,
        "workload_identity": {
            "dataset_fingerprint": "dataset-example",
            "environment_fingerprint": "environment-example",
            "git_commit": "abc123",
            "model_identifier": "example/model",
        },
        "worker_pid": 10_000 + ordinal,
        "failure_class": None,
        "metrics": {"validation_loss": 0.4, "steps_completed": 20.0},
    }


def _verification(*, verified: bool = True) -> dict:
    return {
        "schema": {"name": "watcherml.recovery-verification", "version": "1.0"},
        "campaign_id": "campaign-verified",
        "candidate_id": "proposal-auto",
        "contract_digest": _digest("contract"),
        "verified": verified,
        "confirmation_run_ids": ["run-confirmation-1"],
        "checks": [],
        "failure_reasons": [] if verified else ["metric_guard_failed"],
    }


def _report(
    *,
    campaign_id: str = "campaign-verified",
    verified: bool = True,
    include_approval_proposal: bool = True,
) -> dict:
    proposals = [
        _proposal("proposal-auto", "halve_batch_preserve_effective_batch")
    ]
    approval_ids = []
    skipped = []
    if include_approval_proposal:
        proposals.append(
            _proposal("proposal-approval", "halve_sequence_length")
        )
        approval_ids.append("proposal-approval")
        skipped.append(
            {
                "proposal_id": "proposal-approval",
                "policy_rule": "halve_sequence_length",
                "code": "authorization_missing",
                "reason": "Explicit authorization was not supplied.",
            }
        )
    trials = [
        _trial("probe", 1),
        _trial("full", 1),
        _trial("confirmation", 1),
    ]
    status = "verified" if verified else "not_recovered"
    reason = "verified_recovery" if verified else "all_ranked_candidates_rejected"
    verifications = [_verification(verified=verified)]
    verifications[0]["campaign_id"] = campaign_id
    return {
        "schema": {"name": "watcherml.recovery-result", "version": "1.0"},
        "preparation": {
            "policy_plan": {
                "proposals": proposals,
                "automatic_proposal_ids": ["proposal-auto"],
                "approval_required_proposal_ids": approval_ids,
            }
        },
        "preparation_digest": _digest("preparation"),
        "campaign": {
            "campaign_id": campaign_id,
            "contract_digest": _digest("contract"),
            "status": status,
            "stopped_reason": reason,
            "planned_candidate_ids": ["proposal-auto"],
            "probe_survivor_ids": ["proposal-auto"],
            "trials": trials,
            "ranking": {
                "confirmation_order": ["proposal-auto"],
                "eligible_candidate_ids": ["proposal-auto"],
            },
            "verifications": verifications,
            "verified": verified,
            "verified_candidate_id": "proposal-auto" if verified else None,
            "verified_run_ids": ["run-confirmation-1"] if verified else [],
            "usage": {
                "attempted_trials": 3,
                "probe_trials": 1,
                "full_trials": 1,
                "confirmation_trials": 1,
                "elapsed_seconds": 3.75,
                "observed_gpu_seconds": 3.0,
                "gpu_measurement_complete": True,
            },
        },
        "executed_proposal_ids": ["proposal-auto"],
        "skipped_proposals": skipped,
        "verified": verified,
        "invariants": {
            "subprocess_trials_only": True,
            "ranking_is_not_a_verdict": True,
            "verifier_is_only_recovery_authority": True,
            "no_unrecorded_retry": True,
        },
    }


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / ".watcherml"
        self.storage = Storage(self.root)
        self.storage.upsert_run(
            SOURCE_RUN,
            project=PROJECT,
            config_json={"batch_size": 32},
            started_at=1.0,
            exit_status="failed",
        )

    def tearDown(self) -> None:
        self.storage.close()
        self.temporary.cleanup()

    def create_campaign(
        self,
        campaign_id: str = "campaign-verified",
        *,
        project: str = PROJECT,
        started_at: float = 10.0,
    ) -> None:
        self.storage.create_recovery_campaign(
            campaign_id,
            project,
            SOURCE_RUN,
            _contract(),
            started_at,
        )

    def seed_trial_rows(
        self,
        campaign_id: str = "campaign-verified",
        report: dict | None = None,
    ) -> None:
        report = report or _report(campaign_id=campaign_id)
        for index, trial in enumerate(report["campaign"]["trials"]):
            self.storage.save_recovery_trial(
                campaign_id,
                trial["run_id"],
                trial["phase"],
                {
                    "proposal_id": trial["candidate_id"],
                    "policy_rule": "halve_batch_preserve_effective_batch",
                },
                {
                    "batch_size": 16,
                    "gradient_accumulation_steps": 2,
                    "__environment__": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                },
                "Reduce activation memory.",
                None,
                trial["status"],
                None,
                trial["run_id"] in report["campaign"]["verified_run_ids"],
                11.0 + index,
            )


class SchemaAndLifecycleTests(StorageTestCase):
    def test_schema_version_and_directories_are_initialized(self):
        self.assertEqual(self.storage.schema_version, STORAGE_SCHEMA_VERSION)
        self.assertTrue(Path(self.storage.db_path).is_file())
        self.assertTrue((self.root / "artifacts").is_dir())

    def test_connection_uses_wal_mode(self):
        mode = self.storage._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_context_manager_closes_connection(self):
        root = Path(self.temporary.name) / "context-storage"
        with Storage(root) as storage:
            self.assertEqual(storage.schema_version, STORAGE_SCHEMA_VERSION)
        with self.assertRaises(sqlite3.ProgrammingError):
            storage._conn.execute("SELECT 1")

    def test_close_is_idempotent(self):
        extra = Storage(Path(self.temporary.name) / "extra")
        extra.close()
        extra.close()

    def test_schema_initialization_is_idempotent(self):
        self.storage.close()
        self.storage = Storage(self.root)
        self.assertEqual(self.storage.schema_version, STORAGE_SCHEMA_VERSION)
        self.assertIsNotNone(self.storage.get_run(SOURCE_RUN))


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / ".watcherml"
        self.root.mkdir()
        self.db_path = self.root / "watcher.db"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _legacy_database(self, *, duplicate_trials: bool = False) -> None:
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            """CREATE TABLE recovery_campaigns(
                campaign_id TEXT PRIMARY KEY, project TEXT, source_run_id TEXT,
                contract_json TEXT, started_at REAL, ended_at REAL,
                stopped_reason TEXT, best_run_id TEXT, report_json TEXT
            )"""
        )
        connection.execute(
            """CREATE TABLE recovery_trials(
                campaign_id TEXT, run_id TEXT, phase TEXT, hypothesis_json TEXT,
                patch_json TEXT, rationale TEXT, confidence REAL, outcome TEXT,
                score REAL, verified INTEGER, created_at REAL
            )"""
        )
        connection.execute(
            "INSERT INTO recovery_campaigns VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "legacy-campaign",
                "legacy-project",
                "legacy-source",
                "{}",
                1.0,
                2.0,
                "legacy-finished",
                "legacy-best",
                "{}",
            ),
        )
        rows = 2 if duplicate_trials else 1
        for index in range(rows):
            connection.execute(
                "INSERT INTO recovery_trials VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "legacy-campaign",
                    "legacy-run",
                    "full",
                    None,
                    "{}",
                    "legacy",
                    None,
                    "success",
                    None,
                    0,
                    1.0 + index,
                ),
            )
        connection.commit()
        connection.close()

    def test_legacy_database_migrates_without_losing_rows(self):
        self._legacy_database()
        storage = Storage(self.root)
        try:
            campaign = storage.get_recovery_campaign("legacy-campaign")
            trials = storage.list_recovery_trials("legacy-campaign")
            self.assertEqual(campaign["best_run_id"], "legacy-best")
            self.assertEqual(campaign["status"], "running")
            self.assertEqual(len(trials), 1)
            self.assertIn("request_digest", trials[0].keys())
            self.assertEqual(storage.schema_version, STORAGE_SCHEMA_VERSION)
        finally:
            storage.close()

    def test_dirty_legacy_duplicates_do_not_block_migration(self):
        self._legacy_database(duplicate_trials=True)
        storage = Storage(self.root)
        try:
            self.assertEqual(len(storage.list_recovery_trials("legacy-campaign")), 2)
            self.assertEqual(storage.schema_version, STORAGE_SCHEMA_VERSION)
        finally:
            storage.close()


class CorePersistenceCompatibilityTests(StorageTestCase):
    def test_run_json_fields_round_trip(self):
        row = self.storage.get_run(SOURCE_RUN)
        self.assertEqual(json.loads(row["config_json"]), {"batch_size": 32})

    def test_unknown_run_fields_are_rejected(self):
        with self.assertRaisesRegex(StorageError, "unknown run fields"):
            self.storage.upsert_run("run", imaginary_field=True)

    def test_nonfinite_json_is_rejected(self):
        with self.assertRaisesRegex(StorageError, "strict JSON"):
            self.storage.upsert_run("run", config_json={"loss": math.nan})

    def test_final_metrics_return_latest_value_per_name(self):
        self.storage.log_metric(SOURCE_RUN, "loss", 1.0, 1, 1.0)
        self.storage.log_metric(SOURCE_RUN, "loss", 0.5, 2, 2.0)
        self.storage.log_metric(SOURCE_RUN, "accuracy", 0.9, 2, 2.0)
        self.assertEqual(
            self.storage.final_metrics(SOURCE_RUN),
            {"loss": 0.5, "accuracy": 0.9},
        )

    def test_complete_failure_capsule_round_trips(self):
        capsule = {
            "schema": {"name": "watcherml.failure-capsule", "version": "1.0"},
            "run_id": SOURCE_RUN,
            "failure_class": "cuda_out_of_memory",
            "captured_at": 3.0,
            "evidence": {"config": {"batch_size": 32}},
        }
        self.storage.save_failure(
            SOURCE_RUN,
            "RuntimeError",
            "CUDA out of memory",
            "traceback",
            {"rule": "cuda_out_of_memory"},
            capsule["evidence"],
            capsule=capsule,
        )
        self.assertEqual(self.storage.get_failure_capsule(SOURCE_RUN), capsule)
        self.assertEqual(
            self.storage.get_failure(SOURCE_RUN)["capsule_schema_version"], "1.0"
        )

    def test_legacy_failure_row_has_marked_compatibility_view(self):
        self.storage.save_failure(
            SOURCE_RUN,
            "RuntimeError",
            "boom",
            "traceback",
            {"rule": "python_exception"},
            {"config": {"x": 1}},
        )
        capsule = self.storage.get_failure_capsule(SOURCE_RUN)
        self.assertEqual(capsule["capsule_schema_version"], "legacy")
        self.assertEqual(capsule["failure_class"], "python_exception")

    def test_resolving_unknown_run_fails(self):
        with self.assertRaisesRegex(StorageError, "not found"):
            self.storage.set_run_resolved("missing-run", True, "verified")


class CampaignPersistenceTests(StorageTestCase):
    def test_campaign_starts_unverified_and_indexes_contract(self):
        self.create_campaign()
        row = self.storage.get_recovery_campaign("campaign-verified")
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["verified"], 0)
        self.assertIsNone(row["best_run_id"])
        self.assertEqual(len(row["contract_digest"]), 64)
        self.assertEqual(json.loads(row["contract_json"]), _contract())

    def test_campaign_id_cannot_be_reused(self):
        self.create_campaign()
        with self.assertRaisesRegex(StorageConflictError, "already exists"):
            self.create_campaign()

    def test_finishing_unknown_campaign_fails(self):
        with self.assertRaisesRegex(StorageError, "not found"):
            self.storage.finish_recovery_campaign(
                "missing", 20.0, "integration_error", None, {"status": "error"}
            )

    def test_integration_error_is_machine_queryable(self):
        self.create_campaign("campaign-error")
        report = {
            "schema": {"name": "watcherml.recovery-result", "version": "1.0"},
            "campaign_id": "campaign-error",
            "status": "integration_error",
            "verified": False,
            "error": {"type": "RuntimeError", "message": "worker failed"},
        }
        self.storage.finish_recovery_campaign(
            "campaign-error", 20.0, "integration_error", None, report
        )
        row = self.storage.get_recovery_campaign("campaign-error")
        self.assertEqual(row["status"], "integration_error")
        self.assertEqual(row["stopped_reason"], "integration_error")
        self.assertEqual(row["verified"], 0)

    def test_finished_campaign_is_immutable_but_exact_finish_is_idempotent(self):
        self.create_campaign("campaign-error")
        report = {"status": "integration_error", "verified": False}
        self.storage.finish_recovery_campaign(
            "campaign-error", 20.0, "integration_error", None, report
        )
        self.storage.finish_recovery_campaign(
            "campaign-error", 21.0, "integration_error", None, report
        )
        with self.assertRaisesRegex(StorageConflictError, "immutable"):
            self.storage.finish_recovery_campaign(
                "campaign-error",
                22.0,
                "another_error",
                None,
                {"status": "another_error", "verified": False},
            )

    def test_authoritative_verified_report_populates_query_columns(self):
        self.create_campaign()
        report = _report()
        self.seed_trial_rows(report=report)
        self.storage.finish_recovery_campaign(
            "campaign-verified", 20.0, "verified_recovery", None, report
        )
        row = self.storage.get_recovery_campaign("campaign-verified")
        self.assertEqual(row["status"], "verified")
        self.assertEqual(row["verified"], 1)
        self.assertEqual(row["verified_candidate_id"], "proposal-auto")
        self.assertEqual(json.loads(row["verified_run_ids_json"]), ["run-confirmation-1"])
        self.assertEqual(json.loads(row["executed_proposal_ids_json"]), ["proposal-auto"])
        self.assertEqual(json.loads(row["usage_json"])["attempted_trials"], 3)
        self.assertIsNone(row["best_run_id"])
        self.assertEqual(len(row["report_digest"]), 64)

    def test_complete_report_can_be_loaded_without_reconstruction(self):
        self.create_campaign()
        report = _report()
        self.seed_trial_rows(report=report)
        self.storage.finish_recovery_campaign(
            "campaign-verified", 20.0, "verified_recovery", None, report
        )
        self.assertEqual(
            self.storage.get_recovery_campaign_report("campaign-verified"), report
        )
        self.assertIsNone(self.storage.get_recovery_campaign_report("missing"))

    def test_contradictory_verified_flags_are_rejected(self):
        self.create_campaign()
        report = _report()
        report["verified"] = False
        with self.assertRaisesRegex(StorageError, "inconsistent"):
            self.storage.finish_recovery_campaign(
                "campaign-verified", 20.0, "verified_recovery", None, report
            )

    def test_unverified_report_cannot_claim_verified_reason(self):
        self.create_campaign()
        report = _report(verified=False)
        report["campaign"]["stopped_reason"] = "verified_recovery"
        with self.assertRaisesRegex(StorageError, "claims"):
            self.storage.finish_recovery_campaign(
                "campaign-verified", 20.0, "verified_recovery", None, report
            )

    def test_legacy_best_run_is_preserved_only_for_legacy_report(self):
        self.create_campaign("legacy-finish")
        self.storage.finish_recovery_campaign(
            "legacy-finish",
            20.0,
            "legacy-finished",
            "legacy-best-run",
            {"status": "legacy-finished", "verified": False},
        )
        row = self.storage.get_recovery_campaign("legacy-finish")
        self.assertEqual(row["best_run_id"], "legacy-best-run")

    def test_campaign_listing_filters_project_status_and_verification(self):
        self.create_campaign("campaign-running", started_at=10.0)
        self.create_campaign("campaign-verified", started_at=11.0)
        report = _report(campaign_id="campaign-verified")
        self.seed_trial_rows("campaign-verified", report)
        self.storage.finish_recovery_campaign(
            "campaign-verified", 20.0, "verified_recovery", None, report
        )
        self.assertEqual(
            [row["campaign_id"] for row in self.storage.list_recovery_campaigns()],
            ["campaign-verified", "campaign-running"],
        )
        self.assertEqual(
            [row["campaign_id"] for row in self.storage.list_recovery_campaigns(
                PROJECT, status="verified", verified=True
            )],
            ["campaign-verified"],
        )


class TrialPersistenceTests(StorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.create_campaign()

    def save_trial(self, run_id: str = "trial-run", **details) -> None:
        arguments = {
            "campaign_id": "campaign-verified",
            "run_id": run_id,
            "phase": "probe",
            "hypothesis": {
                "proposal_id": "proposal-auto",
                "policy_rule": "halve_batch_preserve_effective_batch",
            },
            "patch": {
                "batch_size": 16,
                "__environment__": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            },
            "rationale": "bounded",
            "confidence": None,
            "outcome": "success",
            "score": None,
            "verified": False,
            "created_at": 11.0,
            "trial_id": "trial-id",
            "request_digest": _digest("request"),
            "execution_manifest_digest": _digest("execution"),
        }
        arguments.update(details)
        self.storage.save_recovery_trial(**arguments)

    def test_trial_requires_existing_campaign(self):
        with self.assertRaisesRegex(StorageError, "not found"):
            self.storage.save_recovery_trial(
                "missing", "run", "probe", None, {}, "", None,
                "success", None, False, 1.0
            )

    def test_trial_extracts_proposal_and_environment_metadata(self):
        self.save_trial()
        row = self.storage.get_recovery_trial("campaign-verified", "trial-run")
        self.assertEqual(row["candidate_id"], "proposal-auto")
        self.assertEqual(row["proposal_id"], "proposal-auto")
        self.assertEqual(row["policy_rule"], "halve_batch_preserve_effective_batch")
        self.assertEqual(json.loads(row["patch_json"]), {"batch_size": 16})
        self.assertEqual(
            json.loads(row["environment_patch_json"]),
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    def test_run_id_is_globally_unique_across_campaigns(self):
        self.save_trial()
        self.create_campaign("second-campaign")
        with self.assertRaisesRegex(StorageConflictError, "run_id"):
            self.storage.save_recovery_trial(
                "second-campaign", "trial-run", "probe", None, {}, "",
                None, "success", None, False, 12.0
            )

    def test_trial_request_and_execution_identities_are_unique_per_campaign(self):
        self.save_trial()
        variants = (
            {"trial_id": "trial-id", "request_digest": _digest("request-2"),
             "execution_manifest_digest": _digest("execution-2")},
            {"trial_id": "trial-id-2", "request_digest": _digest("request"),
             "execution_manifest_digest": _digest("execution-2")},
            {"trial_id": "trial-id-2", "request_digest": _digest("request-2"),
             "execution_manifest_digest": _digest("execution")},
        )
        for index, details in enumerate(variants):
            with self.subTest(details=details):
                with self.assertRaisesRegex(StorageConflictError, "unique"):
                    self.save_trial("trial-run-{}".format(index), **details)

    def test_trial_listing_uses_creation_order(self):
        self.save_trial("later", created_at=20.0)
        self.save_trial(
            "earlier",
            created_at=10.0,
            trial_id="trial-earlier",
            request_digest=_digest("request-earlier"),
            execution_manifest_digest=_digest("execution-earlier"),
        )
        self.assertEqual(
            [row["run_id"] for row in self.storage.list_recovery_trials(
                "campaign-verified"
            )],
            ["earlier", "later"],
        )

    def test_campaign_report_enriches_preliminary_trial_rows(self):
        report = _report()
        self.seed_trial_rows(report=report)
        self.storage.finish_recovery_campaign(
            "campaign-verified", 20.0, "verified_recovery", None, report
        )
        full = self.storage.get_recovery_trial("campaign-verified", "run-full-1")
        confirmation = self.storage.get_recovery_trial(
            "campaign-verified", "run-confirmation-1"
        )
        self.assertEqual(full["trial_id"], "trial-full-1")
        self.assertEqual(full["request_digest"], _digest("request-full-1"))
        self.assertEqual(full["worker_pid"], 10_001)
        self.assertEqual(json.loads(full["metrics_json"])["validation_loss"], 0.4)
        self.assertEqual(full["verified"], 0)
        self.assertEqual(confirmation["verified"], 1)

    def test_report_can_insert_missing_preliminary_trial_rows(self):
        report = _report()
        self.storage.finish_recovery_campaign(
            "campaign-verified", 20.0, "verified_recovery", None, report
        )
        self.assertEqual(len(self.storage.list_recovery_trials("campaign-verified")), 3)
        row = self.storage.get_recovery_trial("campaign-verified", "run-probe-1")
        self.assertEqual(row["candidate_id"], "proposal-auto")

    def test_report_rejects_duplicate_trial_identities(self):
        fields = ("trial_id", "run_id", "request_digest", "execution_manifest_digest")
        for field in fields:
            report = _report()
            report["campaign"]["trials"][1][field] = report["campaign"]["trials"][0][field]
            with self.subTest(field=field):
                storage = Storage(Path(self.temporary.name) / ("duplicate-" + field))
                try:
                    storage.upsert_run(SOURCE_RUN, project=PROJECT)
                    storage.create_recovery_campaign(
                        "campaign-verified", PROJECT, SOURCE_RUN, _contract(), 10.0
                    )
                    with self.assertRaises(StorageConflictError):
                        storage.finish_recovery_campaign(
                            "campaign-verified", 20.0, "verified_recovery", None, report
                        )
                finally:
                    storage.close()


class ProposalVerificationAndArtifactTests(StorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.create_campaign()

    def finish(self, report: dict | None = None) -> dict:
        report = report or _report()
        self.seed_trial_rows(report=report)
        self.storage.finish_recovery_campaign(
            "campaign-verified", 20.0,
            report["campaign"]["stopped_reason"], None, report
        )
        return report

    def test_executed_and_skipped_proposals_are_normalized(self):
        self.finish()
        rows = self.storage.list_recovery_proposals("campaign-verified")
        self.assertEqual([row["proposal_id"] for row in rows], [
            "proposal-auto", "proposal-approval"
        ])
        self.assertEqual(rows[0]["authorization_mode"], "automatic")
        self.assertEqual(rows[0]["state"], "executed")
        self.assertEqual(rows[1]["authorization_mode"], "approval_required")
        self.assertEqual(rows[1]["state"], "skipped")
        self.assertEqual(rows[1]["skip_code"], "authorization_missing")

    def test_unaccounted_proposal_is_rejected(self):
        report = _report()
        report["skipped_proposals"] = []
        self.seed_trial_rows(report=report)
        with self.assertRaisesRegex(StorageError, "skip record"):
            self.storage.finish_recovery_campaign(
                "campaign-verified", 20.0, "verified_recovery", None, report
            )

    def test_proposal_without_authorization_partition_is_rejected(self):
        report = _report()
        report["preparation"]["policy_plan"]["approval_required_proposal_ids"] = []
        self.seed_trial_rows(report=report)
        with self.assertRaisesRegex(StorageError, "authorization mode"):
            self.storage.finish_recovery_campaign(
                "campaign-verified", 20.0, "verified_recovery", None, report
            )

    def test_verification_reports_are_normalized(self):
        self.finish()
        rows = self.storage.list_recovery_verifications("campaign-verified")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_id"], "proposal-auto")
        self.assertEqual(rows[0]["verified"], 1)
        self.assertEqual(
            json.loads(rows[0]["confirmation_run_ids_json"]),
            ["run-confirmation-1"],
        )

    def test_recovery_result_artifact_updates_campaign_index(self):
        report = self.finish()
        path = self.storage.artifact_path(
            "campaign-verified", RECOVERY_RESULT_FILENAME
        )
        payload = json.dumps(report, sort_keys=True).encode("utf-8")
        Path(path).write_bytes(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        self.storage.log_artifact(
            "campaign-verified", path, checksum, len(payload)
        )
        row = self.storage.get_recovery_campaign("campaign-verified")
        self.assertEqual(row["artifact_path"], path)
        self.assertEqual(row["artifact_checksum"], checksum)
        self.assertEqual(row["artifact_size_bytes"], len(payload))

    def test_ordinary_artifact_does_not_modify_campaign_artifact_index(self):
        self.finish()
        path = self.storage.artifact_path("campaign-verified", "stdout.log")
        Path(path).write_text("hello", encoding="utf-8")
        self.storage.log_artifact("campaign-verified", path, _digest("hello"), 5)
        row = self.storage.get_recovery_campaign("campaign-verified")
        self.assertIsNone(row["artifact_path"])


class ResolutionMemoryTests(StorageTestCase):
    def _campaign(self, campaign_id: str, *, verified: bool) -> None:
        self.create_campaign(campaign_id, started_at=10.0 if verified else 9.0)
        report = _report(campaign_id=campaign_id, verified=verified)
        run_id_map = {}
        for trial in report["campaign"]["trials"]:
            old_run_id = trial["run_id"]
            suffix = campaign_id.removeprefix("campaign-")
            trial["run_id"] = "{}-{}".format(old_run_id, suffix)
            trial["trial_id"] = "{}-{}".format(trial["trial_id"], suffix)
            trial["request_digest"] = _digest(
                "{}:{}".format(trial["request_digest"], campaign_id)
            )
            trial["execution_manifest_digest"] = _digest(
                "{}:{}".format(trial["execution_manifest_digest"], campaign_id)
            )
            run_id_map[old_run_id] = trial["run_id"]
        if verified:
            report["campaign"]["verified_run_ids"] = [
                run_id_map[run_id]
                for run_id in report["campaign"]["verified_run_ids"]
            ]
        for verification in report["campaign"]["verifications"]:
            verification["confirmation_run_ids"] = [
                run_id_map[run_id]
                for run_id in verification["confirmation_run_ids"]
            ]
        self.seed_trial_rows(campaign_id, report)
        self.storage.finish_recovery_campaign(
            campaign_id,
            20.0,
            report["campaign"]["stopped_reason"],
            None,
            report,
        )

    def setUp(self) -> None:
        super().setUp()
        self.storage.save_failure(
            SOURCE_RUN,
            "RuntimeError",
            "CUDA out of memory",
            "traceback",
            {"rule": "cuda_out_of_memory"},
            {"config": {"batch_size": 32}},
        )

    def test_full_success_and_verified_recovery_are_separate_counts(self):
        self._campaign("campaign-verified", verified=True)
        self._campaign("campaign-unverified", verified=False)
        memory = self.storage.resolution_memory(PROJECT)
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0]["attempts"], 2)
        self.assertEqual(memory[0]["successes"], 2)
        self.assertEqual(memory[0]["verified_recoveries"], 1)
        self.assertEqual(memory[0]["success_rate"], 1.0)
        self.assertEqual(memory[0]["verification_rate"], 0.5)

    def test_resolution_memory_project_filter_is_exact(self):
        self._campaign("campaign-verified", verified=True)
        self.assertEqual(self.storage.resolution_memory("other-project"), [])


if __name__ == "__main__":
    unittest.main()