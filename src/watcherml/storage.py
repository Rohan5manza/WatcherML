"""Local SQLite persistence for WatcherML v1.

The database is deliberately local-first and dependency-free.  Recovery
campaigns retain their complete immutable JSON artifacts while also indexing
the fields needed by the CLI and web API.  Existing pre-v1 databases are
migrated in place; legacy columns remain readable but are not recovery truth.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Mapping, Optional


STORAGE_SCHEMA_VERSION = "2.0"
RECOVERY_RESULT_FILENAME = "recovery-result.json"
DEFAULT_DIR = os.path.join(os.getcwd(), ".watcherml")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StorageError(RuntimeError):
    """Base error for persistence failures and invalid stored artifacts."""


class StorageConflictError(StorageError):
    """Raised when an immutable identity would be reused or overwritten."""


class Storage:
    def __init__(self, root: str = DEFAULT_DIR):
        self.root = os.path.abspath(os.fspath(root))
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "artifacts"), exist_ok=True)
        self.db_path = os.path.join(self.root, "watcher.db")
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        self._configure_connection()
        self._init_schema()

    def _configure_connection(self) -> None:
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    @property
    def schema_version(self) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT version FROM storage_metadata WHERE component='storage'"
            ).fetchone()
            return row["version"] if row else "legacy"

    def _migrate_add_column(self, table: str, column: str, declaration: str) -> None:
        for value in (table, column):
            if not _IDENTIFIER.fullmatch(value):
                raise StorageError("unsafe schema identifier")
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info({})".format(table))
        }
        if column not in columns:
            self._conn.execute(
                "ALTER TABLE {} ADD COLUMN {} {}".format(
                    table, column, declaration
                )
            )

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn
            c.execute(
                """CREATE TABLE IF NOT EXISTS storage_metadata (
                    component TEXT PRIMARY KEY,
                    version TEXT NOT NULL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    project TEXT,
                    config_json TEXT,
                    started_at REAL,
                    ended_at REAL,
                    duration_seconds REAL,
                    exit_status TEXT,
                    git_json TEXT,
                    env_json TEXT,
                    gpu_json TEXT,
                    resource_json TEXT,
                    dataset_fingerprint TEXT,
                    reproduction_score REAL,
                    warnings_json TEXT,
                    capsule_schema_version TEXT,
                    capture_completeness REAL
                )"""
            )
            for name, declaration in (
                ("project", "TEXT"),
                ("config_json", "TEXT"),
                ("started_at", "REAL"),
                ("ended_at", "REAL"),
                ("duration_seconds", "REAL"),
                ("exit_status", "TEXT"),
                ("git_json", "TEXT"),
                ("env_json", "TEXT"),
                ("gpu_json", "TEXT"),
                ("resource_json", "TEXT"),
                ("dataset_fingerprint", "TEXT"),
                ("reproduction_score", "REAL"),
                ("warnings_json", "TEXT"),
                ("capsule_schema_version", "TEXT"),
                ("capture_completeness", "REAL"),
                ("display_name", "TEXT"),
                ("tags_json", "TEXT"),
                ("resolved", "INTEGER NOT NULL DEFAULT 0"),
                ("resolved_note", "TEXT"),
            ):
                self._migrate_add_column("runs", name, declaration)

            c.execute(
                """CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT,
                    name TEXT,
                    value REAL,
                    step INTEGER,
                    timestamp REAL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT,
                    path TEXT,
                    checksum TEXT,
                    size_bytes INTEGER
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS failures (
                    run_id TEXT PRIMARY KEY,
                    exception_type TEXT,
                    message TEXT,
                    traceback TEXT,
                    diagnosis_json TEXT,
                    evidence_json TEXT,
                    capsule_schema_version TEXT,
                    failure_class TEXT,
                    captured_at REAL,
                    capsule_json TEXT
                )"""
            )
            for name, declaration in (
                ("capsule_schema_version", "TEXT"),
                ("failure_class", "TEXT"),
                ("captured_at", "REAL"),
                ("capsule_json", "TEXT"),
            ):
                self._migrate_add_column("failures", name, declaration)

            c.execute(
                """CREATE TABLE IF NOT EXISTS resource_samples (
                    run_id TEXT,
                    t REAL,
                    cpu_pct REAL,
                    ram_pct REAL,
                    gpu_util_pct REAL,
                    gpu_mem_used_mib REAL
                )"""
            )
            for name in (
                "disk_read_mbps",
                "disk_write_mbps",
                "net_sent_mbps",
                "net_recv_mbps",
            ):
                self._migrate_add_column("resource_samples", name, "REAL")

            # The first nine fields match the pre-v1 table exactly.
            c.execute(
                """CREATE TABLE IF NOT EXISTS recovery_campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    project TEXT,
                    source_run_id TEXT,
                    contract_json TEXT,
                    started_at REAL,
                    ended_at REAL,
                    stopped_reason TEXT,
                    best_run_id TEXT,
                    report_json TEXT
                )"""
            )
            campaign_columns = (
                ("schema_version", "TEXT NOT NULL DEFAULT '1.0'"),
                ("status", "TEXT NOT NULL DEFAULT 'running'"),
                ("contract_digest", "TEXT"),
                ("preparation_digest", "TEXT"),
                ("report_digest", "TEXT"),
                ("verified", "INTEGER NOT NULL DEFAULT 0"),
                ("verified_candidate_id", "TEXT"),
                ("verified_run_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("planned_candidate_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("probe_survivor_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("executed_proposal_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("skipped_proposals_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("usage_json", "TEXT"),
                ("ranking_json", "TEXT"),
                ("verification_reports_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("artifact_path", "TEXT"),
                ("artifact_checksum", "TEXT"),
                ("artifact_size_bytes", "INTEGER"),
            )
            for name, declaration in campaign_columns:
                self._migrate_add_column("recovery_campaigns", name, declaration)

            c.execute(
                """CREATE TABLE IF NOT EXISTS recovery_trials (
                    campaign_id TEXT,
                    run_id TEXT,
                    phase TEXT,
                    hypothesis_json TEXT,
                    patch_json TEXT,
                    rationale TEXT,
                    confidence REAL,
                    outcome TEXT,
                    score REAL,
                    verified INTEGER,
                    created_at REAL
                )"""
            )
            trial_columns = (
                ("trial_id", "TEXT"),
                ("candidate_id", "TEXT"),
                ("proposal_id", "TEXT"),
                ("policy_rule", "TEXT"),
                ("status", "TEXT"),
                ("failure_class", "TEXT"),
                ("request_digest", "TEXT"),
                ("execution_manifest_digest", "TEXT"),
                ("worker_pid", "INTEGER"),
                ("duration_seconds", "REAL"),
                ("gpu_seconds", "REAL"),
                ("progress_steps", "INTEGER"),
                ("peak_vram_bytes", "INTEGER"),
                ("metrics_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("workload_identity_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("environment_patch_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("trial_json", "TEXT"),
            )
            for name, declaration in trial_columns:
                self._migrate_add_column("recovery_trials", name, declaration)

            c.execute(
                """CREATE TABLE IF NOT EXISTS recovery_proposals (
                    campaign_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    policy_rule TEXT NOT NULL,
                    authorization_mode TEXT NOT NULL,
                    state TEXT NOT NULL,
                    skip_code TEXT,
                    skip_reason TEXT,
                    rationale TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    PRIMARY KEY (campaign_id, proposal_id)
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS recovery_verifications (
                    campaign_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    confirmation_run_ids_json TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (campaign_id, candidate_id)
                )"""
            )

            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_project_started "
                "ON runs(project, started_at DESC)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_run_timestamp "
                "ON metrics(run_id, timestamp)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_campaign_project_started "
                "ON recovery_campaigns(project, started_at DESC)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_campaign_source "
                "ON recovery_campaigns(source_run_id, started_at DESC)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_campaign_status "
                "ON recovery_campaigns(status, verified)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_trial_campaign_created "
                "ON recovery_trials(campaign_id, created_at)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_trial_candidate_phase "
                "ON recovery_trials(campaign_id, candidate_id, phase)"
            )
            self._create_unique_index_if_clean(
                "uq_recovery_trial_run",
                "recovery_trials",
                ("run_id",),
                "run_id IS NOT NULL",
            )
            self._create_unique_index_if_clean(
                "uq_recovery_trial_id",
                "recovery_trials",
                ("campaign_id", "trial_id"),
                "trial_id IS NOT NULL",
            )
            self._create_unique_index_if_clean(
                "uq_recovery_request_digest",
                "recovery_trials",
                ("campaign_id", "request_digest"),
                "request_digest IS NOT NULL",
            )
            self._create_unique_index_if_clean(
                "uq_recovery_execution_digest",
                "recovery_trials",
                ("campaign_id", "execution_manifest_digest"),
                "execution_manifest_digest IS NOT NULL",
            )
            c.execute(
                "INSERT INTO storage_metadata(component, version) VALUES('storage', ?) "
                "ON CONFLICT(component) DO UPDATE SET version=excluded.version",
                (STORAGE_SCHEMA_VERSION,),
            )
            c.commit()

    def _create_unique_index_if_clean(
        self,
        index: str,
        table: str,
        columns: tuple[str, ...],
        where: str,
    ) -> None:
        identifiers = (index, table) + columns
        if any(not _IDENTIFIER.fullmatch(item) for item in identifiers):
            raise StorageError("unsafe index identifier")
        grouped = ", ".join(columns)
        duplicate = self._conn.execute(
            "SELECT 1 FROM {} WHERE {} GROUP BY {} HAVING COUNT(*) > 1 LIMIT 1".format(
                table, where, grouped
            )
        ).fetchone()
        if duplicate is None:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}({}) WHERE {}".format(
                    index, table, grouped, where
                )
            )

    # -- runs ---------------------------------------------------------------
    def upsert_run(self, run_id: str, **fields) -> None:
        with self._lock:
            allowed = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(runs)")
            }
            unknown = sorted(set(fields) - allowed)
            if unknown:
                raise StorageError("unknown run fields: {}".format(unknown))
            existing = self._get_run_unlocked(run_id)
            json_fields = {
                "config_json",
                "git_json",
                "env_json",
                "gpu_json",
                "resource_json",
                "warnings_json",
                "tags_json",
            }
            row = dict(existing) if existing else {"run_id": run_id}
            for key, value in fields.items():
                row[key] = (
                    _json_dumps(value, key) if key in json_fields else value
                )
            columns = list(row)
            placeholders = ",".join("?" for _ in columns)
            updates = ",".join(
                "{}=excluded.{}".format(column, column)
                for column in columns
                if column != "run_id"
            )
            conflict = (
                "DO UPDATE SET " + updates if updates else "DO NOTHING"
            )
            self._conn.execute(
                "INSERT INTO runs ({}) VALUES ({}) ON CONFLICT(run_id) {}".format(
                    ",".join(columns), placeholders, conflict
                ),
                [row[column] for column in columns],
            )
            self._conn.commit()

    def _get_run_unlocked(self, run_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._get_run_unlocked(run_id)

    def list_runs(self, project: Optional[str] = None):
        with self._lock:
            if project:
                return self._conn.execute(
                    "SELECT * FROM runs WHERE project=? ORDER BY started_at DESC",
                    (project,),
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC"
            ).fetchall()

    # -- metrics ------------------------------------------------------------
    def log_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        step: Optional[int],
        timestamp: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO metrics(run_id,name,value,step,timestamp) VALUES(?,?,?,?,?)",
                (run_id, name, value, step, timestamp),
            )
            self._conn.commit()

    def get_metrics(self, run_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM metrics WHERE run_id=? ORDER BY timestamp ASC",
                (run_id,),
            ).fetchall()

    def final_metrics(self, run_id: str) -> dict:
        values = {}
        for row in self.get_metrics(run_id):
            values[row["name"]] = row["value"]
        return values

    # -- artifacts ----------------------------------------------------------
    def log_artifact(
        self, run_id: str, path: str, checksum: str, size_bytes: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts(run_id,path,checksum,size_bytes) VALUES(?,?,?,?)",
                (run_id, path, checksum, size_bytes),
            )
            if os.path.basename(path) == RECOVERY_RESULT_FILENAME:
                self._conn.execute(
                    "UPDATE recovery_campaigns SET artifact_path=?, "
                    "artifact_checksum=?, artifact_size_bytes=? WHERE campaign_id=?",
                    (path, checksum, size_bytes, run_id),
                )
            self._conn.commit()

    def get_artifacts(self, run_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM artifacts WHERE run_id=?", (run_id,)
            ).fetchall()

    def artifact_path(self, run_id: str, filename: str) -> str:
        directory = os.path.join(self.root, "artifacts", run_id)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

    # -- failures -----------------------------------------------------------
    def save_failure(
        self,
        run_id: str,
        exception_type: str,
        message: str,
        traceback_str: str,
        diagnosis: dict,
        evidence: dict,
        capsule: Optional[dict] = None,
    ) -> None:
        schema = (capsule or {}).get("schema") or {}
        schema_version = (capsule or {}).get("capsule_schema_version") or schema.get(
            "version"
        )
        failure_class = (capsule or {}).get("failure_class") or diagnosis.get("rule")
        captured_at = (capsule or {}).get("captured_at")
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO failures(
                    run_id,exception_type,message,traceback,diagnosis_json,evidence_json,
                    capsule_schema_version,failure_class,captured_at,capsule_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    exception_type,
                    message,
                    traceback_str,
                    _json_dumps(diagnosis, "diagnosis"),
                    _json_dumps(evidence, "evidence"),
                    schema_version,
                    failure_class,
                    captured_at,
                    _json_dumps(capsule, "capsule") if capsule else None,
                ),
            )
            self._conn.commit()

    def get_failure(self, run_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM failures WHERE run_id=?", (run_id,)
            ).fetchone()

    def get_failure_capsule(self, run_id: str) -> Optional[dict]:
        row = self.get_failure(run_id)
        if row is None:
            return None
        if row["capsule_json"]:
            return _json_loads(row["capsule_json"], "failure capsule")
        diagnosis = _json_loads(row["diagnosis_json"] or "{}", "diagnosis")
        evidence = _json_loads(row["evidence_json"] or "{}", "evidence")
        return {
            "capsule_schema_version": row["capsule_schema_version"] or "legacy",
            "run_id": run_id,
            "exception_type": row["exception_type"],
            "message": row["message"],
            "traceback": row["traceback"],
            "failure_class": row["failure_class"]
            or diagnosis.get("rule", "unclassified"),
            "classification": diagnosis,
            "diagnosis": diagnosis,
            "evidence": evidence,
            "evidence_index": [],
            "capture_completeness": None,
        }

    def list_failures(self, project: Optional[str] = None):
        with self._lock:
            if project:
                return self._conn.execute(
                    """SELECT f.*, r.project FROM failures f
                       JOIN runs r ON f.run_id=r.run_id
                       WHERE r.project=? ORDER BY r.started_at DESC""",
                    (project,),
                ).fetchall()
            return self._conn.execute(
                """SELECT f.*, r.project FROM failures f
                   JOIN runs r ON f.run_id=r.run_id
                   ORDER BY r.started_at DESC"""
            ).fetchall()

    # -- resource samples ---------------------------------------------------
    def save_resource_samples(self, run_id: str, samples: list) -> None:
        if not samples:
            return
        rows = [
            (
                run_id,
                sample.get("t"),
                sample.get("cpu_pct"),
                sample.get("ram_pct"),
                sample.get("gpu_util_pct"),
                sample.get("gpu_mem_used_mib"),
                sample.get("disk_read_mbps"),
                sample.get("disk_write_mbps"),
                sample.get("net_sent_mbps"),
                sample.get("net_recv_mbps"),
            )
            for sample in samples
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT INTO resource_samples(
                    run_id,t,cpu_pct,ram_pct,gpu_util_pct,gpu_mem_used_mib,
                    disk_read_mbps,disk_write_mbps,net_sent_mbps,net_recv_mbps
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()

    def get_resource_samples(self, run_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM resource_samples WHERE run_id=? ORDER BY t ASC",
                (run_id,),
            ).fetchall()

    # -- run labels and resolution -----------------------------------------
    def set_run_display_name(self, run_id: str, display_name: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET display_name=? WHERE run_id=?",
                (display_name, run_id),
            )
            self._conn.commit()

    def set_run_tags(self, run_id: str, tags: list) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET tags_json=? WHERE run_id=?",
                (_json_dumps(tags, "tags"), run_id),
            )
            self._conn.commit()

    def set_run_resolved(
        self, run_id: str, resolved: bool, note: Optional[str] = None
    ) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET resolved=?, resolved_note=? WHERE run_id=?",
                (1 if resolved else 0, note, run_id),
            )
            if cursor.rowcount != 1:
                raise StorageError("run {!r} was not found".format(run_id))
            self._conn.commit()

    # -- recovery campaigns ------------------------------------------------
    def create_recovery_campaign(
        self,
        campaign_id: str,
        project: str,
        source_run_id: str,
        contract: dict,
        started_at: float,
    ) -> None:
        encoded_contract = _json_dumps(contract, "recovery contract")
        digest = hashlib.sha256(encoded_contract.encode("utf-8")).hexdigest()
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO recovery_campaigns(
                        campaign_id,project,source_run_id,contract_json,started_at,
                        schema_version,status,contract_digest,verified
                    ) VALUES(?,?,?,?,?,'1.0','running',?,0)""",
                    (
                        campaign_id,
                        project,
                        source_run_id,
                        encoded_contract,
                        started_at,
                        digest,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StorageConflictError(
                    "recovery campaign {!r} already exists".format(campaign_id)
                ) from exc
            self._conn.commit()

    def finish_recovery_campaign(
        self,
        campaign_id: str,
        ended_at: float,
        stopped_reason: str,
        best_run_id: Optional[str],
        report: dict,
    ) -> None:
        encoded_report = _json_dumps(report, "recovery report")
        with self._lock:
            row = self._conn.execute(
                "SELECT ended_at, report_json FROM recovery_campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise StorageError(
                    "recovery campaign {!r} was not found".format(campaign_id)
                )
            if row["ended_at"] is not None:
                if row["report_json"] == encoded_report:
                    return
                raise StorageConflictError(
                    "finished recovery campaigns are immutable"
                )

            normalized = _recovery_report_fields(report, stopped_reason)
            legacy_best = (
                None
                if normalized["authoritative_v1"]
                else best_run_id
            )
            self._conn.execute(
                """UPDATE recovery_campaigns SET
                    ended_at=?, stopped_reason=?, best_run_id=?, report_json=?,
                    schema_version=?, status=?, contract_digest=COALESCE(?,contract_digest),
                    preparation_digest=?, report_digest=?, verified=?,
                    verified_candidate_id=?, verified_run_ids_json=?,
                    planned_candidate_ids_json=?, probe_survivor_ids_json=?,
                    executed_proposal_ids_json=?, skipped_proposals_json=?,
                    usage_json=?, ranking_json=?, verification_reports_json=?
                   WHERE campaign_id=?""",
                (
                    ended_at,
                    normalized["stopped_reason"],
                    legacy_best,
                    encoded_report,
                    normalized["schema_version"],
                    normalized["status"],
                    normalized["contract_digest"],
                    normalized["preparation_digest"],
                    hashlib.sha256(encoded_report.encode("utf-8")).hexdigest(),
                    1 if normalized["verified"] else 0,
                    normalized["verified_candidate_id"],
                    _json_dumps(normalized["verified_run_ids"], "verified run ids"),
                    _json_dumps(normalized["planned_candidate_ids"], "planned candidates"),
                    _json_dumps(normalized["probe_survivor_ids"], "probe survivors"),
                    _json_dumps(normalized["executed_proposal_ids"], "executed proposals"),
                    _json_dumps(normalized["skipped_proposals"], "skipped proposals"),
                    _json_dumps(normalized["usage"], "campaign usage")
                    if normalized["usage"] is not None
                    else None,
                    _json_dumps(normalized["ranking"], "campaign ranking")
                    if normalized["ranking"] is not None
                    else None,
                    _json_dumps(normalized["verifications"], "verifications"),
                    campaign_id,
                ),
            )
            self._synchronize_trials_unlocked(
                campaign_id,
                normalized["trials"],
                set(normalized["verified_run_ids"]),
                ended_at,
            )
            self._synchronize_proposals_unlocked(campaign_id, report, normalized)
            self._synchronize_verifications_unlocked(
                campaign_id, normalized["verifications"]
            )
            self._conn.commit()

    def get_recovery_campaign(self, campaign_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM recovery_campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()

    def get_recovery_campaign_report(self, campaign_id: str) -> Optional[dict]:
        row = self.get_recovery_campaign(campaign_id)
        if row is None or not row["report_json"]:
            return None
        return _json_loads(row["report_json"], "recovery report")

    def list_recovery_campaigns(
        self,
        project: Optional[str] = None,
        *,
        status: Optional[str] = None,
        verified: Optional[bool] = None,
    ):
        clauses = []
        values = []
        if project is not None:
            clauses.append("project=?")
            values.append(project)
        if status is not None:
            clauses.append("status=?")
            values.append(status)
        if verified is not None:
            clauses.append("verified=?")
            values.append(1 if verified else 0)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM recovery_campaigns{} ORDER BY started_at DESC".format(
                    where
                ),
                tuple(values),
            ).fetchall()

    def save_recovery_trial(
        self,
        campaign_id: str,
        run_id: str,
        phase: str,
        hypothesis: Optional[dict],
        patch: dict,
        rationale: str,
        confidence: Optional[float],
        outcome: str,
        score: Optional[float],
        verified: bool,
        created_at: float,
        *,
        trial_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        proposal_id: Optional[str] = None,
        policy_rule: Optional[str] = None,
        status: Optional[str] = None,
        failure_class: Optional[str] = None,
        request_digest: Optional[str] = None,
        execution_manifest_digest: Optional[str] = None,
        worker_pid: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        gpu_seconds: Optional[float] = None,
        progress_steps: Optional[int] = None,
        peak_vram_bytes: Optional[int] = None,
        metrics: Optional[dict] = None,
        workload_identity: Optional[dict] = None,
        environment_patch: Optional[dict] = None,
        trial: Optional[dict] = None,
    ) -> None:
        metadata = hypothesis or {}
        inferred_proposal = proposal_id or metadata.get("proposal_id")
        inferred_candidate = candidate_id or inferred_proposal
        inferred_rule = policy_rule or metadata.get("policy_rule")
        config_patch = dict(patch or {})
        embedded_environment = config_patch.pop("__environment__", {})
        environment = environment_patch or embedded_environment or {}
        with self._lock:
            if self.get_recovery_campaign(campaign_id) is None:
                raise StorageError(
                    "recovery campaign {!r} was not found".format(campaign_id)
                )
            duplicate = self._conn.execute(
                "SELECT 1 FROM recovery_trials WHERE run_id=? LIMIT 1", (run_id,)
            ).fetchone()
            if duplicate is not None:
                raise StorageConflictError(
                    "recovery run_id {!r} must be unique".format(run_id)
                )
            self._assert_trial_identity_available_unlocked(
                campaign_id,
                trial_id=trial_id,
                request_digest=request_digest,
                execution_manifest_digest=execution_manifest_digest,
            )
            self._conn.execute(
                """INSERT INTO recovery_trials(
                    campaign_id,run_id,phase,hypothesis_json,patch_json,rationale,
                    confidence,outcome,score,verified,created_at,trial_id,candidate_id,
                    proposal_id,policy_rule,status,failure_class,request_digest,
                    execution_manifest_digest,worker_pid,duration_seconds,gpu_seconds,
                    progress_steps,peak_vram_bytes,metrics_json,workload_identity_json,
                    environment_patch_json,trial_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    campaign_id,
                    run_id,
                    phase,
                    _json_dumps(hypothesis, "trial metadata") if hypothesis else None,
                    _json_dumps(config_patch, "config patch"),
                    rationale,
                    confidence,
                    outcome,
                    score,
                    1 if verified else 0,
                    created_at,
                    trial_id,
                    inferred_candidate,
                    inferred_proposal,
                    inferred_rule,
                    status or outcome,
                    failure_class,
                    request_digest,
                    execution_manifest_digest,
                    worker_pid,
                    duration_seconds,
                    gpu_seconds,
                    progress_steps,
                    peak_vram_bytes,
                    _json_dumps(metrics or {}, "trial metrics"),
                    _json_dumps(workload_identity or {}, "workload identity"),
                    _json_dumps(environment, "environment patch"),
                    _json_dumps(trial, "campaign trial") if trial else None,
                ),
            )
            self._conn.commit()

    def get_recovery_trial(
        self, campaign_id: str, run_id: str
    ) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM recovery_trials WHERE campaign_id=? AND run_id=?",
                (campaign_id, run_id),
            ).fetchone()

    def list_recovery_trials(self, campaign_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM recovery_trials WHERE campaign_id=? ORDER BY created_at ASC",
                (campaign_id,),
            ).fetchall()

    def list_recovery_proposals(self, campaign_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM recovery_proposals WHERE campaign_id=? "
                "ORDER BY rowid ASC",
                (campaign_id,),
            ).fetchall()

    def list_recovery_verifications(self, campaign_id: str):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM recovery_verifications WHERE campaign_id=? "
                "ORDER BY ordinal ASC",
                (campaign_id,),
            ).fetchall()

    def _assert_trial_identity_available_unlocked(
        self,
        campaign_id: str,
        *,
        trial_id: Optional[str],
        request_digest: Optional[str],
        execution_manifest_digest: Optional[str],
        excluding_run_id: Optional[str] = None,
    ) -> None:
        for column, value in (
            ("trial_id", trial_id),
            ("request_digest", request_digest),
            ("execution_manifest_digest", execution_manifest_digest),
        ):
            if value is None:
                continue
            sql = "SELECT run_id FROM recovery_trials WHERE campaign_id=? AND {}=?".format(
                column
            )
            values = [campaign_id, value]
            if excluding_run_id is not None:
                sql += " AND run_id<>?"
                values.append(excluding_run_id)
            if self._conn.execute(sql + " LIMIT 1", tuple(values)).fetchone():
                raise StorageConflictError(
                    "{} must be unique within a recovery campaign".format(column)
                )

    def _synchronize_trials_unlocked(
        self,
        campaign_id: str,
        trials: list,
        verified_run_ids: set[str],
        ended_at: float,
    ) -> None:
        seen = {"trial_id": set(), "run_id": set(), "request_digest": set(),
                "execution_manifest_digest": set()}
        for index, trial in enumerate(trials):
            if not isinstance(trial, Mapping):
                raise StorageError("campaign trials must be objects")
            for field in seen:
                value = trial.get(field)
                if not isinstance(value, str) or not value:
                    raise StorageError("campaign trial {} is missing".format(field))
                if value in seen[field]:
                    raise StorageConflictError(
                        "campaign trial {} values must be unique".format(field)
                    )
                seen[field].add(value)
            run_id = trial["run_id"]
            self._assert_trial_identity_available_unlocked(
                campaign_id,
                trial_id=trial["trial_id"],
                request_digest=trial["request_digest"],
                execution_manifest_digest=trial["execution_manifest_digest"],
                excluding_run_id=run_id,
            )
            existing = self._conn.execute(
                "SELECT 1 FROM recovery_trials WHERE campaign_id=? AND run_id=?",
                (campaign_id, run_id),
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    """INSERT INTO recovery_trials(
                        campaign_id,run_id,phase,patch_json,rationale,outcome,verified,
                        created_at,metrics_json,workload_identity_json,
                        environment_patch_json
                    ) VALUES(?,?,?,'{}','',?,?,?,'{}','{}','{}')""",
                    (
                        campaign_id,
                        run_id,
                        trial.get("phase"),
                        trial.get("status"),
                        1 if run_id in verified_run_ids else 0,
                        ended_at + index * 0.000001,
                    ),
                )
            self._conn.execute(
                """UPDATE recovery_trials SET
                    phase=?, trial_id=?, candidate_id=?, proposal_id=?, status=?,
                    outcome=?, failure_class=?, request_digest=?,
                    execution_manifest_digest=?, worker_pid=?, duration_seconds=?,
                    gpu_seconds=?, progress_steps=?, peak_vram_bytes=?, metrics_json=?,
                    workload_identity_json=?, verified=?, trial_json=?
                   WHERE campaign_id=? AND run_id=?""",
                (
                    trial.get("phase"),
                    trial.get("trial_id"),
                    trial.get("candidate_id"),
                    trial.get("candidate_id"),
                    trial.get("status"),
                    trial.get("status"),
                    trial.get("failure_class"),
                    trial.get("request_digest"),
                    trial.get("execution_manifest_digest"),
                    trial.get("worker_pid"),
                    trial.get("duration_seconds"),
                    trial.get("gpu_seconds"),
                    trial.get("progress_steps"),
                    trial.get("peak_vram_bytes"),
                    _json_dumps(trial.get("metrics") or {}, "trial metrics"),
                    _json_dumps(
                        trial.get("workload_identity") or {}, "workload identity"
                    ),
                    1 if run_id in verified_run_ids else 0,
                    _json_dumps(dict(trial), "campaign trial"),
                    campaign_id,
                    run_id,
                ),
            )

    def _synchronize_proposals_unlocked(
        self, campaign_id: str, report: dict, normalized: dict
    ) -> None:
        preparation = report.get("preparation") or {}
        plan = preparation.get("policy_plan") or {}
        proposals = plan.get("proposals") or []
        automatic = set(plan.get("automatic_proposal_ids") or [])
        approval = set(plan.get("approval_required_proposal_ids") or [])
        executed = set(normalized["executed_proposal_ids"])
        skipped = {
            item.get("proposal_id"): item
            for item in normalized["skipped_proposals"]
            if isinstance(item, Mapping)
        }
        self._conn.execute(
            "DELETE FROM recovery_proposals WHERE campaign_id=?", (campaign_id,)
        )
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                raise StorageError("policy proposals must be objects")
            proposal_id = proposal.get("proposal_id")
            if proposal_id in automatic:
                mode = "automatic"
            elif proposal_id in approval:
                mode = "approval_required"
            else:
                raise StorageError("proposal authorization mode is missing")
            skip = skipped.get(proposal_id)
            state = "executed" if proposal_id in executed else "skipped"
            if state == "skipped" and skip is None:
                raise StorageError("every unexecuted proposal needs a skip record")
            self._conn.execute(
                """INSERT INTO recovery_proposals(
                    campaign_id,proposal_id,policy_rule,authorization_mode,state,
                    skip_code,skip_reason,rationale,proposal_json
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    campaign_id,
                    proposal_id,
                    proposal.get("policy_rule"),
                    mode,
                    state,
                    skip.get("code") if skip else None,
                    skip.get("reason") if skip else None,
                    proposal.get("rationale") or "",
                    _json_dumps(dict(proposal), "intervention proposal"),
                ),
            )

    def _synchronize_verifications_unlocked(
        self, campaign_id: str, verifications: list
    ) -> None:
        self._conn.execute(
            "DELETE FROM recovery_verifications WHERE campaign_id=?", (campaign_id,)
        )
        for ordinal, report in enumerate(verifications):
            if not isinstance(report, Mapping):
                raise StorageError("verification reports must be objects")
            self._conn.execute(
                """INSERT INTO recovery_verifications(
                    campaign_id,candidate_id,verified,confirmation_run_ids_json,
                    report_json,ordinal
                ) VALUES(?,?,?,?,?,?)""",
                (
                    campaign_id,
                    report.get("candidate_id"),
                    1 if report.get("verified") is True else 0,
                    _json_dumps(
                        report.get("confirmation_run_ids") or [],
                        "confirmation run ids",
                    ),
                    _json_dumps(dict(report), "verification report"),
                    ordinal,
                ),
            )

    # -- verified resolution memory ---------------------------------------
    def resolution_memory(self, project: Optional[str] = None) -> list:
        """Aggregate full-trial outcomes without calling them recoveries.

        ``successes`` means a full trial completed. ``verified_recoveries``
        means the candidate later passed the independent confirmation verifier.
        """
        sql = """SELECT rt.patch_json, rt.status, rt.outcome, rt.candidate_id,
                        rc.source_run_id, rc.project, rc.verified_candidate_id
                 FROM recovery_trials rt
                 JOIN recovery_campaigns rc ON rt.campaign_id=rc.campaign_id
                 WHERE rt.phase='full'"""
        values = ()
        if project is not None:
            sql += " AND rc.project=?"
            values = (project,)
        with self._lock:
            trial_rows = self._conn.execute(sql, values).fetchall()

        signatures = {}
        for row in trial_rows:
            failure = self.get_failure(row["source_run_id"])
            if failure is None:
                continue
            diagnosis = _json_loads(failure["diagnosis_json"] or "{}", "diagnosis")
            patch = _json_loads(row["patch_json"] or "{}", "config patch")
            patch.pop("__environment__", None)
            key = (
                diagnosis.get("rule", "unknown"),
                tuple(sorted(patch)),
            )
            entry = signatures.setdefault(
                key,
                {
                    "failure_class": key[0],
                    "patch_keys": list(key[1]),
                    "attempts": 0,
                    "successes": 0,
                    "verified_recoveries": 0,
                    "example_patches": [],
                },
            )
            entry["attempts"] += 1
            if (row["status"] or row["outcome"]) == "success":
                entry["successes"] += 1
            if row["candidate_id"] == row["verified_candidate_id"]:
                entry["verified_recoveries"] += 1
            if patch not in entry["example_patches"] and len(entry["example_patches"]) < 3:
                entry["example_patches"].append(patch)
        results = list(signatures.values())
        for item in results:
            item["success_rate"] = item["successes"] / item["attempts"]
            item["verification_rate"] = (
                item["verified_recoveries"] / item["attempts"]
            )
        results.sort(
            key=lambda item: (item["verified_recoveries"], item["attempts"]),
            reverse=True,
        )
        return results


def _recovery_report_fields(report: dict, stopped_reason: str) -> dict:
    if not isinstance(report, Mapping):
        raise StorageError("recovery report must be an object")
    schema = report.get("schema") or {}
    campaign = report.get("campaign") or {}
    authoritative = schema.get("name") == "watcherml.recovery-result"
    status = campaign.get("status") or report.get("status") or "finished"
    verified = campaign.get("verified") is True
    if authoritative and report.get("verified") is not verified:
        raise StorageError("recovery report verified flags are inconsistent")
    normalized_reason = campaign.get("stopped_reason") or stopped_reason
    if verified and normalized_reason != "verified_recovery":
        raise StorageError("verified campaign lacks verified_recovery reason")
    if not verified and normalized_reason == "verified_recovery":
        raise StorageError("unverified campaign claims verified_recovery")
    collections = {
        "verified_run_ids": campaign.get("verified_run_ids") or [],
        "planned_candidate_ids": campaign.get("planned_candidate_ids") or [],
        "probe_survivor_ids": campaign.get("probe_survivor_ids") or [],
        "trials": campaign.get("trials") or [],
        "verifications": campaign.get("verifications") or [],
        "executed_proposal_ids": report.get("executed_proposal_ids") or [],
        "skipped_proposals": report.get("skipped_proposals") or [],
    }
    if any(not isinstance(value, list) for value in collections.values()):
        raise StorageError("recovery report collections must be arrays")
    return {
        "authoritative_v1": authoritative,
        "schema_version": schema.get("version") or "legacy",
        "status": status,
        "stopped_reason": normalized_reason,
        "contract_digest": campaign.get("contract_digest"),
        "preparation_digest": report.get("preparation_digest"),
        "verified": verified,
        "verified_candidate_id": campaign.get("verified_candidate_id"),
        "usage": campaign.get("usage"),
        "ranking": campaign.get("ranking"),
        **collections,
    }


def _json_dumps(value, name: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StorageError("{} is not strict JSON".format(name)) from exc


def _json_loads(encoded: str, name: str):
    try:
        return json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StorageError("stored {} is invalid JSON".format(name)) from exc