"""Local-mode storage: SQLite metadata + a content-addressed artifact directory.

No Docker, no Postgres. This is the default 'local mode' backend described in
the WatcherML product spec. A server-mode adapter (Postgres + S3) can implement
the same interface later without changing SDK call sites.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Optional

DEFAULT_DIR = os.path.join(os.getcwd(), ".watcherml")


class Storage:
    def __init__(self, root: str = DEFAULT_DIR):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "artifacts"), exist_ok=True)
        self.db_path = os.path.join(self.root, "watcher.db")
        # check_same_thread=False + an explicit lock: the CLI only ever uses one
        # thread, but the web UI (FastAPI) runs sync endpoints in a thread pool,
        # so the same Storage instance can be hit from multiple threads.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _migrate_add_column(self, table: str, column: str, coltype: str):
        """Adds a column if it doesn't already exist -- lets an existing local
        .watcherml/watcher.db upgrade in place across WatcherML versions
        instead of requiring a wipe. SQLite has no 'ADD COLUMN IF NOT EXISTS',
        so we just swallow the 'duplicate column' error."""
        try:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

    def _init_schema(self):
        with self._lock:
            c = self._conn
            c.execute("""
                CREATE TABLE IF NOT EXISTS runs (
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
                )
            """)
            self._migrate_add_column("runs", "display_name", "TEXT")
            self._migrate_add_column("runs", "tags_json", "TEXT")
            self._migrate_add_column("runs", "resolved", "INTEGER DEFAULT 0")
            self._migrate_add_column("runs", "resolved_note", "TEXT")
            self._migrate_add_column("runs", "capsule_schema_version", "TEXT")
            self._migrate_add_column("runs", "capture_completeness", "REAL")
            c.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT,
                    name TEXT,
                    value REAL,
                    step INTEGER,
                    timestamp REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT,
                    path TEXT,
                    checksum TEXT,
                    size_bytes INTEGER
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS failures (
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
                )
            """)
            self._migrate_add_column("failures", "capsule_schema_version", "TEXT")
            self._migrate_add_column("failures", "failure_class", "TEXT")
            self._migrate_add_column("failures", "captured_at", "REAL")
            self._migrate_add_column("failures", "capsule_json", "TEXT")
            c.execute("""
                CREATE TABLE IF NOT EXISTS resource_samples (
                    run_id TEXT,
                    t REAL,
                    cpu_pct REAL,
                    ram_pct REAL,
                    gpu_util_pct REAL,
                    gpu_mem_used_mib REAL
                )
            """)
            self._migrate_add_column("resource_samples", "disk_read_mbps", "REAL")
            self._migrate_add_column("resource_samples", "disk_write_mbps", "REAL")
            self._migrate_add_column("resource_samples", "net_sent_mbps", "REAL")
            self._migrate_add_column("resource_samples", "net_recv_mbps", "REAL")
            c.execute("""
                CREATE TABLE IF NOT EXISTS recovery_campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    project TEXT,
                    source_run_id TEXT,
                    contract_json TEXT,
                    started_at REAL,
                    ended_at REAL,
                    stopped_reason TEXT,
                    best_run_id TEXT,
                    report_json TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS recovery_trials (
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
                )
            """)
            c.commit()

    # -- runs ----------------------------------------------------------
    def upsert_run(self, run_id: str, **fields):
        with self._lock:
            existing = self._get_run_unlocked(run_id)
            json_fields = {"config_json", "git_json", "env_json", "gpu_json",
                           "resource_json", "warnings_json"}
            row = dict(existing) if existing else {"run_id": run_id}
            for k, v in fields.items():
                row[k] = json.dumps(v, sort_keys=True, default=str) if k in json_fields else v
            cols = list(row.keys())
            placeholders = ",".join("?" for _ in cols)
            updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "run_id")
            sql = (f"INSERT INTO runs ({','.join(cols)}) VALUES ({placeholders}) "
                   f"ON CONFLICT(run_id) DO UPDATE SET {updates}")
            self._conn.execute(sql, [row[c] for c in cols])
            self._conn.commit()

    def _get_run_unlocked(self, run_id: str) -> Optional[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return cur.fetchone()

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._get_run_unlocked(run_id)

    def list_runs(self, project: Optional[str] = None):
        with self._lock:
            if project:
                cur = self._conn.execute(
                    "SELECT * FROM runs WHERE project=? ORDER BY started_at DESC", (project,))
            else:
                cur = self._conn.execute("SELECT * FROM runs ORDER BY started_at DESC")
            return cur.fetchall()

    # -- metrics ---------------------------------------------------------
    def log_metric(self, run_id: str, name: str, value: float, step: Optional[int], timestamp: float):
        with self._lock:
            self._conn.execute(
                "INSERT INTO metrics (run_id, name, value, step, timestamp) VALUES (?,?,?,?,?)",
                (run_id, name, value, step, timestamp),
            )
            self._conn.commit()

    def get_metrics(self, run_id: str):
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM metrics WHERE run_id=? ORDER BY timestamp ASC", (run_id,))
            return cur.fetchall()

    def final_metrics(self, run_id: str) -> dict:
        """Last logged value per metric name."""
        rows = self.get_metrics(run_id)
        out = {}
        for r in rows:
            out[r["name"]] = r["value"]
        return out

    # -- artifacts ---------------------------------------------------------
    def log_artifact(self, run_id: str, path: str, checksum: str, size_bytes: int):
        with self._lock:
            self._conn.execute(
                "INSERT INTO artifacts (run_id, path, checksum, size_bytes) VALUES (?,?,?,?)",
                (run_id, path, checksum, size_bytes),
            )
            self._conn.commit()

    def get_artifacts(self, run_id: str):
        with self._lock:
            cur = self._conn.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,))
            return cur.fetchall()

    # -- failures ---------------------------------------------------------
    def save_failure(self, run_id: str, exception_type: str, message: str,
                     traceback_str: str, diagnosis: dict, evidence: dict,
                     capsule: Optional[dict] = None):
        """Persist both query-friendly columns and the complete public capsule.

        ``capsule`` is optional so older internal callers keep working during
        the v1 migration. New failure paths must always provide it.
        """
        schema_version = (capsule or {}).get("capsule_schema_version")
        failure_class = (capsule or {}).get("failure_class") or diagnosis.get("rule")
        captured_at = (capsule or {}).get("captured_at")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO failures "
                "(run_id, exception_type, message, traceback, diagnosis_json, evidence_json, "
                "capsule_schema_version, failure_class, captured_at, capsule_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, exception_type, message, traceback_str,
                 json.dumps(diagnosis, sort_keys=True, default=str),
                 json.dumps(evidence, sort_keys=True, default=str),
                 schema_version, failure_class, captured_at,
                 json.dumps(capsule, sort_keys=True, default=str) if capsule else None),
            )
            self._conn.commit()

    def get_failure(self, run_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM failures WHERE run_id=?", (run_id,))
            return cur.fetchone()

    def get_failure_capsule(self, run_id: str) -> Optional[dict]:
        """Return the full v1 capsule, or a marked legacy view for old rows."""
        row = self.get_failure(run_id)
        if row is None:
            return None
        if row["capsule_json"]:
            return json.loads(row["capsule_json"])
        diagnosis = json.loads(row["diagnosis_json"] or "{}")
        evidence = json.loads(row["evidence_json"] or "{}")
        return {
            "capsule_schema_version": row["capsule_schema_version"] or "legacy",
            "run_id": run_id,
            "exception_type": row["exception_type"],
            "message": row["message"],
            "traceback": row["traceback"],
            "failure_class": row["failure_class"] or diagnosis.get("rule", "unclassified"),
            "classification": diagnosis,
            "diagnosis": diagnosis,
            "evidence": evidence,
            "evidence_index": [],
            "capture_completeness": None,
        }

    def list_failures(self, project: Optional[str] = None):
        with self._lock:
            if project:
                cur = self._conn.execute("""
                    SELECT f.*, r.project FROM failures f JOIN runs r ON f.run_id = r.run_id
                    WHERE r.project=? ORDER BY r.started_at DESC
                """, (project,))
            else:
                cur = self._conn.execute("""
                    SELECT f.*, r.project FROM failures f JOIN runs r ON f.run_id = r.run_id
                    ORDER BY r.started_at DESC
                """)
            return cur.fetchall()

    def artifact_path(self, run_id: str, filename: str) -> str:
        d = os.path.join(self.root, "artifacts", run_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, filename)

    # -- resource time series (for the trace visualization) ------------------
    def save_resource_samples(self, run_id: str, samples: list):
        if not samples:
            return
        rows = [
            (run_id, s.get("t"), s.get("cpu_pct"), s.get("ram_pct"),
             s.get("gpu_util_pct"), s.get("gpu_mem_used_mib"),
             s.get("disk_read_mbps"), s.get("disk_write_mbps"),
             s.get("net_sent_mbps"), s.get("net_recv_mbps"))
            for s in samples
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO resource_samples (run_id, t, cpu_pct, ram_pct, gpu_util_pct, "
                "gpu_mem_used_mib, disk_read_mbps, disk_write_mbps, net_sent_mbps, net_recv_mbps) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def get_resource_samples(self, run_id: str):
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM resource_samples WHERE run_id=? ORDER BY t ASC", (run_id,))
            return cur.fetchall()

    # -- run naming, tagging, resolution status -----------------------------
    def set_run_display_name(self, run_id: str, display_name: Optional[str]):
        with self._lock:
            self._conn.execute("UPDATE runs SET display_name=? WHERE run_id=?",
                                (display_name, run_id))
            self._conn.commit()

    def set_run_tags(self, run_id: str, tags: list):
        with self._lock:
            self._conn.execute("UPDATE runs SET tags_json=? WHERE run_id=?",
                                (json.dumps(tags), run_id))
            self._conn.commit()

    def set_run_resolved(self, run_id: str, resolved: bool, note: Optional[str] = None):
        with self._lock:
            self._conn.execute("UPDATE runs SET resolved=?, resolved_note=? WHERE run_id=?",
                                (1 if resolved else 0, note, run_id))
            self._conn.commit()

    # -- resolution memory ---------------------------------------------------
    def resolution_memory(
    self,
    project: Optional[str] = None,
) -> list:
        """Aggregate completed recovery trials by failure and intervention shape.

        A full trial always counts as an attempt.

        A successful full trial counts as a completed success, but it only counts
        as verified when the independent confirmation verifier has promoted it.
        These concepts must remain separate.
        """
        with self._lock:
            if project is None:
                cursor = self._conn.execute(
                    """
                    SELECT
                        rt.patch_json,
                        rt.outcome,
                        rt.verified,
                        rc.source_run_id,
                        rc.project
                    FROM recovery_trials AS rt
                    JOIN recovery_campaigns AS rc
                        ON rt.campaign_id = rc.campaign_id
                    WHERE rt.phase = 'full'
                    """
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT
                        rt.patch_json,
                        rt.outcome,
                        rt.verified,
                        rc.source_run_id,
                        rc.project
                    FROM recovery_trials AS rt
                    JOIN recovery_campaigns AS rc
                        ON rt.campaign_id = rc.campaign_id
                    WHERE rt.phase = 'full'
                    AND rc.project = ?
                    """,
                    (project,),
                )

            trial_rows = cursor.fetchall()

        signatures = {}

        for row in trial_rows:
            failure_row = self.get_failure(
                row["source_run_id"]
            )

            if failure_row is None:
                continue

            diagnosis = json.loads(
                failure_row["diagnosis_json"] or "{}"
            )

            failure_class = (
                failure_row["failure_class"]
                or diagnosis.get("rule")
                or "unknown"
            )

            try:
                patch = json.loads(
                    row["patch_json"] or "{}"
                )
            except (TypeError, json.JSONDecodeError):
                patch = {}

            if not isinstance(patch, dict):
                patch = {}

            patch_shape = tuple(
                sorted(patch.keys())
            )

            signature_key = (
                failure_class,
                patch_shape,
            )

            entry = signatures.setdefault(
                signature_key,
                {
                    "failure_class": failure_class,
                    "patch_keys": list(patch_shape),

                    # Every full trial counts here.
                    "attempts": 0,

                    # Full trials that completed without failure.
                    "completed_successes": 0,

                    # Full trials independently confirmed later.
                    "verified_successes": 0,

                    "failed_attempts": 0,
                    "example_patches": [],
                },
            )

            entry["attempts"] += 1

            if row["outcome"] == "success":
                entry["completed_successes"] += 1
            else:
                entry["failed_attempts"] += 1

            if bool(row["verified"]):
                entry["verified_successes"] += 1

            if (
                patch not in entry["example_patches"]
                and len(entry["example_patches"]) < 3
            ):
                entry["example_patches"].append(patch)

        results = list(signatures.values())

        for entry in results:
            attempts = entry["attempts"]

            entry["completion_rate"] = (
                entry["completed_successes"] / attempts
                if attempts
                else 0.0
            )

            entry["verified_success_rate"] = (
                entry["verified_successes"] / attempts
                if attempts
                else 0.0
            )

            # Compatibility fields for the current API/UI. Here, "success" means
            # that a full trial completed, not that it was independently verified.
            entry["successes"] = entry[
                "completed_successes"
            ]
            entry["success_rate"] = entry[
                "completion_rate"
            ]

        results.sort(
            key=lambda entry: (
                entry["attempts"],
                entry["completed_successes"],
            ),
            reverse=True,
        )

        return results
    # -- recovery agent memory (campaigns + trials) --------------------------
    def create_recovery_campaign(self, campaign_id: str, project: str, source_run_id: str,
                                  contract: dict, started_at: float):
        with self._lock:
            self._conn.execute(
                "INSERT INTO recovery_campaigns "
                "(campaign_id, project, source_run_id, contract_json, started_at) "
                "VALUES (?,?,?,?,?)",
                (campaign_id, project, source_run_id, json.dumps(contract), started_at),
            )
            self._conn.commit()

    def finish_recovery_campaign(self, campaign_id: str, ended_at: float, stopped_reason: str,
                                  best_run_id: Optional[str], report: dict):
        with self._lock:
            self._conn.execute(
                "UPDATE recovery_campaigns SET ended_at=?, stopped_reason=?, best_run_id=?, "
                "report_json=? WHERE campaign_id=?",
                (ended_at, stopped_reason, best_run_id, json.dumps(report), campaign_id),
            )
            self._conn.commit()

    def get_recovery_campaign(self, campaign_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM recovery_campaigns WHERE campaign_id=?", (campaign_id,))
            return cur.fetchone()

    def list_recovery_campaigns(self, project: Optional[str] = None):
        with self._lock:
            if project:
                cur = self._conn.execute(
                    "SELECT * FROM recovery_campaigns WHERE project=? ORDER BY started_at DESC",
                    (project,))
            else:
                cur = self._conn.execute(
                    "SELECT * FROM recovery_campaigns ORDER BY started_at DESC")
            return cur.fetchall()

    def save_recovery_trial(self, campaign_id: str, run_id: str, phase: str,
                             hypothesis: Optional[dict], patch: dict, rationale: str,
                             confidence: Optional[float], outcome: str, score: Optional[float],
                             verified: bool, created_at: float):
        with self._lock:
            self._conn.execute(
                "INSERT INTO recovery_trials "
                "(campaign_id, run_id, phase, hypothesis_json, patch_json, rationale, "
                "confidence, outcome, score, verified, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (campaign_id, run_id, phase, json.dumps(hypothesis) if hypothesis else None,
                 json.dumps(patch), rationale, confidence, outcome, score,
                 1 if verified else 0, created_at),
            )
            self._conn.commit()

    def list_recovery_trials(self, campaign_id: str):
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM recovery_trials WHERE campaign_id=? ORDER BY created_at ASC",
                (campaign_id,))
            return cur.fetchall()
