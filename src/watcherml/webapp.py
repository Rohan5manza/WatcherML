"""Read-only local web API for WatcherML v1 evidence and audit trails.

The browser surface reads the same SQLite database as the SDK and CLI.  It can
label runs and export evidence, but it never launches GPU work, authorizes an
intervention, promotes a recovery, or mutates verifier-backed campaign data.
Recovery execution stays in the explicit Python/CLI workflow.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import collectors
from .capsule import (
    build_evidence_index,
    compare_to_last_success,
    find_similar_failures,
)
from .diff import compare_runs
from .export import export_capsule
from .storage import RECOVERY_RESULT_FILENAME, Storage


API_VERSION = "1.0"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "webstatic")


class RunUpdate(BaseModel):
    """Human labels only; recovery truth is not editable from the UI."""

    display_name: Optional[str] = Field(default=None, max_length=256)
    tags: Optional[list[str]] = None
    # Accepted only to return a clear compatibility error to older frontends.
    resolved: Optional[bool] = None
    resolved_note: Optional[str] = None


def _safe_json(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _row_get(row, name: str, default=None):
    if row is None:
        return default
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def _gpu_name(gpu_value) -> Optional[str]:
    gpu = _safe_json(gpu_value, {}) or {}
    devices = gpu.get("gpus") or []
    if not devices:
        return None
    return devices[0].get("name")


def _display_name(row) -> str:
    explicit = _row_get(row, "display_name")
    if explicit:
        return explicit
    config = _safe_json(row["config_json"], {}) or {}
    model = (
        config.get("model")
        or config.get("architecture")
        or config.get("model_name")
        or _nested(config, "model.name")
    )
    batch = config.get("batch_size") or _nested(
        config, "trainer.per_device_train_batch_size"
    )
    if model and batch is not None:
        return "{} — batch {}".format(model, batch)
    return str(model) if model else row["run_id"]


def _nested(payload: dict, dotted: str):
    current = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _failure_class(storage: Storage, run_id: str) -> Optional[str]:
    failure = storage.get_failure(run_id)
    if failure is None:
        return None
    return _row_get(failure, "failure_class") or (
        _safe_json(failure["diagnosis_json"], {}) or {}
    ).get("rule")


def _run_summary(storage: Storage, row) -> dict:
    config = _safe_json(row["config_json"], {}) or {}
    return {
        "run_id": row["run_id"],
        "display_name": _display_name(row),
        "project": row["project"],
        "status": row["exit_status"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "duration_seconds": row["duration_seconds"],
        "capture_completeness": _row_get(row, "capture_completeness"),
        # Kept as historical run metadata, never used as recovery proof.
        "reproduction_score": _row_get(row, "reproduction_score"),
        "config": config,
        "final_metrics": storage.final_metrics(row["run_id"]),
        "git_dirty": (_safe_json(row["git_json"], {}) or {}).get("dirty"),
        "hardware": _gpu_name(row["gpu_json"]) or "CPU only",
        "warning_count": len(_safe_json(row["warnings_json"], []) or []),
        "failure_category": _failure_class(storage, row["run_id"]),
        "tags": _safe_json(_row_get(row, "tags_json"), []) or [],
        "resolved": bool(_row_get(row, "resolved", 0)),
        "resolved_note": _row_get(row, "resolved_note"),
        "simulated": bool(config.get("_simulated")),
    }


def _campaign_summary(storage: Storage, row) -> dict:
    usage = _safe_json(_row_get(row, "usage_json"), {}) or {}
    verified = bool(_row_get(row, "verified", 0))
    status = _row_get(row, "status", "running") or "running"
    if verified:
        verification_status = "verified"
    elif row["ended_at"] is None or status == "running":
        verification_status = "pending"
    else:
        verification_status = "not_verified"
    return {
        "campaign_id": row["campaign_id"],
        "project": row["project"],
        "source_run_id": row["source_run_id"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "status": status,
        "stopped_reason": row["stopped_reason"],
        "verified": verified,
        "verification_status": verification_status,
        "verified_candidate_id": _row_get(row, "verified_candidate_id"),
        "verified_run_ids": _safe_json(
            _row_get(row, "verified_run_ids_json"), []
        )
        or [],
        "trial_count": int(usage.get("attempted_trials", 0)),
        "phase_counts": {
            "probe": int(usage.get("probe_trials", 0)),
            "full": int(usage.get("full_trials", 0)),
            "confirmation": int(usage.get("confirmation_trials", 0)),
        },
        "artifact_available": bool(_row_get(row, "artifact_path")),
    }


def _trial_payload(storage: Storage, row) -> dict:
    run = storage.get_run(row["run_id"])
    metrics = _safe_json(_row_get(row, "metrics_json"), {}) or {}
    if not metrics and run is not None:
        metrics = storage.final_metrics(row["run_id"])
    resource = (
        _safe_json(run["resource_json"], {}) if run is not None else {}
    ) or {}
    peak_bytes = _row_get(row, "peak_vram_bytes")
    if peak_bytes is None and resource.get("vram_used_mib_peak") is not None:
        peak_bytes = int(float(resource["vram_used_mib_peak"]) * 1024**2)
    return {
        "campaign_id": row["campaign_id"],
        "trial_id": _row_get(row, "trial_id"),
        "run_id": row["run_id"],
        "candidate_id": _row_get(row, "candidate_id")
        or _row_get(row, "proposal_id"),
        "proposal_id": _row_get(row, "proposal_id"),
        "policy_rule": _row_get(row, "policy_rule"),
        "phase": row["phase"],
        "status": _row_get(row, "status") or row["outcome"],
        "failure_class": _row_get(row, "failure_class"),
        "verified": bool(row["verified"]),
        "request_digest": _row_get(row, "request_digest"),
        "execution_manifest_digest": _row_get(
            row, "execution_manifest_digest"
        ),
        "worker_pid": _row_get(row, "worker_pid"),
        "duration_seconds": _row_get(row, "duration_seconds"),
        "gpu_seconds": _row_get(row, "gpu_seconds"),
        "progress_steps": _row_get(row, "progress_steps"),
        "peak_vram_bytes": peak_bytes,
        "peak_vram_gib": (
            round(peak_bytes / 1024**3, 3) if peak_bytes is not None else None
        ),
        "metrics": metrics,
        "workload_identity": _safe_json(
            _row_get(row, "workload_identity_json"), {}
        )
        or {},
        "config_patch": _safe_json(row["patch_json"], {}) or {},
        "environment_patch": _safe_json(
            _row_get(row, "environment_patch_json"), {}
        )
        or {},
        "rationale": row["rationale"],
        "run": _run_summary(storage, run) if run is not None else None,
    }


def _proposal_payload(row) -> dict:
    return {
        "campaign_id": row["campaign_id"],
        "proposal_id": row["proposal_id"],
        "policy_rule": row["policy_rule"],
        "authorization_mode": row["authorization_mode"],
        "state": row["state"],
        "skip_code": row["skip_code"],
        "skip_reason": row["skip_reason"],
        "rationale": row["rationale"],
        "proposal": _safe_json(row["proposal_json"], {}) or {},
    }


def _verification_payload(row) -> dict:
    return {
        "campaign_id": row["campaign_id"],
        "candidate_id": row["candidate_id"],
        "verified": bool(row["verified"]),
        "confirmation_run_ids": _safe_json(
            row["confirmation_run_ids_json"], []
        )
        or [],
        "report": _safe_json(row["report_json"], {}) or {},
        "ordinal": row["ordinal"],
    }


def _campaign_detail(storage: Storage, row) -> dict:
    campaign_id = row["campaign_id"]
    report = storage.get_recovery_campaign_report(campaign_id)
    contract = _safe_json(row["contract_json"], {}) or {}
    trials = [
        _trial_payload(storage, item)
        for item in storage.list_recovery_trials(campaign_id)
    ]
    proposals = [
        _proposal_payload(item)
        for item in storage.list_recovery_proposals(campaign_id)
    ]
    verifications = [
        _verification_payload(item)
        for item in storage.list_recovery_verifications(campaign_id)
    ]
    return {
        **_campaign_summary(storage, row),
        "contract": contract,
        "contract_digest": _row_get(row, "contract_digest"),
        "preparation_digest": _row_get(row, "preparation_digest"),
        "report_digest": _row_get(row, "report_digest"),
        "planned_candidate_ids": _safe_json(
            _row_get(row, "planned_candidate_ids_json"), []
        )
        or [],
        "probe_survivor_ids": _safe_json(
            _row_get(row, "probe_survivor_ids_json"), []
        )
        or [],
        "executed_proposal_ids": _safe_json(
            _row_get(row, "executed_proposal_ids_json"), []
        )
        or [],
        "skipped_proposals": _safe_json(
            _row_get(row, "skipped_proposals_json"), []
        )
        or [],
        "usage": _safe_json(_row_get(row, "usage_json"), {}) or {},
        "ranking": _safe_json(_row_get(row, "ranking_json"), None),
        "report": report,
        "trials": trials,
        "proposals": proposals,
        "verifications": verifications,
        "artifact": {
            "available": bool(_row_get(row, "artifact_path")),
            "path": _row_get(row, "artifact_path"),
            "checksum": _row_get(row, "artifact_checksum"),
            "size_bytes": _row_get(row, "artifact_size_bytes"),
            "download_url": (
                "/api/campaigns/{}/artifact".format(campaign_id)
                if _row_get(row, "artifact_path")
                else None
            ),
        },
    }


def _require_run(storage: Storage, run_id: str):
    row = storage.get_run(run_id)
    if row is None:
        raise HTTPException(404, "Run {!r} not found".format(run_id))
    return row


def _require_campaign(storage: Storage, campaign_id: str):
    row = storage.get_recovery_campaign(campaign_id)
    if row is None:
        raise HTTPException(
            404, "Campaign {!r} not found".format(campaign_id)
        )
    return row


def create_app(storage: Optional[Storage] = None) -> FastAPI:
    owns_storage = storage is None
    shared = Storage() if storage is None else storage
    app = FastAPI(
        title="WatcherML",
        version=API_VERSION,
        description="Local deterministic OOM evidence and recovery audit API",
    )
    app.state.storage = shared

    if owns_storage:
        @app.on_event("shutdown")
        def _close_owned_storage() -> None:
            shared.close()

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "api_version": API_VERSION,
            "storage_schema_version": shared.schema_version,
            "mode": "local_read_only_recovery_ui",
        }

    @app.get("/api/overview")
    def overview():
        runs = shared.list_runs()
        campaigns = shared.list_recovery_campaigns()
        projects = {row["project"] for row in runs if row["project"]}
        active_runs = [row for row in runs if row["exit_status"] == "running"]
        needs_attention = [
            row
            for row in runs
            if row["exit_status"] == "failed"
            and not bool(_row_get(row, "resolved", 0))
        ]
        active_campaigns = [
            row
            for row in campaigns
            if (_row_get(row, "status", "running") or "running") == "running"
        ]
        verified = [
            row for row in campaigns if bool(_row_get(row, "verified", 0))
        ][:5]
        gpu = collectors.collect_gpu_info()
        return {
            "project_count": len(projects),
            "run_count": len(runs),
            "active_run_count": len(active_runs),
            "failure_count": sum(row["exit_status"] == "failed" for row in runs),
            "runs_needing_attention": [
                _run_summary(shared, row) for row in needs_attention[:10]
            ],
            "active_campaign_count": len(active_campaigns),
            "verified_recovery_count": len(
                [row for row in campaigns if bool(_row_get(row, "verified", 0))]
            ),
            "recent_verified_recoveries": [
                _campaign_summary(shared, row) for row in verified
            ],
            # Compatibility alias for an older frontend; values are verifier-backed.
            "recent_verified_fixes": [
                {
                    "campaign_id": row["campaign_id"],
                    "project": row["project"],
                    "source_run_id": row["source_run_id"],
                    "verified_candidate_id": _row_get(
                        row, "verified_candidate_id"
                    ),
                    "verified_run_ids": _safe_json(
                        _row_get(row, "verified_run_ids_json"), []
                    )
                    or [],
                }
                for row in verified
            ],
            "gpu_available": bool(gpu.get("available", False)),
            "gpu_name": _gpu_name(gpu),
            "llm_required": False,
            "recovery_execution_surface": "sdk_or_cli",
        }

    @app.get("/api/projects")
    def list_projects():
        projects = {}
        for row in shared.list_runs():
            item = projects.setdefault(
                row["project"],
                {
                    "name": row["project"],
                    "run_count": 0,
                    "failure_count": 0,
                    "unresolved_failure_count": 0,
                    "last_started_at": None,
                },
            )
            item["run_count"] += 1
            if row["exit_status"] == "failed":
                item["failure_count"] += 1
                if not bool(_row_get(row, "resolved", 0)):
                    item["unresolved_failure_count"] += 1
            if (
                item["last_started_at"] is None
                or (row["started_at"] or 0) > item["last_started_at"]
            ):
                item["last_started_at"] = row["started_at"]
        return sorted(
            projects.values(),
            key=lambda item: item["last_started_at"] or 0,
            reverse=True,
        )

    @app.get("/api/projects/{project}/runs")
    def list_project_runs(project: str):
        rows = shared.list_runs(project=project)
        if not rows:
            raise HTTPException(
                404, "No runs found for project {!r}".format(project)
            )
        return [_run_summary(shared, row) for row in rows]

    @app.get("/api/runs")
    def list_runs(
        project: Optional[str] = None,
        status: Optional[str] = None,
        hardware: Optional[str] = None,
        failure_category: Optional[str] = None,
        unresolved: Optional[bool] = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        summaries = [
            _run_summary(shared, row)
            for row in shared.list_runs(project=project)
        ]
        if status:
            summaries = [item for item in summaries if item["status"] == status]
        if hardware:
            summaries = [
                item for item in summaries if item["hardware"] == hardware
            ]
        if failure_category:
            summaries = [
                item
                for item in summaries
                if item["failure_category"] == failure_category
            ]
        if unresolved is not None:
            summaries = [
                item for item in summaries if item["resolved"] is not unresolved
            ]
        return summaries[:limit]

    @app.patch("/api/runs/{run_id}")
    def update_run(run_id: str, update: RunUpdate):
        _require_run(shared, run_id)
        if update.resolved is not None or update.resolved_note is not None:
            raise HTTPException(
                409,
                "Recovery resolution is verifier-owned and cannot be edited from the UI",
            )
        if update.display_name is not None:
            shared.set_run_display_name(run_id, update.display_name or None)
        if update.tags is not None:
            cleaned = []
            for tag in update.tags:
                value = tag.strip()
                if not value or len(value) > 64:
                    raise HTTPException(
                        422, "Tags must be non-empty and at most 64 characters"
                    )
                if value not in cleaned:
                    cleaned.append(value)
            if len(cleaned) > 32:
                raise HTTPException(422, "At most 32 tags are allowed")
            shared.set_run_tags(run_id, cleaned)
        return _run_summary(shared, _require_run(shared, run_id))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        row = _require_run(shared, run_id)
        metrics = {}
        for point in shared.get_metrics(run_id):
            metrics.setdefault(point["name"], []).append(
                {
                    "step": point["step"],
                    "value": point["value"],
                    "timestamp": point["timestamp"],
                }
            )
        return {
            **_run_summary(shared, row),
            "git": _safe_json(row["git_json"], {}) or {},
            "env": _safe_json(row["env_json"], {}) or {},
            "gpu": _safe_json(row["gpu_json"], {}) or {},
            "resource_summary": _safe_json(row["resource_json"], {}) or {},
            "dataset_fingerprint": row["dataset_fingerprint"],
            "warnings": _safe_json(row["warnings_json"], []) or [],
            "metrics_over_time": metrics,
            "artifacts": [dict(item) for item in shared.get_artifacts(run_id)],
            "has_failure": shared.get_failure(run_id) is not None,
        }

    @app.get("/api/runs/{run_id}/samples")
    def get_samples(
        run_id: str,
        limit: int = Query(default=10_000, ge=1, le=100_000),
    ):
        _require_run(shared, run_id)
        return [
            dict(item) for item in shared.get_resource_samples(run_id)[:limit]
        ]

    @app.get("/api/runs/{run_id}/export")
    def export_run_capsule(run_id: str):
        _require_run(shared, run_id)
        descriptor, path = tempfile.mkstemp(
            prefix="watcher-run-{}-".format(run_id), suffix=".zip"
        )
        os.close(descriptor)
        try:
            export_capsule(shared, run_id, out_path=path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise
        return FileResponse(
            path,
            filename="watcher-run-{}.zip".format(run_id),
            media_type="application/zip",
            background=BackgroundTask(_remove_file, path),
        )

    @app.get("/api/runs/{run_id}/metrics.csv")
    def export_metrics_csv(run_id: str):
        _require_run(shared, run_id)
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(("step", "name", "value", "timestamp"))
        for row in shared.get_metrics(run_id):
            writer.writerow(
                (row["step"], row["name"], row["value"], row["timestamp"])
            )
        return Response(
            content=stream.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="{}_metrics.csv"'.format(
                    run_id
                )
            },
        )

    @app.get("/api/failures")
    def list_failures(
        project: Optional[str] = None,
        unresolved: Optional[bool] = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        output = []
        for failure in shared.list_failures(project=project):
            run = shared.get_run(failure["run_id"])
            item = {
                "run_id": failure["run_id"],
                "project": _row_get(failure, "project"),
                "message": failure["message"],
                "failure_class": _row_get(failure, "failure_class")
                or (_safe_json(failure["diagnosis_json"], {}) or {}).get("rule"),
                "rule": (_safe_json(failure["diagnosis_json"], {}) or {}).get(
                    "rule"
                ),
                "captured_at": _row_get(failure, "captured_at"),
                "capsule_schema_version": _row_get(
                    failure, "capsule_schema_version"
                ),
                "resolved": bool(_row_get(run, "resolved", 0)),
            }
            if unresolved is None or item["resolved"] is not unresolved:
                output.append(item)
        return output[:limit]

    @app.get("/api/runs/{run_id}/failure")
    def get_failure(run_id: str):
        run = _require_run(shared, run_id)
        failure = shared.get_failure(run_id)
        if failure is None:
            raise HTTPException(404, "Run {!r} did not fail".format(run_id))
        capsule = shared.get_failure_capsule(run_id) or {}
        evidence = capsule.get("evidence") or _safe_json(
            failure["evidence_json"], {}
        )
        diagnosis = (
            capsule.get("classification")
            or capsule.get("diagnosis")
            or _safe_json(failure["diagnosis_json"], {})
            or {}
        )
        failure_payload = capsule.get("failure") or {}
        evidence_index = capsule.get("evidence_index") or build_evidence_index(
            evidence
        )
        return {
            **capsule,
            "run_id": run_id,
            "display_name": _display_name(run),
            "exception_type": failure_payload.get("exception_type")
            or failure["exception_type"],
            "message": failure_payload.get("message") or failure["message"],
            "traceback": failure_payload.get("traceback")
            or failure["traceback"],
            "failure_class": capsule.get("failure_class")
            or failure_payload.get("class")
            or _row_get(failure, "failure_class")
            or diagnosis.get("rule"),
            "diagnosis": diagnosis,
            "evidence": evidence,
            "evidence_index": evidence_index,
            "similar_previous_failures": find_similar_failures(
                shared,
                run["project"],
                run_id,
                diagnosis.get("rule", ""),
            ),
            "comparison_to_last_success": compare_to_last_success(
                shared, run["project"], run_id
            ),
            "simulated": bool((evidence.get("config") or {}).get("_simulated")),
            "resolved": bool(_row_get(run, "resolved", 0)),
            "resolved_note": _row_get(run, "resolved_note"),
        }

    @app.get("/api/compare")
    def compare(a: str, b: str):
        try:
            return compare_runs(shared, a, b)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/campaigns")
    def list_campaigns(
        project: Optional[str] = None,
        status: Optional[str] = None,
        verified: Optional[bool] = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        rows = shared.list_recovery_campaigns(
            project=project, status=status, verified=verified
        )
        return [_campaign_summary(shared, row) for row in rows[:limit]]

    @app.get("/api/campaigns/{campaign_id}")
    def get_campaign(campaign_id: str):
        return _campaign_detail(
            shared, _require_campaign(shared, campaign_id)
        )

    @app.get("/api/campaigns/{campaign_id}/report")
    def get_campaign_report(campaign_id: str):
        _require_campaign(shared, campaign_id)
        report = shared.get_recovery_campaign_report(campaign_id)
        if report is None:
            raise HTTPException(404, "Campaign report is not available yet")
        return report

    @app.get("/api/campaigns/{campaign_id}/trials")
    def get_campaign_trials(campaign_id: str):
        _require_campaign(shared, campaign_id)
        return [
            _trial_payload(shared, row)
            for row in shared.list_recovery_trials(campaign_id)
        ]

    @app.get("/api/campaigns/{campaign_id}/proposals")
    def get_campaign_proposals(campaign_id: str):
        _require_campaign(shared, campaign_id)
        return [
            _proposal_payload(row)
            for row in shared.list_recovery_proposals(campaign_id)
        ]

    @app.get("/api/campaigns/{campaign_id}/verifications")
    def get_campaign_verifications(campaign_id: str):
        _require_campaign(shared, campaign_id)
        return [
            _verification_payload(row)
            for row in shared.list_recovery_verifications(campaign_id)
        ]

    @app.get("/api/campaigns/{campaign_id}/artifact")
    def download_campaign_artifact(campaign_id: str):
        row = _require_campaign(shared, campaign_id)
        path_value = _row_get(row, "artifact_path")
        if not path_value:
            raise HTTPException(404, "Recovery result artifact is not available")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(404, "Recovery result artifact is missing on disk")
        expected_size = _row_get(row, "artifact_size_bytes")
        payload = path.read_bytes()
        if expected_size is not None and len(payload) != expected_size:
            raise HTTPException(409, "Recovery result artifact size check failed")
        expected_checksum = _row_get(row, "artifact_checksum")
        checksum = hashlib.sha256(payload).hexdigest()
        if expected_checksum and checksum != expected_checksum:
            raise HTTPException(409, "Recovery result artifact checksum failed")
        return FileResponse(
            path,
            filename="{}-{}".format(campaign_id, RECOVERY_RESULT_FILENAME),
            media_type="application/json",
            headers={"X-WatcherML-SHA256": checksum},
        )

    @app.get("/api/memory")
    def resolution_memory(project: Optional[str] = None):
        return shared.resolution_memory(project=project)

    @app.get("/api/settings")
    def settings():
        gpu = collectors.collect_gpu_info()
        return {
            "data_directory": shared.root,
            "database": shared.db_path,
            "database_path": shared.db_path,
            "storage_schema_version": shared.schema_version,
            "gpu": gpu,
            "gpu_available": bool(gpu.get("available", False)),
            "llm_required": False,
            "recovery_execution_surface": "sdk_or_cli",
            "trial_isolation": "fresh_subprocess",
            "web_recovery_mutations_enabled": False,
        }

    @app.get("/")
    def index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.isfile(index_path):
            raise HTTPException(404, "WatcherML web assets are not installed")
        return FileResponse(index_path)

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _remove_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass