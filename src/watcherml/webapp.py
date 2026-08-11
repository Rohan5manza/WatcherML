"""Local web UI backend. Serves the WatcherML app (Overview, Projects, Runs,
Failures, Campaigns, Memory, Settings) and a JSON API reading from the same
SQLite storage the CLI uses. No Postgres, no Docker -- this is local mode.

Started via `watcher ui`.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import collectors
from .capsule import build_evidence_index, compare_to_last_success, find_similar_failures
from .diff import compare_runs
from .export import export_capsule
from .storage import Storage

STATIC_DIR = os.path.join(os.path.dirname(__file__), "webstatic")


def _safe_json(s, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _gpu_name(gpu_json) -> Optional[str]:
    gpu_info = _safe_json(gpu_json, {}) or {}
    gpus = gpu_info.get("gpus") or []
    return gpus[0]["name"] if gpus else None


def _display_name(row) -> str:
    """Human-readable name: explicit user-set name, else a heuristic built
    from config (e.g. 'resnet50 -- batch 32'), else just the run_id."""
    if row["display_name"]:
        return row["display_name"]
    config = _safe_json(row["config_json"], {}) or {}
    model = config.get("model") or config.get("architecture") or config.get("model_name")
    batch = config.get("batch_size")
    if model and batch:
        return f"{model} \u2014 batch {batch}"
    if model:
        return str(model)
    return row["run_id"]


def _row_to_run_summary(storage: Storage, row) -> dict:
    diagnosis = None
    failure = storage.get_failure(row["run_id"]) if row["exit_status"] == "failed" else None
    if failure:
        diagnosis = (_safe_json(failure["diagnosis_json"], {}) or {}).get("rule")
    config = _safe_json(row["config_json"], {}) or {}
    return {
        "run_id": row["run_id"],
        "display_name": _display_name(row),
        "project": row["project"],
        "status": row["exit_status"],
        "started_at": row["started_at"],
        "duration_seconds": row["duration_seconds"],
        "reproduction_score": row["reproduction_score"],
        "config": config,
        "final_metrics": storage.final_metrics(row["run_id"]),
        "git_dirty": (_safe_json(row["git_json"], {}) or {}).get("dirty"),
        "hardware": _gpu_name(row["gpu_json"]) or "CPU only",
        "warning_count": len(_safe_json(row["warnings_json"], []) or []),
        "failure_category": diagnosis,
        "tags": _safe_json(row["tags_json"], []) or [],
        "resolved": bool(row["resolved"]) if row["resolved"] is not None else False,
        "simulated": bool(config.get("_simulated")),
    }


class RunUpdate(BaseModel):
    display_name: Optional[str] = None
    tags: Optional[list] = None
    resolved: Optional[bool] = None
    resolved_note: Optional[str] = None


def create_app(storage: Optional[Storage] = None) -> FastAPI:
    storage = storage or Storage()
    app = FastAPI(title="WatcherML")

    # -- overview ------------------------------------------------------
    @app.get("/api/overview")
    def overview():
        all_runs = storage.list_runs()
        projects = {r["project"] for r in all_runs}
        active = [r for r in all_runs if r["exit_status"] == "running"]
        needs_attention = [
            r for r in all_runs
            if r["exit_status"] == "failed" and not r["resolved"]
        ]
        campaigns = storage.list_recovery_campaigns()
        active_campaigns = [c for c in campaigns if c["ended_at"] is None]
        recent_verified = [c for c in campaigns if c["best_run_id"]][:5]
        gpu_info = collectors.collect_gpu_info()
        return {
            "project_count": len(projects),
            "run_count": len(all_runs),
            "active_run_count": len(active),
            "runs_needing_attention": [_row_to_run_summary(storage, r) for r in needs_attention[:10]],
            "active_campaign_count": len(active_campaigns),
            "recent_verified_fixes": [
                {"campaign_id": c["campaign_id"], "project": c["project"],
                 "best_run_id": c["best_run_id"], "source_run_id": c["source_run_id"]}
                for c in recent_verified
            ],
            "gpu_available": gpu_info.get("available", False),
            "gpu_name": _gpu_name(json.dumps(gpu_info)),
            
        }

    # -- projects ------------------------------------------------------
    @app.get("/api/projects")
    def list_projects():
        rows = storage.list_runs()
        projects: dict = {}
        for row in rows:
            p = projects.setdefault(row["project"], {
                "name": row["project"], "run_count": 0, "failure_count": 0,
                "last_started_at": None,
            })
            p["run_count"] += 1
            if row["exit_status"] == "failed":
                p["failure_count"] += 1
            if p["last_started_at"] is None or (row["started_at"] or 0) > p["last_started_at"]:
                p["last_started_at"] = row["started_at"]
        return sorted(projects.values(), key=lambda p: p["last_started_at"] or 0, reverse=True)

    @app.get("/api/projects/{project}/runs")
    def list_project_runs(project: str):
        rows = storage.list_runs(project=project)
        if not rows:
            raise HTTPException(404, f"No runs found for project '{project}'")
        return [_row_to_run_summary(storage, r) for r in rows]

    # -- global runs (cross-project, filterable) ------------------------
    @app.get("/api/runs")
    def list_all_runs(project: Optional[str] = None, status: Optional[str] = None,
                       hardware: Optional[str] = None, failure_category: Optional[str] = None):
        rows = storage.list_runs(project=project)
        summaries = [_row_to_run_summary(storage, r) for r in rows]
        if status:
            summaries = [s for s in summaries if s["status"] == status]
        if hardware:
            summaries = [s for s in summaries if s["hardware"] == hardware]
        if failure_category:
            summaries = [s for s in summaries if s["failure_category"] == failure_category]
        return summaries

    @app.patch("/api/runs/{run_id}")
    def update_run(run_id: str, update: RunUpdate):
        if storage.get_run(run_id) is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        if update.display_name is not None:
            storage.set_run_display_name(run_id, update.display_name)
        if update.tags is not None:
            storage.set_run_tags(run_id, update.tags)
        if update.resolved is not None:
            storage.set_run_resolved(run_id, update.resolved, update.resolved_note)
        return _row_to_run_summary(storage, storage.get_run(run_id))

    # -- run detail ------------------------------------------------------
    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        row = storage.get_run(run_id)
        if row is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        metrics_over_time: dict = {}
        for m in storage.get_metrics(run_id):
            metrics_over_time.setdefault(m["name"], []).append(
                {"step": m["step"], "value": m["value"], "timestamp": m["timestamp"]})
        return {
            **_row_to_run_summary(storage, row),
            "ended_at": row["ended_at"],
            "git": _safe_json(row["git_json"], {}),
            "env": _safe_json(row["env_json"], {}),
            "gpu": _safe_json(row["gpu_json"], {}),
            "resource_summary": _safe_json(row["resource_json"], {}),
            "dataset_fingerprint": row["dataset_fingerprint"],
            "warnings": _safe_json(row["warnings_json"], []),
            "resolved_note": row["resolved_note"],
            "metrics_over_time": metrics_over_time,
            "artifacts": [dict(a) for a in storage.get_artifacts(run_id)],
            "has_failure": storage.get_failure(run_id) is not None,
        }

    @app.get("/api/runs/{run_id}/samples")
    def get_samples(run_id: str):
        if storage.get_run(run_id) is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        return [dict(s) for s in storage.get_resource_samples(run_id)]

    @app.get("/api/runs/{run_id}/export")
    def export_run_capsule(run_id: str):
        if storage.get_run(run_id) is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        out_path = os.path.join(tempfile.gettempdir(), f"watcher-run-{run_id}.zip")
        export_capsule(storage, run_id, out_path=out_path)
        return FileResponse(out_path, filename=f"watcher-run-{run_id}.zip",
                             media_type="application/zip")

    @app.get("/api/runs/{run_id}/metrics.csv")
    def export_metrics_csv(run_id: str):
        """Plain CSV of every logged metric point -- step, name, value,
        timestamp -- for import into a spreadsheet or another tool. This is
        the run's metrics only; use /export for the full reproduction capsule."""
        if storage.get_run(run_id) is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        rows = storage.get_metrics(run_id)
        lines = ["step,name,value,timestamp"]
        for r in rows:
            lines.append(f"{r['step']},{r['name']},{r['value']},{r['timestamp']}")
        csv_text = "\n".join(lines)
        return Response(
            content=csv_text, media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{run_id}_metrics.csv"'},
        )

    # -- failure capsule ------------------------------------------------
    @app.get("/api/failures")
    def list_failures(project: Optional[str] = None):
        rows = storage.list_failures(project=project)
        out = []
        for row in rows:
            out.append({
                "run_id": row["run_id"],
                "project": row["project"],
                "message": row["message"],
                "rule": (_safe_json(row["diagnosis_json"], {}) or {}).get("rule"),
            })
        return out

    @app.get("/api/runs/{run_id}/failure")
    def get_failure(run_id: str):
        row = storage.get_run(run_id)
        if row is None:
            raise HTTPException(404, f"Run '{run_id}' not found")

        capsule = storage.get_failure_capsule(run_id)
        if capsule is None:
            raise HTTPException(404, f"Run '{run_id}' did not fail")

        evidence = capsule.get("evidence") or {}

        # Compatibility for failures recorded before capsule schema v1.
        if capsule.get("capsule_schema_version") == "legacy":
            classification = (
                capsule.get("classification")
                or capsule.get("diagnosis")
                or {}
            )

            nearest_success = compare_to_last_success(
                storage,
                row["project"],
                run_id,
            )

            capsule = {
                **capsule,
                "evidence_index": build_evidence_index(evidence),
                "similar_previous_failures": find_similar_failures(
                    storage,
                    row["project"],
                    run_id,
                    classification.get("rule", ""),
                ),
                "nearest_successful_run": nearest_success,
                "comparison_to_last_success": nearest_success,
            }

        return {
            **capsule,
            "display_name": _display_name(row),
            "simulated": bool(
                (evidence.get("config") or {}).get("_simulated")
            ),
            "resolved": (
                bool(row["resolved"])
                if row["resolved"] is not None
                else False
            ),
        }

    # -- comparison ------------------------------------------------------
    @app.get("/api/compare")
    def compare(a: str, b: str):
        try:
            return compare_runs(storage, a, b)
        except ValueError as e:
            raise HTTPException(404, str(e))

    # -- campaigns --------------------------------------------------------
    @app.get("/api/campaigns")
    def list_campaigns(project: Optional[str] = None):
        rows = storage.list_recovery_campaigns(project=project)
        out = []
        for row in rows:
            trials = storage.list_recovery_trials(row["campaign_id"])
            out.append({
                "campaign_id": row["campaign_id"],
                "project": row["project"],
                "source_run_id": row["source_run_id"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "stopped_reason": row["stopped_reason"],
                "best_run_id": row["best_run_id"],
                "trial_count": len(trials),
                "status": "active" if row["ended_at"] is None else "stopped",
            })
        return out

    @app.get("/api/campaigns/{campaign_id}")
    def get_campaign(campaign_id: str):
        row = storage.get_recovery_campaign(campaign_id)
        if row is None:
            raise HTTPException(404, f"Campaign '{campaign_id}' not found")
        contract = _safe_json(row["contract_json"], {})
        goal_metric = contract.get("goal_metric")
        trials = storage.list_recovery_trials(campaign_id)
        trial_list = []
        for t in trials:
            trial_run = storage.get_run(t["run_id"])
            final_metrics = storage.final_metrics(t["run_id"]) if trial_run else {}
            resource_summary = _safe_json(trial_run["resource_json"], {}) if trial_run else {}
            vram_peak_mib = (resource_summary or {}).get("vram_used_mib_peak")
            objective_value = final_metrics.get(goal_metric) if goal_metric else None
            result_summary = (
                f"{goal_metric} {objective_value:.4g}" if objective_value is not None
                else t["outcome"] if t["outcome"] != "success" else None
            )
            trial_list.append({
                "run_id": t["run_id"],
                "phase": t["phase"],
                "patch": _safe_json(t["patch_json"], {}),
                "rationale": t["rationale"],
                "confidence": t["confidence"],
                "outcome": t["outcome"],
                "score": t["score"],
                "verified": bool(t["verified"]),
                "final_metrics": final_metrics,
                "peak_vram_gb": round(vram_peak_mib / 1024, 2) if vram_peak_mib else None,
                "objective_value": objective_value,
                "result_summary": result_summary,
            })
        return {
            "campaign_id": row["campaign_id"],
            "project": row["project"],
            "source_run_id": row["source_run_id"],
            "contract": contract,
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "stopped_reason": row["stopped_reason"],
            "best_run_id": row["best_run_id"],
            "report": _safe_json(row["report_json"], {}),
            "trials": trial_list,
        }

    # -- resolution memory -------------------------------------------------
    @app.get("/api/memory")
    def memory(project: Optional[str] = None):
        return storage.resolution_memory(project=project)

    # -- settings ------------------------------------------------------
    @app.get("/api/settings")
    def settings():
        return {
            "data_directory": storage.root,
            "database_path": storage.db_path,
            "gpu": collectors.collect_gpu_info(),
        }

    # -- static frontend ------------------------------------------------
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app