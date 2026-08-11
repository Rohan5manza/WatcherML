"""Portable, checksum-verifiable WatcherML run exports."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from typing import Optional

from .storage import Storage


EXPORT_SCHEMA_NAME = "watcherml.run-export"
EXPORT_SCHEMA_VERSION = "1.0"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def export_capsule(storage: Storage, run_id: str, out_path: Optional[str] = None) -> str:
    """Export captured evidence and references, never dataset/checkpoint bytes."""
    row = storage.get_run(run_id)
    if row is None:
        raise ValueError(f"Run {run_id} not found.")

    git = _json_load(row["git_json"])
    env = _json_load(row["env_json"])
    config = _json_load(row["config_json"])
    artifacts = [
        {"path": item["path"], "checksum": item["checksum"],
         "size_bytes": item["size_bytes"]}
        for item in storage.get_artifacts(run_id)
    ]
    failure_capsule = storage.get_failure_capsule(run_id)

    run_summary = {
        "run_id": run_id,
        "project": row["project"],
        "status": row["exit_status"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "duration_seconds": row["duration_seconds"],
        "dataset_fingerprint": row["dataset_fingerprint"],
        "git": {key: value for key, value in git.items() if key != "diff_patch"},
        "environment": {
            "python_version": env.get("python_version"),
            "platform": env.get("platform"),
            "package_count": env.get("package_count"),
            "fingerprint": env.get("fingerprint"),
        },
        "artifacts_are_references_only": True,
    }

    payloads: dict[str, bytes] = {
        "run.json": _json_bytes(run_summary),
        "config.json": _json_bytes(config),
        "artifacts.json": _json_bytes(artifacts),
        "requirements.txt": _requirements_bytes(env.get("packages") or {}),
    }
    if failure_capsule is not None:
        payloads["failure-capsule.json"] = _json_bytes(failure_capsule)
    if git.get("diff_patch"):
        payloads["working-tree.patch"] = str(git["diff_patch"]).encode("utf-8")

    payloads["README.txt"] = _readme(run_summary, git, failure_capsule).encode("utf-8")
    content_manifest = [
        {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
        for path, data in sorted(payloads.items())
    ]
    manifest = {
        "schema": {"name": EXPORT_SCHEMA_NAME, "version": EXPORT_SCHEMA_VERSION},
        "run_id": run_id,
        "project": row["project"],
        "failure_capsule_schema_version": (
            failure_capsule.get("capsule_schema_version") if failure_capsule else None),
        "contents": content_manifest,
        "notes": [
            "Dataset and checkpoint bytes are not embedded.",
            "Artifact entries are references with captured checksums.",
            "Verify every payload against manifest.json before using it.",
        ],
    }

    out_path = out_path or f"watcher-run-{run_id}.zip"
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(out_path, "w") as archive:
        _write_zip_member(archive, "manifest.json", _json_bytes(manifest))
        for path, data in sorted(payloads.items()):
            _write_zip_member(archive, path, data)
    return out_path


def _json_load(value: Optional[str]) -> dict:
    try:
        return json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _requirements_bytes(packages: dict) -> bytes:
    lines = [f"{name}=={version}" for name, version in sorted(packages.items(), key=lambda x: x[0].lower())]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _write_zip_member(archive: zipfile.ZipFile, path: str, data: bytes) -> None:
    info = zipfile.ZipInfo(path, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def _readme(run: dict, git: dict, failure_capsule: Optional[dict]) -> str:
    lines = [
        "WatcherML run export",
        "====================",
        "",
        f"Run: {run['run_id']}",
        f"Project: {run['project']}",
        f"Status: {run['status']}",
        "",
        "This bundle contains captured evidence and references. It does not claim",
        "that the run is reproducible until the dataset, code, entrypoint, and",
        "hardware constraints have been supplied and a rerun has been verified.",
        "",
        "Suggested reconstruction steps:",
    ]
    step = 1
    if git.get("commit"):
        lines.append(f"{step}. Check out git commit {git['commit']}.")
        step += 1
    if git.get("diff_patch"):
        lines.append(f"{step}. Review and apply working-tree.patch.")
        step += 1
    lines.extend([
        f"{step}. Create an isolated environment and install requirements.txt.",
        f"{step + 1}. Restore a dataset matching the fingerprint in run.json.",
        f"{step + 2}. Supply the training entrypoint and captured config.json.",
        f"{step + 3}. Rerun in a fresh process and compare the declared verifier outputs.",
    ])
    if failure_capsule:
        lines.extend([
            "",
            "Failure evidence is in failure-capsule.json. Its classification is",
            "deterministic; it contains no LLM-generated diagnosis.",
        ])
    return "\n".join(lines) + "\n"
