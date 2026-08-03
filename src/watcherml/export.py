"""Reproduction capsules: a portable bundle to re-run a specific experiment elsewhere.

References + checksums by default — never silently duplicates large datasets
or checkpoints.
"""
from __future__ import annotations

import json
import zipfile
from typing import Optional

from .storage import Storage


def export_capsule(storage: Storage, run_id: str, out_path: Optional[str] = None) -> str:
    row = storage.get_run(run_id)
    if row is None:
        raise ValueError(f"Run {run_id} not found.")

    git = json.loads(row["git_json"] or "{}")
    env = json.loads(row["env_json"] or "{}")
    config = json.loads(row["config_json"] or "{}")
    artifacts = [dict(a) for a in storage.get_artifacts(run_id)]

    manifest = {
        "run_id": run_id,
        "project": row["project"],
        "config": config,
        "seeds": config.get("seed"),
        "dataset_fingerprint": row["dataset_fingerprint"],
        "dataset_retrieval_instructions": (
            "This capsule stores a fingerprint, not the dataset itself. "
            "Place a dataset matching this fingerprint at the path used in set_dataset(), "
            "or retrieve it via your team's data versioning system (e.g. DVC)."
        ),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_was_dirty": git.get("dirty"),
        "python_version": env.get("python_version"),
        "package_count": env.get("package_count"),
        "artifacts": [{"path": a["path"], "checksum": a["checksum"], "size_bytes": a["size_bytes"]}
                      for a in artifacts],
        "reproduction_command": (
            f"git checkout {git.get('commit')} && "
            f"git apply run_{run_id}.patch  # if the run had uncommitted changes\n"
            f"pip install -r requirements_{run_id}.txt\n"
            f"python your_training_script.py  # re-run with the captured config"
        ),
    }

    out_path = out_path or f"watcher-run-{run_id}.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr(f"config_{run_id}.json", json.dumps(config, indent=2))
        if git.get("diff_patch"):
            zf.writestr(f"run_{run_id}.patch", git["diff_patch"])
        pkgs = env.get("packages", {})
        req_lines = [f"{name}=={version}" for name, version in sorted(pkgs.items())]
        zf.writestr(f"requirements_{run_id}.txt", "\n".join(req_lines))

    return out_path
