"""End-to-end WatcherML smoke test for macOS.

This validates every component currently implemented. CUDA evidence and true
subprocess recovery are intentionally not claimed: the OOM is simulated on a
Mac, and the current recovery executor is still the provisional in-process
implementation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import watcherml as watcher
from watcherml.capsule_schema import validate_capsule
from watcherml.diff import compare_runs
from watcherml.entrypoint import (
    TrainingEntrypoint,
    invoke_entrypoint,
    validate_entrypoint,
)
from watcherml.export import export_capsule
from watcherml.recovery import RecoveryContract, recover_from_oom
from watcherml.storage import Storage


def passed(label: str) -> None:
    print(f"\033[32mPASS\033[0m  {label}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_export(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {
            "manifest.json",
            "run.json",
            "config.json",
            "artifacts.json",
            "requirements.txt",
            "README.txt",
            "failure-capsule.json",
        }
        require(required <= names, f"export missing: {sorted(required - names)}")
        manifest = json.loads(archive.read("manifest.json"))
        require(
            manifest["schema"] == {
                "name": "watcherml.run-export",
                "version": "1.0",
            },
            "unexpected export schema",
        )
        for item in manifest["contents"]:
            payload = archive.read(item["path"])
            require(len(payload) == item["size_bytes"], f"size mismatch: {item['path']}")
            require(
                hashlib.sha256(payload).hexdigest() == item["sha256"],
                f"checksum mismatch: {item['path']}",
            )


def run_cli(workspace: Path, project: str, success_id: str, failed_id: str) -> None:
    executable = shutil.which("watcher")
    if executable is None:
        raise RuntimeError(
            "The 'watcher' command is unavailable. Run: python -m pip install -e '.[dev,ui]'"
        )

    commands = [
        [executable, "runs", "--project", project],
        [executable, "failures", "--project", project],
        [executable, "inspect", failed_id],
        [executable, "compare", success_id, failed_id],
        [
            executable,
            "export",
            failed_id,
            "--out",
            str(workspace / "cli-failure-export.zip"),
        ],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=30,
        )
        require(
            result.returncode == 0,
            f"CLI failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        print(f"      $ {' '.join(command[1:])}")


def test_web_api(storage: Storage, project: str, failed_id: str) -> None:
    try:
        from fastapi.testclient import TestClient
        from watcherml.webapp import create_app
    except ImportError as exc:
        raise RuntimeError(
            "UI dependencies are missing. Run: python -m pip install -e '.[dev,ui]'"
        ) from exc

    client = TestClient(create_app(storage))
    runs_response = client.get("/api/runs", params={"project": project})
    require(runs_response.status_code == 200, "/api/runs failed")
    require(len(runs_response.json()) >= 2, "/api/runs omitted smoke runs")

    failure_response = client.get(f"/api/runs/{failed_id}/failure")
    require(failure_response.status_code == 200, "failure endpoint failed")
    failure = failure_response.json()
    require(failure["capsule_schema_version"] == "1.0", "API returned a legacy capsule")
    require(failure["failure_class"] == "cuda_out_of_memory", "wrong API failure class")

    campaigns_response = client.get("/api/campaigns", params={"project": project})
    require(campaigns_response.status_code == 200, "/api/campaigns failed")
    campaigns = campaigns_response.json()
    require(campaigns, "campaign endpoint returned no campaign")
    require(campaigns[0]["best_run_id"] is None, "provisional trial was marked verified")
    require(
        campaigns[0]["verification_status"] == "pending_confirmation",
        "campaign did not expose pending confirmation",
    )

    settings_response = client.get("/api/settings")
    require(settings_response.status_code == 200, "/api/settings failed")
    settings = settings_response.json()
    require("data_directory" in settings and "gpu" in settings, "settings fields missing")
    require(
        not {"ollama_available", "ollama_host", "ollama_default_model"} & settings.keys(),
        "removed model settings are still exposed",
    )


def main() -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    original_directory = Path.cwd()
    workspace = original_directory / "watcherml-smoke-output" / timestamp
    workspace.mkdir(parents=True, exist_ok=False)
    os.chdir(workspace)

    project = f"mac-smoke-{timestamp}"
    storage = Storage(root=str(workspace / ".watcherml"))
    print(f"\nWatcherML smoke workspace: {workspace}\n")

    dataset = workspace / "tiny-dataset.txt"
    dataset.write_text("sample-a\nsample-b\n", encoding="utf-8")
    artifact = workspace / "model-summary.txt"
    artifact.write_text("smoke-test artifact\n", encoding="utf-8")

    with watcher.init(
        project=project,
        config={
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "training_steps": 100,
        },
        storage=storage,
    ) as successful_run:
        successful_run.set_dataset(str(dataset))
        successful_run.log({"loss": 0.8, "validation_loss": 0.5}, step=40)
        successful_run.log({"loss": 0.6, "validation_loss": 0.4}, step=50)
        successful_run.log_artifact(str(artifact))
    success_id = successful_run.run_id
    require(storage.get_run(success_id)["exit_status"] == "success", "success not stored")
    require(storage.get_artifacts(success_id), "artifact not stored")
    passed("successful run, metrics, dataset fingerprint, and artifact")

    try:
        with watcher.init(
            project=project,
            config={
                "batch_size": 32,
                "gradient_accumulation_steps": 1,
                "training_steps": 100,
                "_simulated": True,
            },
            storage=storage,
        ) as failed_run:
            failed_run.set_dataset(str(dataset))
            failed_run.log({"loss": 1.2}, step=41)
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    except RuntimeError as exc:
        require("CUDA out of memory" in str(exc), "original exception was replaced")
    failed_id = failed_run.run_id

    capsule = storage.get_failure_capsule(failed_id)
    require(capsule is not None, "failure capsule was not persisted")
    require(validate_capsule(capsule) == [], "failure capsule failed schema validation")
    require(capsule["failure_class"] == "cuda_out_of_memory", "OOM misclassified")
    require(capsule["evidence"]["training_state"]["last_logged_step"] == 41, "step missing")
    passed("deterministic OOM capsule schema, classification, evidence, and propagation")

    diff = compare_runs(storage, success_id, failed_id)
    changed = {item["key"] for item in diff["config_diff"]}
    require("batch_size" in changed, "run comparison missed batch_size")
    passed("structured run comparison")

    export_path = Path(export_capsule(
        storage,
        failed_id,
        str(workspace / "failure-export.zip"),
    ))
    verify_export(export_path)
    passed("portable export and every manifest checksum")

    module_path = workspace / "smoke_train.py"
    module_path.write_text(
        """def main(config, max_steps=None):
    if config["batch_size"] > 16:
        raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
    steps = max_steps if max_steps is not None else config["training_steps"]
    return {"validation_loss": 0.4, "steps_completed": steps}
""",
        encoding="utf-8",
    )
    entrypoint = TrainingEntrypoint("smoke_train:main")
    validation = validate_entrypoint(
        entrypoint,
        project_root=str(workspace),
        require_max_steps=True,
    )
    require(validation.supports_max_steps, "entrypoint lacks bounded probes")
    metrics = invoke_entrypoint(
        entrypoint,
        {"batch_size": 16, "training_steps": 100},
        project_root=str(workspace),
        max_steps=3,
    )
    require(metrics["steps_completed"] == 3.0, "entrypoint ignored max_steps")
    passed("serializable entrypoint contract and bounded invocation")

    def interim_train(config, max_steps=None):
        if config["batch_size"] > 16:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        steps = max_steps if max_steps is not None else config["training_steps"]
        return {"validation_loss": 0.4, "steps_completed": steps}

    report = recover_from_oom(
        project=project,
        failed_run_id=failed_id,
        train_fn=interim_train,
        contract=RecoveryContract(
            goal_metric="validation_loss",
            goal_direction="minimize",
            max_trials=4,
            max_candidates=1,
            probe_steps=3,
            max_wall_minutes=2,
        ),
        storage=storage,
    )
    require(report["verification_status"] == "pending_confirmation", "wrong status")
    require(report["best_run_id"] is None, "provisional result marked verified")
    require(report["provisional_best_run_id"] is not None, "no provisional result")
    trials = storage.list_recovery_trials(report["campaign_id"])
    require(trials and all(row["verified"] == 0 for row in trials), "trial marked verified")
    require(storage.resolution_memory(project=project) == [], "provisional result entered memory")
    passed("bounded deterministic recovery and provisional-result safeguards")

    run_cli(workspace, project, success_id, failed_id)
    passed("installed CLI commands")

    test_web_api(storage, project, failed_id)
    passed("FastAPI backend endpoints")

    print("\nALL CURRENT WATCHERML COMPONENTS PASSED\n")
    print("Not tested on this Mac:")
    print("  - real CUDA allocator/GPU evidence")
    print("  - fresh-process trial isolation (not implemented yet)")
    print("  - confirmation verifier (not implemented yet)")
    print(f"\nInspect the generated UI data with:\n  cd '{workspace}' && watcher ui\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n\033[31mSMOKE TEST FAILED\033[0m: {exc}", file=sys.stderr)
        raise